// m2v-run: run one compute entry point from a SPIR-V file on the first Vulkan device.
// usage: m2v-run SPV ENTRY GX GY GZ [--iters N] [--local X Y Z] [--push FILE] [--buf FILE]... [--dump IDX:FILE]...
// Buffers are bound to set 0, bindings 0..n-1 in the order given (storage buffers). Prints GPU time per dispatch
// from timestamp queries around N back-to-back dispatches. Build: cc -O2 m2v-run.c -lvulkan -o m2v-run
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

// Most storage buffers one dispatch can bind (the fused oMLX kernels take 17 and more).
#define M2V_MAX_BUFS 64

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s failed: %d (line %d)\n", #x, r_, __LINE__); exit(2); } } while (0)

static void* readf(const char* p, size_t* n) {
  FILE* f = fopen(p, "rb"); if (!f) { perror(p); exit(3); }
  fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
  void* b = malloc(*n ? *n : 1); if (fread(b, 1, *n, f) != *n) exit(3); fclose(f); return b;
}

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "usage: m2v-run SPV ENTRY GX GY GZ [--iters N] [--push F] [--buf F]... [--dump I:F]...\n"); return 1; }
  const char* spv = argv[1]; const char* entry = argv[2];
  uint32_t g[3] = { atoi(argv[3]), atoi(argv[4]), atoi(argv[5]) };
  int iters = 1; uint32_t local[3] = {0, 0, 0}; const char* pushf = NULL; const char* bufs[M2V_MAX_BUFS]; int nb = 0; const char* dumps[M2V_MAX_BUFS]; int nd = 0;
  for (int i = 6; i < argc; i++) {
    if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--local")) { local[0] = atoi(argv[++i]); local[1] = atoi(argv[++i]); local[2] = atoi(argv[++i]); }
    else if (!strcmp(argv[i], "--push")) pushf = argv[++i];
    else if (!strcmp(argv[i], "--buf")) { if (nb >= M2V_MAX_BUFS) { fprintf(stderr, "m2v-run: more than %d buffers\n", M2V_MAX_BUFS); return 2; } bufs[nb++] = argv[++i]; }
    else if (!strcmp(argv[i], "--dump")) { if (nd >= M2V_MAX_BUFS) { fprintf(stderr, "m2v-run: more than %d dumps\n", M2V_MAX_BUFS); return 2; } dumps[nd++] = argv[++i]; }
  }
  VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "m2v", 1, "m2v", 1, VK_API_VERSION_1_3 };
  VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &ai };
  VkInstance inst; CK(vkCreateInstance(&ici, 0, &inst));
  uint32_t nph = 0; vkEnumeratePhysicalDevices(inst, &nph, 0);
  VkPhysicalDevice phs[8]; if (nph > 8) nph = 8; vkEnumeratePhysicalDevices(inst, &nph, phs);
  int want = getenv("M2V_DEV") ? atoi(getenv("M2V_DEV")) : 0;
  VkPhysicalDevice ph = phs[want];
  VkPhysicalDeviceVulkan11Properties p11 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES };
  VkPhysicalDeviceProperties2 pr = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p11 };
  vkGetPhysicalDeviceProperties2(ph, &pr);
  fprintf(stderr, "device: %s subgroup=%u ts_period_ns=%.3f\n", pr.properties.deviceName, p11.subgroupSize, pr.properties.limits.timestampPeriod);
  VkPhysicalDeviceVulkan12Features f12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcm = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
  f12.pNext = &fcm;
  VkPhysicalDeviceVulkan11Features f11 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12 };
  VkPhysicalDeviceFeatures2 f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11 };
  vkGetPhysicalDeviceFeatures2(ph, &f2);  // enable everything the device offers (shaderFloat16, 16-bit storage, variable pointers ...)
  float prio = 1.0f; VkDeviceQueueCreateInfo qi = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0, 0, 1, &prio };
  uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(ph, &nq, 0);
  VkQueueFamilyProperties qp[16]; if (nq > 16) nq = 16; vkGetPhysicalDeviceQueueFamilyProperties(ph, &nq, qp);
  for (uint32_t i = 0; i < nq; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qi.queueFamilyIndex = i; break; }
  const char* exts[1] = { "VK_KHR_cooperative_matrix" };
  fprintf(stderr, "cooperativeMatrix feature: %d\n", (int)fcm.cooperativeMatrix);
  VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2, 0, 1, &qi, 0, 0, fcm.cooperativeMatrix ? 1u : 0u, fcm.cooperativeMatrix ? exts : 0 };
  VkDevice dev; CK(vkCreateDevice(ph, &dci, 0, &dev));
  VkQueue q; vkGetDeviceQueue(dev, qi.queueFamilyIndex, 0, &q);
  VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(ph, &mp);
  uint32_t mt = ~0u; for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
    VkMemoryPropertyFlags want_f = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if ((mp.memoryTypes[i].propertyFlags & want_f) == want_f) { mt = i; break; }
  }
  if (mt == ~0u) { fprintf(stderr, "no host-visible coherent memory\n"); return 4; }
  VkBuffer buf[M2V_MAX_BUFS]; VkDeviceMemory mem[M2V_MAX_BUFS]; void* map[M2V_MAX_BUFS]; size_t sz[M2V_MAX_BUFS];
  for (int i = 0; i < nb; i++) {
    void* d = readf(bufs[i], &sz[i]);
    VkBufferCreateInfo bi = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, 0, 0, sz[i] ? sz[i] : 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
    CK(vkCreateBuffer(dev, &bi, 0, &buf[i]));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, buf[i], &mr);
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, mr.size, mt };
    CK(vkAllocateMemory(dev, &mai, 0, &mem[i])); CK(vkBindBufferMemory(dev, buf[i], mem[i], 0));
    CK(vkMapMemory(dev, mem[i], 0, VK_WHOLE_SIZE, 0, &map[i])); memcpy(map[i], d, sz[i]); free(d);
  }
  size_t pn = 0; void* pd = pushf ? readf(pushf, &pn) : NULL;
  VkDescriptorSetLayoutBinding lb[M2V_MAX_BUFS]; for (int i = 0; i < nb; i++) lb[i] = (VkDescriptorSetLayoutBinding){ i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, 0 };
  VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, 0, 0, nb, lb };
  VkDescriptorSetLayout dl; CK(vkCreateDescriptorSetLayout(dev, &dli, 0, &dl));
  VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pn };
  VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, 0, 0, 1, &dl, pn ? 1 : 0, pn ? &pcr : 0 };
  VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &pli, 0, &pl));
  size_t cn; void* code = readf(spv, &cn);
  VkShaderModuleCreateInfo smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, 0, 0, cn, code };
  VkShaderModule sm; CK(vkCreateShaderModule(dev, &smi, 0, &sm));
  // clspv emits the workgroup size as spec constants 0..2 (default 1,1,1); the real size is in its reflection.
  VkSpecializationMapEntry sme[3] = { { 0, 0, 4 }, { 1, 4, 4 }, { 2, 8, 4 } };
  VkSpecializationInfo spi = { 3, sme, sizeof local, local };
  VkComputePipelineCreateInfo cpi = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, 0, 0,
    { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, 0, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, entry, local[0] ? &spi : 0 }, pl };
  VkPipeline pipe; CK(vkCreateComputePipelines(dev, 0, 1, &cpi, 0, &pipe));
  VkDescriptorPoolSize ps = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nb ? nb : 1 };
  VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, 0, 0, 1, 1, &ps };
  VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpi, 0, &dp));
  VkDescriptorSetAllocateInfo dai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, 0, dp, 1, &dl };
  VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev, &dai, &ds));
  VkDescriptorBufferInfo bin[M2V_MAX_BUFS]; VkWriteDescriptorSet wr[M2V_MAX_BUFS];
  for (int i = 0; i < nb; i++) { bin[i] = (VkDescriptorBufferInfo){ buf[i], 0, VK_WHOLE_SIZE };
    wr[i] = (VkWriteDescriptorSet){ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, ds, i, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &bin[i] }; }
  vkUpdateDescriptorSets(dev, nb, wr, 0, 0);
  VkQueryPoolCreateInfo qpi = { VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, 0, 0, VK_QUERY_TYPE_TIMESTAMP, 2 };
  VkQueryPool qpool; CK(vkCreateQueryPool(dev, &qpi, 0, &qpool));
  VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, 0, 0, qi.queueFamilyIndex };
  VkCommandPool cp; CK(vkCreateCommandPool(dev, &cpci, 0, &cp));
  VkCommandBufferAllocateInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, 0, cp, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
  VkCommandBuffer cb; CK(vkAllocateCommandBuffers(dev, &cbi, &cb));
  VkCommandBufferBeginInfo bi2 = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
  CK(vkBeginCommandBuffer(cb, &bi2));
  vkCmdResetQueryPool(cb, qpool, 0, 2);
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, 0);
  if (pn) vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, (uint32_t)pn, pd);
  vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
  VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
  for (int it = 0; it < iters; it++) {
    vkCmdDispatch(cb, g[0], g[1], g[2]);
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, 0, 0, 0);
  }
  vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
  CK(vkEndCommandBuffer(cb));
  VkFenceCreateInfo fi = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO }; VkFence fence; CK(vkCreateFence(dev, &fi, 0, &fence));
  VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO, 0, 0, 0, 0, 1, &cb };
  struct timespec w0, w1; clock_gettime(CLOCK_MONOTONIC, &w0);
  CK(vkQueueSubmit(q, 1, &si, fence));
  CK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 15ull * 1000000000ull));
  clock_gettime(CLOCK_MONOTONIC, &w1);
  double wall_us = (w1.tv_sec - w0.tv_sec) * 1e6 + (w1.tv_nsec - w0.tv_nsec) / 1e3;
  uint64_t ts[2]; CK(vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof ts, ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
  double ns = (double)(ts[1] - ts[0]) * pr.properties.limits.timestampPeriod;
  printf("iters=%d total_us=%.1f per_dispatch_us=%.3f wall_us=%.1f\n", iters, ns / 1e3, ns / 1e3 / iters, wall_us);
  for (int i = 0; i < nd; i++) {
    int idx = atoi(dumps[i]); const char* fn = strchr(dumps[i], ':') + 1;
    FILE* f = fopen(fn, "wb"); fwrite(map[idx], 1, sz[idx], f); fclose(f);
  }
  return 0;
}
