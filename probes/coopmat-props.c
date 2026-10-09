// coopmat-props: print VkCooperativeMatrixPropertiesKHR for device 0 (or M2V_DEV). Build: cc -O2 coopmat-props.c -lvulkan -o coopmat-props
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

static const char* ct(VkComponentTypeKHR t) {
  switch (t) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR: return "f16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR: return "f32";
    case VK_COMPONENT_TYPE_FLOAT64_KHR: return "f64";
    case VK_COMPONENT_TYPE_SINT8_KHR: return "s8";
    case VK_COMPONENT_TYPE_SINT16_KHR: return "s16";
    case VK_COMPONENT_TYPE_SINT32_KHR: return "s32";
    case VK_COMPONENT_TYPE_UINT8_KHR: return "u8";
    case VK_COMPONENT_TYPE_UINT16_KHR: return "u16";
    case VK_COMPONENT_TYPE_UINT32_KHR: return "u32";
    default: return "other";
  }
}

int main(void) {
  VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "m2v", 1, "m2v", 1, VK_API_VERSION_1_3 };
  VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &ai };
  VkInstance inst;
  if (vkCreateInstance(&ici, 0, &inst)) return 2;
  uint32_t n = 0;
  vkEnumeratePhysicalDevices(inst, &n, 0);
  VkPhysicalDevice ph[8];
  if (n > 8) n = 8;
  vkEnumeratePhysicalDevices(inst, &n, ph);
  int want = getenv("M2V_DEV") ? atoi(getenv("M2V_DEV")) : 0;
  VkPhysicalDeviceProperties p;
  vkGetPhysicalDeviceProperties(ph[want], &p);
  printf("device %s\n", p.deviceName);
  PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR f =
      (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
  if (!f) { printf("no vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR\n"); return 0; }
  uint32_t c = 0;
  f(ph[want], &c, 0);
  VkCooperativeMatrixPropertiesKHR props[64];
  if (c > 64) c = 64;
  for (uint32_t i = 0; i < c; i++) { props[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR; props[i].pNext = 0; }
  f(ph[want], &c, props);
  printf("%u cooperative matrix shapes\n", c);
  for (uint32_t i = 0; i < c; i++)
    printf("  M=%u N=%u K=%u  A=%s B=%s C=%s Result=%s scope=%d saturating=%d\n", props[i].MSize, props[i].NSize, props[i].KSize,
           ct(props[i].AType), ct(props[i].BType), ct(props[i].CType), ct(props[i].ResultType), (int)props[i].scope,
           (int)props[i].saturatingAccumulation);
  return 0;
}
