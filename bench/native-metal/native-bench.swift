import Foundation
import Metal

let nActivation = 1 << 23
let nSoftmax = 4096
let tol: [String: Double] = ["f32": 2e-5, "f16": 2e-3]

struct RNG {
    var state: UInt64 = 7
    var spare: Double?
    mutating func uniform() -> Double {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        z ^= z >> 31
        return (Double(z >> 11) + 0.5) / 9007199254740992.0
    }
    mutating func normal() -> Float {
        if let value = spare { spare = nil; return Float(value) }
        let u1 = uniform(), u2 = uniform()
        let r = sqrt(-2.0 * log(u1)), angle = 2.0 * Double.pi * u2
        spare = r * sin(angle)
        return Float(r * cos(angle))
    }
}

struct Row {
    let name: String
    let timeUS: Double
    let gpuUS: Double
    let wallUS: Double
    let iterations: Int
    let error: Double
    let tolerance: Double
    let metric: String
    let value: Double
}

let device: MTLDevice = { guard let value = MTLCreateSystemDefaultDevice() else { fputs("FAIL: Metal device unavailable\n", stderr); exit(1) }; return value }()
let queue: MTLCommandQueue = { guard let value = device.makeCommandQueue() else { fputs("FAIL: Metal command queue unavailable\n", stderr); exit(1) }; return value }()
// Precise-rsqrt capture (docs/precise-rsqrt.md). One kernel over raw binary32 bits from a file; the output buffer is
// written back byte for byte, with no float conversion on the host. The kernel is compiled twice: with the frozen
// engine options (Metal 3.0, safe math) and with nil options, which is how every other row of this harness compiles.
// usage: native-bench --rsqrt KERNEL.metal INPUT.u32le.bin OUTPREFIX
func rsqrtCapture(_ args: [String]) throws {
    let source = try String(contentsOfFile: args[0], encoding: .utf8)
    let input = try Data(contentsOf: URL(fileURLWithPath: args[1]))
    let words = input.count / 4, group = 64, sentinel: UInt32 = 0xA5A5A5A5
    guard input.count % 4 == 0, words % group == 0 else { throw NSError(domain: "rsqrt", code: 1, userInfo: [NSLocalizedDescriptionKey: "FAIL: input must be whole 64-word groups"]) }
    let safe = MTLCompileOptions(); safe.languageVersion = .version3_0; safe.mathMode = .safe
    let probe = MTLCompileOptions()
    for (tag, options) in [("safe_msl30", Optional(safe)), ("bench_default", nil)] {
        let shown = options ?? probe
        let lib = try device.makeLibrary(source: source, options: options)
        let ps = try pipeline(lib, "rsqrt_precise")
        let inBuf = device.makeBuffer(bytes: (input as NSData).bytes, length: input.count, options: .storageModeShared)!
        let outBuf = device.makeBuffer(length: input.count, options: .storageModeShared)!
        outBuf.contents().bindMemory(to: UInt32.self, capacity: words).update(repeating: sentinel, count: words)
        _ = try dispatch(ps, [inBuf, outBuf], MTLSize(width: words / group, height: 1, depth: 1), MTLSize(width: group, height: 1, depth: 1))
        let got = output(outBuf, as: UInt32.self, count: words)
        let inputKept = Data(bytes: inBuf.contents(), count: input.count) == input
        try Data(bytes: outBuf.contents(), count: input.count).write(to: URL(fileURLWithPath: args[2] + ".\(tag).out.u32le.bin"))
        let report: [String: Any] = ["tag": tag, "device": device.name, "macos": ProcessInfo.processInfo.operatingSystemVersionString,
            "options_given": options == nil ? "nil" : "explicit", "languageVersion_raw": shown.languageVersion.rawValue, "mathMode_raw": shown.mathMode.rawValue,
            "mathFloatingPointFunctions_raw": shown.mathFloatingPointFunctions.rawValue,
            "words": words, "input_buffer_unchanged": inputKept, "unwritten_lanes": got.filter { $0 == sentinel }.count,
            "maxTotalThreadsPerThreadgroup": ps.maxTotalThreadsPerThreadgroup, "threadExecutionWidth": ps.threadExecutionWidth]
        print(String(data: try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]), encoding: .utf8)!)
    }
}
if CommandLine.arguments.count == 5 && CommandLine.arguments[1] == "--rsqrt" {
    do { try rsqrtCapture(Array(CommandLine.arguments[2...])) } catch { fputs("\(error)\n", stderr); exit(1) }
    exit(0)
}

let base = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
let outDir = URL(fileURLWithPath: CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : base.path + "/results", isDirectory: true)
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

func library(_ name: String) throws -> MTLLibrary {
    let source = try String(contentsOf: base.appendingPathComponent(name + ".metal"), encoding: .utf8)
    do { return try device.makeLibrary(source: source, options: nil) }
    catch { throw NSError(domain: "Metal compile \(name)", code: 1, userInfo: [NSLocalizedDescriptionKey: "FAIL: compiling \(name).metal: \(error)"]) }
}

func pipeline(_ lib: MTLLibrary, _ name: String) throws -> MTLComputePipelineState {
    guard let fn = lib.makeFunction(name: name) else { throw NSError(domain: "Metal", code: 1, userInfo: [NSLocalizedDescriptionKey: "FAIL: missing Metal function \(name)"]) }
    return try device.makeComputePipelineState(function: fn)
}

func buffer<T>(_ values: [T]) -> MTLBuffer {
    let b = device.makeBuffer(length: max(1, values.count * MemoryLayout<T>.stride), options: .storageModeShared)!
    values.withUnsafeBytes { b.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
    return b
}

func fill<T>(_ b: MTLBuffer, _ values: [T]) {
    values.withUnsafeBytes { b.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
}

func dispatch(_ ps: MTLComputePipelineState, _ buffers: [MTLBuffer], _ groupCount: MTLSize, _ threads: MTLSize, _ constants: [(Int, UInt32)] = []) throws -> (Double, Double) {
    guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw NSError(domain: "Metal", code: 1, userInfo: [NSLocalizedDescriptionKey: "FAIL: command buffer allocation"]) }
    enc.setComputePipelineState(ps)
    for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
    for (index, value) in constants { var x = value; enc.setBytes(&x, length: MemoryLayout<UInt32>.size, index: index) }
    enc.dispatchThreadgroups(groupCount, threadsPerThreadgroup: threads)
    enc.endEncoding()
    let start = ProcessInfo.processInfo.systemUptime
    cb.commit(); cb.waitUntilCompleted()
    guard cb.status == MTLCommandBufferStatus.completed else { throw NSError(domain: "Metal", code: 1, userInfo: [NSLocalizedDescriptionKey: "FAIL: command buffer: \(String(describing: cb.error))"]) }
    let wall = (ProcessInfo.processInfo.systemUptime - start) * 1e6
    return ((cb.gpuEndTime - cb.gpuStartTime) * 1e6, wall)
}

func timed(_ ps: MTLComputePipelineState, _ buffers: [MTLBuffer], _ groups: MTLSize, _ threads: MTLSize, _ constants: [(Int, UInt32)] = []) throws -> (Int, Double, Double, Double) {
    let pilot = try dispatch(ps, buffers, groups, threads, constants)
    let iters = max(1, min(500, Int(2_000_000 / max(pilot.1, 1.0))))
    guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { throw NSError(domain: "Metal", code: 1, userInfo: [NSLocalizedDescriptionKey: "FAIL: command buffer allocation"]) }
    enc.setComputePipelineState(ps)
    for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
    for (index, value) in constants { var x = value; enc.setBytes(&x, length: MemoryLayout<UInt32>.size, index: index) }
    for _ in 0..<iters {
        enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        enc.memoryBarrier(scope: .buffers)
    }
    enc.endEncoding()
    let start = ProcessInfo.processInfo.systemUptime
    cb.commit(); cb.waitUntilCompleted()
    let wall = (ProcessInfo.processInfo.systemUptime - start) * 1e6 / Double(iters)
    guard cb.status == MTLCommandBufferStatus.completed else { throw NSError(domain: "Metal", code: 1, userInfo: [NSLocalizedDescriptionKey: "FAIL: timed command buffer failed"]) }
    let gpu = (cb.gpuEndTime - cb.gpuStartTime) * 1e6 / Double(iters)
    let chosen = iters < 50 ? max(gpu, wall) : wall
    return (iters, chosen, gpu, wall)
}

func output<T>(_ b: MTLBuffer, as: T.Type, count: Int) -> [T] {
    Array(UnsafeBufferPointer(start: b.contents().bindMemory(to: T.self, capacity: count), count: count))
}

func activation(_ type: String, _ ps: MTLComputePipelineState) throws -> Row {
    var rng = RNG(); let x = (0..<nActivation).map { _ in rng.normal() }
    let input: MTLBuffer, result: MTLBuffer, tolValue: Double
    if type == "f32" {
        input = buffer(x); result = device.makeBuffer(length: nActivation * 4, options: .storageModeShared)!; tolValue = tol[type]!
    } else {
        let half: [Float16] = x.map { Float16($0) }; input = buffer(half); result = device.makeBuffer(length: nActivation * 2, options: .storageModeShared)!; tolValue = tol[type]!
    }
    let n = buffer([UInt32(nActivation)]), act = buffer([UInt32(0)]), inplace = UInt32(0)
    let groups = MTLSize(width: (nActivation + 255) / 256, height: 1, depth: 1), threads = MTLSize(width: 256, height: 1, depth: 1)
    let timedResult = try timed(ps, [input, result, n, act], groups, threads, [(4, inplace)])
    var maxError = 0.0
    if type == "f32" {
        let got = output(result, as: Float.self, count: nActivation)
        for i in 0..<nActivation { let ref = Double(x[i]) / (1.0 + exp(-Double(x[i]))); maxError = max(maxError, abs(Double(got[i]) - ref) / (1.0 + abs(ref))) }
    } else {
        let got = output(result, as: Float16.self, count: nActivation)
        for i in 0..<nActivation { let q = Double(Float16(x[i])); let ref = q / (1.0 + exp(-q)); maxError = max(maxError, abs(Double(got[i]) - ref) / (1.0 + abs(ref))) }
    }
    let us = timedResult.1
    return Row(name: "activation_\(type)", timeUS: us, gpuUS: timedResult.2, wallUS: timedResult.3, iterations: timedResult.0, error: maxError, tolerance: tolValue, metric: "GBps", value: Double(2 * nActivation * (type == "f32" ? 4 : 2)) / (us * 1e3))
}

func softmax(_ type: String, _ ps: MTLComputePipelineState) throws -> Row {
    var rng = RNG(); let values = (0..<(nSoftmax * nSoftmax)).map { _ in rng.normal() * 3 }
    let inputF: [Float] = values
    let sinks: MTLBuffer, input: MTLBuffer, result: MTLBuffer, tolValue = type == "f32" ? 1e-5 : tol[type]!
    if type == "f32" { input = buffer(inputF); result = input; sinks = buffer([Float](repeating: 0, count: 1)) }
    else { let h: [Float16] = inputF.map { Float16($0) }; input = buffer(h); result = input; sinks = buffer([Float16](repeating: 0, count: 1)) }
    let meta = buffer([UInt32(nSoftmax), UInt32(nSoftmax), 1]), noSinks = UInt32(0)
    let groups = MTLSize(width: nSoftmax, height: 1, depth: 1), threads = MTLSize(width: 256, height: 1, depth: 1)
    let original = inputF
    let timedResult = try timed(ps, [input, sinks, meta], groups, threads, [(3, noSinks)])
    if type == "f32" { fill(input, original) } else { fill(input, original.map { Float16($0) }) }
    _ = try dispatch(ps, [input, sinks, meta], groups, threads, [(3, noSinks)])
    var maxError = 0.0
    if type == "f32" {
        let got = output(result, as: Float.self, count: nSoftmax * nSoftmax)
        for row in 0..<nSoftmax { let base = row * nSoftmax; let maxv = original[base..<(base+nSoftmax)].map { Double($0) }.max()!; let sum = original[base..<(base+nSoftmax)].reduce(0.0) { $0 + exp(Double($1) - maxv) }; for col in 0..<nSoftmax { maxError = max(maxError, abs(Double(got[base+col]) - exp(Double(original[base+col]) - maxv) / sum)) } }
    } else {
        let got = output(result, as: Float16.self, count: nSoftmax * nSoftmax)
        for row in 0..<nSoftmax { let base = row * nSoftmax; let q = original[base..<(base+nSoftmax)].map { Double(Float16($0)) }; let maxv = q.max()!; let sum = q.reduce(0.0) { $0 + exp($1 - maxv) }; for col in 0..<nSoftmax { maxError = max(maxError, abs(Double(got[base+col]) - exp(q[col] - maxv) / sum)) } }
    }
    let us = timedResult.1
    return Row(name: "softmax_\(type)", timeUS: us, gpuUS: timedResult.2, wallUS: timedResult.3, iterations: timedResult.0, error: maxError, tolerance: tolValue, metric: "GBps", value: Double(2 * nSoftmax * nSoftmax * (type == "f32" ? 4 : 2)) / (us * 1e3))
}
func matmul(_ n: Int, _ mr: Int, _ nr: Int, _ ps: MTLComputePipelineState) throws -> Row {
    var rng = RNG(); let a = (0..<(n*n)).map { _ in rng.normal() }, b = (0..<(n*n)).map { _ in rng.normal() }
    let aa = buffer(a), bb = buffer(b), cc = device.makeBuffer(length: n*n*4, options: .storageModeShared)!, dims = buffer([UInt32(n), UInt32(n), UInt32(n)])
    let gx = n / (8 * nr), gy = n / (8 * mr), threads = MTLSize(width: 32, height: 1, depth: 1)
    let tm = try timed(ps, [aa, bb, cc, dims], MTLSize(width: gx, height: gy, depth: 1), threads)
    var ref = [Double](repeating: 0, count: n*n), got = output(cc, as: Float.self, count: n*n)
    for i in 0..<n {
        let aoff = i * n
        for k in 0..<n {
            let av = Double(a[aoff + k]), boff = k * n
            for j in 0..<n { ref[aoff + j] += av * Double(b[boff + j]) }
        }
    }
    var maxError = 0.0, maxRef = 0.0
    for i in 0..<(n*n) { maxError = max(maxError, abs(Double(got[i]) - ref[i])); maxRef = max(maxRef, abs(ref[i])) }
    maxError /= maxRef + 1e-9
    let name = mr == 1 ? "tilematmul_f32_\(n)" : "rt\(mr)x\(nr)_\(n)"
    let us = tm.1
    return Row(name: name, timeUS: us, gpuUS: tm.2, wallUS: tm.3, iterations: tm.0, error: maxError, tolerance: 1e-5, metric: "GFLOPs", value: 2.0 * Double(n*n*n) / (us * 1e3))
}

func chipName() -> String {
    var size = 0; sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
    guard size > 0 else { return "Apple Silicon" }
    var value = [CChar](repeating: 0, count: size); sysctlbyname("machdep.cpu.brand_string", &value, &size, nil, 0)
    let brand = String(cString: value)
    return brand.isEmpty ? "Apple Silicon" : brand
}

func fail(_ error: Error) -> Never { fputs("\(error)\n", stderr); exit(1) }

var rows: [Row] = []
do {
    let actLib = try library("activation"), smLib = try library("softmax"), mmLib = try library("tilematmul"), rtLib = try library("tilematmul_rt")
    fputs("running activation_f32\n", stderr); rows.append(try activation("f32", pipeline(actLib, "activation_f32")))
    fputs("running activation_f16\n", stderr); rows.append(try activation("f16", pipeline(actLib, "activation_f16")))
    fputs("running softmax_f32\n", stderr); rows.append(try softmax("f32", pipeline(smLib, "softmax_f32")))
    fputs("running softmax_f16\n", stderr); rows.append(try softmax("f16", pipeline(smLib, "softmax_f16")))
    for n in [256, 1024] { fputs("running tilematmul_f32_\(n)\n", stderr); rows.append(try matmul(n, 1, 1, pipeline(mmLib, "tile_matmul_f32"))) }
    for n in [256, 1024] { for (mr, nr) in [(2,2), (4,2), (4,4)] { fputs("running rt\(mr)x\(nr)_\(n)\n", stderr); rows.append(try matmul(n, mr, nr, pipeline(rtLib, "tile_matmul_rt\(mr)x\(nr)_f32"))) } }
} catch { fail(error) }

let header = String(format: "%-22@ %14@ %12@ %8@  %-18@ %12@", "kernel" as NSString, "native Metal" as NSString, "GPU us" as NSString, "iters" as NSString, "check" as NSString, "metric" as NSString)
print("# native Metal bench  device=\(chipName())  macOS=\(ProcessInfo.processInfo.operatingSystemVersionString)")
print(header); print(String(repeating: "-", count: 101))
var jsonRows: [String: [String: Any]] = [:]
for r in rows {
    let ok = r.error < r.tolerance
    print(String(format: "%-22@ %14.1f %12.1f %8d  err=%8.2e %@ %@=%.1f", r.name as NSString, r.timeUS, r.gpuUS, r.iterations, r.error, (ok ? "OK" : "FAIL") as NSString, r.metric as NSString, r.value))
    jsonRows[r.name] = ["us": r.timeUS, "gpu_us": r.gpuUS, "wall_us": r.wallUS, "iters": r.iterations, "err": r.error, "ok": ok, "tol": r.tolerance, "metric": r.metric, "value": r.value]
}
let date = ISO8601DateFormatter().string(from: Date()).prefix(10)
let result: [String: Any] = ["device": chipName(), "macos": ProcessInfo.processInfo.operatingSystemVersionString, "date": String(date), "rows": jsonRows]
let jsonURL = outDir.appendingPathComponent("results-native-metal-\(date).json")
try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]).write(to: jsonURL, options: .atomic)
print("# wrote \(jsonURL.path)")
if rows.contains(where: { $0.error >= $0.tolerance }) { exit(2) }
