import Foundation
import MetalJIT

// ==========================================================================
// MetalJITBenchmark.swift — Benchmark standalone sin XCTest
// ==========================================================================
// Compilar: swiftc -O -o /tmp/MetalJITBenchmark MetalJITBenchmark.swift
// Ejecutar: /tmp/MetalJITBenchmark
// ==========================================================================

func mjMeasure(_ label: String, iterations: Int = 1, _ block: () throws -> Void) rethrows -> Double {
    let start = DispatchTime.now()
    for _ in 0..<iterations { try block() }
    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000.0
    let perIt = elapsed / Double(iterations)
    let iterStr = iterations > 1 ? " (x\(iterations))" : ""
    let paddedLabel = label.padding(toLength: 38, withPad: " ", startingAt: 0)
    print(String(format: "  %@ %8.4f s%@", paddedLabel, perIt, iterStr))
    return perIt
}

let COUNT   = 1_000_000
let BATCHES = 100_000
let PER_B   = 6

let compiler   = JITCompiler()
let dispatcher = ComputeDispatcher()

print("""
=================================================================
 MetalJIT Benchmark — \(COUNT.formatted()) elementos / \(BATCHES.formatted()) lotes
=================================================================
""")

// --- CORE ---
print("\n[CORE]")
let cpu = try compiler.compile()
var uin = [UInt64](repeating: 1, count: COUNT); var uout = [UInt64](repeating: 0, count: COUNT)
_ = try mjMeasure("dispatch UInt64") { try uin.withUnsafeMutableBufferPointer { i in try uout.withUnsafeMutableBufferPointer { o in try dispatcher.dispatch(pipeline: cpu, input: i, output: o) } } }
var fin = [Float64](repeating: 3.14, count: COUNT); var fout = [Float64](repeating: 0, count: COUNT)
_ = try mjMeasure("dispatch Float64") { try fin.withUnsafeMutableBufferPointer { i in try fout.withUnsafeMutableBufferPointer { o in try dispatcher.dispatchFloat64(pipeline: cpu, input: i, output: o) } } }
var hin = [Float16](repeating: 1.0, count: COUNT); var hout = [Float16](repeating: 0, count: COUNT)
_ = try mjMeasure("dispatch Float16") { try hin.withUnsafeMutableBufferPointer { i in try hout.withUnsafeMutableBufferPointer { o in try dispatcher.dispatchFloat16(pipeline: cpu, input: i, output: o) } } }
print(String(repeating: "-", count: 66))

// --- BATCH ---
print("\n[BATCH]")
let bp = try compiler.compile()
var pl = [UInt64](repeating: 1, count: BATCHES * PER_B); var rl = [UInt64](repeating: 0, count: BATCHES)
_ = try mjMeasure("batch UInt64 (\(BATCHES/1000)k x \(PER_B))") { try pl.withUnsafeMutableBufferPointer { p in try rl.withUnsafeMutableBufferPointer { r in try dispatcher.dispatchBatch(pipeline: bp, payloads: p, results: r, numBatches: BATCHES, elementsPerBatch: PER_B) } } }
var fpl = [Float64](repeating: 1.0, count: BATCHES * 4); var frl = [Float64](repeating: 0, count: BATCHES)
_ = try mjMeasure("batch Float64 (\(BATCHES/1000)k x 4)") { try fpl.withUnsafeMutableBufferPointer { p in try frl.withUnsafeMutableBufferPointer { r in try dispatcher.dispatchBatchFloat64(pipeline: bp, payloads: p, results: r, numBatches: BATCHES, elementsPerBatch: 4) } } }
print(String(repeating: "-", count: 66))

// --- COMPILER ---
print("\n[COMPILER]")
let benchShader = """
#include <metal_stdlib>
using namespace metal;
kernel void bk(device const uint64_t* in [[buffer(0)]], device uint64_t* out [[buffer(1)]], uint id [[thread_position_in_grid]]) { out[id] = in[id] * 2; }
"""
_ = try mjMeasure("compilar JIT", iterations: 10) { let p = try compiler.compile(shaderSource: benchShader, kernelName: "bk"); try p.destroy() }
_ = try mjMeasure("built-in (5 kernels)", iterations: 5) { for k in BuiltInKernel.allCases { let p = try compiler.compile(builtIn: k); try p.destroy() } }
print(String(repeating: "-", count: 66))

// --- KERNELS ---
print("\n[KERNELS]")
for k in [BuiltInKernel.crypto, BuiltInKernel.logic, BuiltInKernel.tensor] {
    let p = try compiler.compile(builtIn: k); let fc = k.defaultFaceCount; let kb = 50_000
    var pd = [UInt64](repeating: 0x123456789ABCDEF, count: kb * fc); var rs = [UInt64](repeating: 0, count: kb)
    _ = try mjMeasure("\(k.rawValue) (\(kb/1000)k x \(fc))") { try pd.withUnsafeMutableBufferPointer { pp in try rs.withUnsafeMutableBufferPointer { rr in try dispatcher.dispatchBatch(pipeline: p, payloads: pp, results: rr, numBatches: kb, elementsPerBatch: fc) } } }
    try p.destroy()
}
print(String(repeating: "-", count: 66))

// --- BRIDGE ---
print("\n[BRIDGE]")
let br = try MetalJITBridge.compilePipeline(withShader: nil, kernelName: nil); defer { try? MetalJITBridge.destroyPipeline(br) }
var bi = [UInt64](repeating: 1, count: COUNT); var bo = [UInt64](repeating: 0, count: COUNT)
_ = try mjMeasure("dispatch via bridge") { try bi.withUnsafeMutableBufferPointer { i in try bo.withUnsafeMutableBufferPointer { o in try MetalJITBridge.dispatch(withPipeline: br, input: i.baseAddress, output: o.baseAddress, elementCount: Int32(COUNT), dataType: 0) } } }
print(String(repeating: "-", count: 66))

// --- BINDING ---
print("\n[BINDING]")
let bd = try compiler.compile(); let bh = bd.handle.handle
_ = mjMeasure("dispatch C wrapper") { bi.withUnsafeMutableBufferPointer { i in bo.withUnsafeMutableBufferPointer { o in _ = mjit_dispatch_uint64(Int32(bh), i.baseAddress, o.baseAddress, Int32(COUNT)) } } }
print(String(repeating: "-", count: 66))

// --- HEAP ---
print("\n[HEAP]")
_ = mjMeasure("1000 allocs + reset", iterations: 5) { let h = mjit_heap_create(10*1024*1024); if h>0 { for _ in 0..<1000 { _=mjit_heap_allocate(h,1024,256) }; mjit_heap_reset(h); mjit_heap_destroy(h) } }
_ = mjMeasure("100 create/destroy", iterations: 5) { for _ in 0..<100 { let h=mjit_heap_create(1024); if h>0 { mjit_heap_destroy(h) } } }
print(String(repeating: "-", count: 66))

// --- CACHE ---
print("\n[CACHE]")
let cs = "#include <metal_stdlib>\nusing namespace metal;\nkernel void cb(device uint64_t* out [[buffer(0)]], uint id [[thread_position_in_grid]]) { out[id]=id; }"
let cd = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("MJT_\(UUID().uuidString)")
try FileManager.default.createDirectory(at: cd, withIntermediateDirectories: true); defer { try? FileManager.default.removeItem(at: cd) }
let cp = cd.appendingPathComponent("b.metallib").path; var ce = [CChar](repeating: 0, count: 512)
_ = mjMeasure("save to cache", iterations: 5) { ce.withUnsafeMutableBufferPointer { b in _ = mjit_cache_compile_and_save(cs, cp, b.baseAddress, 512) } }
_ = mjMeasure("load from cache", iterations: 5) { let h = ce.withUnsafeMutableBufferPointer { b in mjit_cache_load_library(cp, "cb", b.baseAddress, 512) }; if h>0 { mjit_destroy_pipeline(h) } }
print(String(repeating: "-", count: 66))

// --- NATIVE ---
print("\n[NATIVE]")
let nc = 1_000_000; var nfi=[Float](repeating:1.0,count:nc); var nfo=[Float](repeating:0.0,count:nc)
var chkF: Float = 0
_ = mjMeasure("Float * 2.0 bucle") { for i in 0..<nc { nfo[i]=nfi[i]*2.0; chkF += nfo[i] } }
print("    (checksum: \(chkF))")
var ndi=[Double](repeating:1.0,count:nc); var ndo=[Double](repeating:0.0,count:nc)
var chkD: Double = 0
_ = mjMeasure("Double * 2.0 bucle") { for i in 0..<nc { ndo[i]=ndi[i]*2.0; chkD += ndo[i] } }
print("    (checksum: \(chkD))")

print("\n=================================================================")
print(" MetalJIT Benchmark — Completado")
print("=================================================================")
