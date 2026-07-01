import Foundation
import MetalJIT

// ==========================================================================
// MetalJITExample.swift
// ==========================================================================
// Ejemplo completo de la API publica de MetalJIT en Swift.
// Cubre: compilacion JIT, despacho UInt64/Float64, batch, multi-pipeline.
//
// Ejecutar:
//   swift run MetalJITRunner
// ==========================================================================

print("""
=========================================================================
 MetalJIT — Ejemplo Swift
=========================================================================
""")

let compiler = JITCompiler()
let dispatcher = ComputeDispatcher()

// -----------------------------------------------------------------------
// 1. PIPELINE CPU-ONLY (sin shader Metal)
// -----------------------------------------------------------------------
print("[1] Pipeline CPU-only")

let cpuPipeline = try compiler.compile()
var input:  [UInt64] = [10, 20, 30, 40, 50]
var output: [UInt64] = [0, 0, 0, 0, 0]

try input.withUnsafeMutableBufferPointer { inPtr in
    try output.withUnsafeMutableBufferPointer { outPtr in
        try dispatcher.dispatch(pipeline: cpuPipeline, input: inPtr, output: outPtr)
    }
}

print("    Entrada : \(input)")
print("    Salida  : \(output)")
assert(output == input, "CPU fallback debe copiar entrada a salida")
print("    ✓ OK\n")

// -----------------------------------------------------------------------
// 2. COMPILACION JIT DE SHADER METAL
// -----------------------------------------------------------------------
print("[2] Compilacion Metal JIT")

let dobleShader = """
#include <metal_stdlib>
using namespace metal;
kernel void doble(
    device const uint64_t* in  [[buffer(0)]],
    device uint64_t* out       [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = in[id] * 2;
}
"""

let doblePipeline = try compiler.compile(shaderSource: dobleShader, kernelName: "doble")
print("    Shader 'doble' compilado")
print("    ✓ OK\n")

// -----------------------------------------------------------------------
// 3. DESPACHO Float64
// -----------------------------------------------------------------------
print("[3] Despacho Float64 (Double)")

let f64Pipeline = try compiler.compile()
var f64in:  [Float64] = [3.1415926535, 2.7182818284, 1.4142135623]
var f64out: [Float64] = [0.0, 0.0, 0.0]

try f64in.withUnsafeMutableBufferPointer { inPtr in
    try f64out.withUnsafeMutableBufferPointer { outPtr in
        try dispatcher.dispatchFloat64(pipeline: f64Pipeline, input: inPtr, output: outPtr)
    }
}

print("    Entrada : \(f64in)")
print("    Salida  : \(f64out)")
assert(f64out == f64in, "Float64 fallback debe copiar")
print("    ✓ OK\n")

// -----------------------------------------------------------------------
// 4. DESPACHO POR LOTES (BATCH)
// -----------------------------------------------------------------------
print("[4] Batch dispatch (3 lotes de 4 elementos)")

let batchPipeline = try compiler.compile()
let batches = 3
let perBatch = 4
var payloads: [UInt64] = [
    1,  2,  3,  4,
    10, 20, 30, 40,
    100, 200, 300, 400
]
var results: [UInt64] = [0, 0, 0]

try payloads.withUnsafeMutableBufferPointer { pPtr in
    try results.withUnsafeMutableBufferPointer { rPtr in
        try dispatcher.dispatchBatch(
            pipeline: batchPipeline,
            payloads: pPtr,
            results: rPtr,
            numBatches: batches,
            elementsPerBatch: perBatch
        )
    }
}

print("    Payloads (planos): \(payloads)")
print("    Resultados       : \(results)")
assert(results == [1, 10, 100], "Batch copia primer elemento de cada lote")
print("    ✓ OK\n")

// -----------------------------------------------------------------------
// 5. MULTI-PIPELINE (varios kernels simultaneos)
// -----------------------------------------------------------------------
print("[5] Multi-pipeline simultaneo")

let tripleShader = """
#include <metal_stdlib>
using namespace metal;
kernel void triple(
    device const uint64_t* in  [[buffer(0)]],
    device uint64_t* out       [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    out[id] = in[id] * 3;
}
"""

let pDoble  = try compiler.compile(shaderSource: dobleShader,  kernelName: "doble")
let pTriple = try compiler.compile(shaderSource: tripleShader, kernelName: "triple")

assert(pDoble !== pTriple, "Pipelines deben ser objetos distintos")

try pDoble.destroy()
print("    Pipeline doble destruido. Pipeline triple sigue activo.")
print("    ✓ OK\n")

// -----------------------------------------------------------------------
// 6. COMPILACION FALLIDA (mensaje de error)
// -----------------------------------------------------------------------
print("[6] Compilacion fallida con mensaje de error")

let shaderRoto = "esto no es un shader valido de Metal para nada"
do {
    let _ = try compiler.compile(shaderSource: shaderRoto, kernelName: "roto")
    fatalError("REGRESION: El motor compilo un shader que debio fallar. Revisar MetalJITCore.")
} catch JITCompilerError.compilationFailed(let msg) {
    print("    Error de compilacion capturado:")
    print("    \(msg)")
    print("    ✓ OK (esperado)\n")
} catch {
    print("    ERROR INESPERADO: \(error)")
    exit(1)
}

// -----------------------------------------------------------------------
// 7. CIERRE
// -----------------------------------------------------------------------
try? pTriple.destroy()
try? cpuPipeline.destroy()
try? f64Pipeline.destroy()
try? batchPipeline.destroy()
try? doblePipeline.destroy()

print("=========================================================================")
print(" MetalJIT Swift — Todos los ejemplos completados correctamente")
print("=========================================================================")