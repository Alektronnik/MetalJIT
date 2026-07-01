// MetalJIT.swift
// Punto de entrada unificado del modulo MetalJIT.
//
// MetalJIT permite compilar shaders Metal en tiempo de ejecucion (JIT)
// y despacharlos sobre buffers de memoria sin copia, explotando la
// arquitectura UMA de Apple Silicon.
//
// Uso tipico:
// ```swift
// import MetalJIT
//
// let compiler = JITCompiler()
// let pipeline = try compiler.compile(shaderSource: miShader, kernelName: "compute")
//
// let dispatcher = ComputeDispatcher()
// var input:  [UInt64] = [1, 2, 3, 4, 5]
// var output: [UInt64] = [0, 0, 0, 0, 0]
//
// try input.withUnsafeMutableBufferPointer { inPtr in
//     try output.withUnsafeMutableBufferPointer { outPtr in
//         try dispatcher.dispatch(pipeline: pipeline, input: inPtr, output: outPtr)
//     }
// }
// ```

// Los tipos publicos JITCompiler, Pipeline, ComputeDispatcher y sus errores
// se exportan automaticamente al ser declarados como `public` en sus
// respectivos archivos dentro del mismo modulo.
