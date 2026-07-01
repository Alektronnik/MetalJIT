import XCTest
@testable import MetalJIT

// ==========================================================================
// MetalJITCoreTests — Tests del nucleo C++ via Swift API
// ==========================================================================

final class MetalJITCoreTests: XCTestCase {

    // MARK: - Compilacion de pipelines

    func testCompileCPUPipeline() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        XCTAssertGreaterThan(pipeline.handle.handle, 0)
    }

    func testCompileGPUPipeline() throws {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void doble(device const uint64_t* in  [[buffer(0)]],
                          device uint64_t* out        [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
            out[id] = in[id] * 2;
        }
        """
        let compiler = JITCompiler()
        let pipeline = try compiler.compile(shaderSource: shader, kernelName: "doble")
        XCTAssertGreaterThan(pipeline.handle.handle, 0)
    }

    func testCompileInvalidShaderProducesError() throws {
        let compiler = JITCompiler()
        let invalid = "esto no es metal shading language"
        XCTAssertThrowsError(try compiler.compile(shaderSource: invalid, kernelName: "nada")) { error in
            guard case JITCompilerError.compilationFailed = error else {
                XCTFail("Esperaba compilationFailed")
                return
            }
        }
    }

    func testCompileInvalidKernelName() throws {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void existe(device uint64_t* out [[buffer(0)]],
                           uint id [[thread_position_in_grid]]) {
            out[id] = 1;
        }
        """
        let compiler = JITCompiler()
        XCTAssertThrowsError(try compiler.compile(shaderSource: shader, kernelName: "no_existe")) { error in
            guard case JITCompilerError.compilationFailed = error else {
                XCTFail("Esperaba compilationFailed por kernel inexistente")
                return
            }
        }
    }

    // MARK: - Multi-pipeline

    func testMultiplePipelinesSimultaneous() throws {
        let compiler = JITCompiler()
        let cpu1 = try compiler.compile()
        let cpu2 = try compiler.compile()
        let cpu3 = try compiler.compile()

        XCTAssertGreaterThan(cpu1.handle.handle, 0)
        XCTAssertGreaterThan(cpu2.handle.handle, 0)
        XCTAssertGreaterThan(cpu3.handle.handle, 0)
        XCTAssertNotEqual(cpu1.handle.handle, cpu2.handle.handle)
        XCTAssertNotEqual(cpu2.handle.handle, cpu3.handle.handle)
    }

    func testDestroyPipelineReleasesHandle() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        let handle = pipeline.handle.handle

        try pipeline.destroy()

        // Intentar despachar con pipeline destruido debe fallar
        let dispatcher = ComputeDispatcher()
        var input:  [UInt64] = [1]
        var output: [UInt64] = [0]

        XCTAssertThrowsError(
            try input.withUnsafeMutableBufferPointer { inPtr in
                try output.withUnsafeMutableBufferPointer { outPtr in
                    try dispatcher.dispatch(pipeline: pipeline, input: inPtr, output: outPtr)
                }
            }
        )
    }

    func testDestroyOnePipelineDoesNotAffectOthers() throws {
        let compiler = JITCompiler()
        let p1 = try compiler.compile()
        let p2 = try compiler.compile()

        try p1.destroy()

        // p2 debe seguir funcionando
        let dispatcher = ComputeDispatcher()
        var input:  [UInt64] = [42]
        var output: [UInt64] = [0]

        try input.withUnsafeMutableBufferPointer { inPtr in
            try output.withUnsafeMutableBufferPointer { outPtr in
                try dispatcher.dispatch(pipeline: p2, input: inPtr, output: outPtr)
            }
        }
        XCTAssertEqual(output, [42])
    }

    // MARK: - Despacho basico

    func testDispatchUInt64CPU() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        let dispatcher = ComputeDispatcher()

        var input:  [UInt64] = [10, 20, 30, 40, 50]
        var output: [UInt64] = [0, 0, 0, 0, 0]

        try input.withUnsafeMutableBufferPointer { inPtr in
            try output.withUnsafeMutableBufferPointer { outPtr in
                try dispatcher.dispatch(pipeline: pipeline, input: inPtr, output: outPtr)
            }
        }
        XCTAssertEqual(output, [10, 20, 30, 40, 50])
    }

    func testDispatchFloat64CPU() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        let dispatcher = ComputeDispatcher()

        var input:  [Float64] = [3.14159, 2.71828, 1.41421]
        var output: [Float64] = [0, 0, 0]

        try input.withUnsafeMutableBufferPointer { inPtr in
            try output.withUnsafeMutableBufferPointer { outPtr in
                try dispatcher.dispatchFloat64(pipeline: pipeline, input: inPtr, output: outPtr)
            }
        }
        XCTAssertEqual(output, input)
    }

    func testDispatchFloat32CPU() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        let dispatcher = ComputeDispatcher()

        var input:  [Float32] = [1.5, 2.5, 3.5]
        var output: [Float32] = [0, 0, 0]

        try input.withUnsafeMutableBufferPointer { inPtr in
            try output.withUnsafeMutableBufferPointer { outPtr in
                try dispatcher.dispatchFloat32(pipeline: pipeline, input: inPtr, output: outPtr)
            }
        }
        XCTAssertEqual(output, [1.5, 2.5, 3.5])
    }

    func testDispatchInt32CPU() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        let dispatcher = ComputeDispatcher()

        var input:  [Int32] = [-5, 0, 42]
        var output: [Int32] = [0, 0, 0]

        try input.withUnsafeMutableBufferPointer { inPtr in
            try output.withUnsafeMutableBufferPointer { outPtr in
                try dispatcher.dispatchInt32(pipeline: pipeline, input: inPtr, output: outPtr)
            }
        }
        XCTAssertEqual(output, [-5, 0, 42])
    }

    func testDispatchFloat16CPU() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        let dispatcher = ComputeDispatcher()

        var input:  [Float16] = [1.0, 2.0, 3.0]
        var output: [Float16] = [0, 0, 0]

        try input.withUnsafeMutableBufferPointer { inPtr in
            try output.withUnsafeMutableBufferPointer { outPtr in
                try dispatcher.dispatchFloat16(pipeline: pipeline, input: inPtr, output: outPtr)
            }
        }
        XCTAssertEqual(output, [1.0, 2.0, 3.0])
    }

    // MARK: - Batch dispatch

    func testBatchDispatchCPU() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        let dispatcher = ComputeDispatcher()

        let batches = 3
        let perBatch = 4
        var payloads: [UInt64] = [1, 2, 3, 4,  10, 20, 30, 40,  100, 200, 300, 400]
        var results:  [UInt64] = [0, 0, 0]

        try payloads.withUnsafeMutableBufferPointer { pPtr in
            try results.withUnsafeMutableBufferPointer { rPtr in
                try dispatcher.dispatchBatch(
                    pipeline: pipeline, payloads: pPtr, results: rPtr,
                    numBatches: batches, elementsPerBatch: perBatch
                )
            }
        }
        XCTAssertEqual(results, [1, 10, 100])
    }

    func testBatchDispatchFloat32() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        let dispatcher = ComputeDispatcher()

        var payloads: [Float32] = [1.5, 2.5,  10.5, 20.5,  100.5, 200.5]
        var results:  [Float32] = [0, 0, 0]

        try payloads.withUnsafeMutableBufferPointer { pPtr in
            try results.withUnsafeMutableBufferPointer { rPtr in
                try dispatcher.dispatchBatchFloat32(
                    pipeline: pipeline, payloads: pPtr, results: rPtr,
                    numBatches: 3, elementsPerBatch: 2
                )
            }
        }
        XCTAssertEqual(results, [1.5, 10.5, 100.5])
    }

    func testBatchDispatchFloat64() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        let dispatcher = ComputeDispatcher()

        var payloads: [Float64] = [1.1, 2.2,  10.1, 20.2,  100.1, 200.2]
        var results:  [Float64] = [0, 0, 0]

        try payloads.withUnsafeMutableBufferPointer { pPtr in
            try results.withUnsafeMutableBufferPointer { rPtr in
                try dispatcher.dispatchBatchFloat64(
                    pipeline: pipeline, payloads: pPtr, results: rPtr,
                    numBatches: 3, elementsPerBatch: 2
                )
            }
        }
        XCTAssertEqual(results, [1.1, 10.1, 100.1])
    }

    // MARK: - Validacion de buffers

    func testEmptyBufferThrows() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        let dispatcher = ComputeDispatcher()

        var empty: [UInt64] = []
        var out:   [UInt64] = []

        XCTAssertThrowsError(
            try empty.withUnsafeMutableBufferPointer { inPtr in
                try out.withUnsafeMutableBufferPointer { outPtr in
                    try dispatcher.dispatch(pipeline: pipeline, input: inPtr, output: outPtr)
                }
            }
        )
    }

    func testMismatchedBufferThrows() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        let dispatcher = ComputeDispatcher()

        var input:  [UInt64] = [1, 2, 3]
        var output: [UInt64] = [0, 0]

        XCTAssertThrowsError(
            try input.withUnsafeMutableBufferPointer { inPtr in
                try output.withUnsafeMutableBufferPointer { outPtr in
                    try dispatcher.dispatch(pipeline: pipeline, input: inPtr, output: outPtr)
                }
            }
        ) { error in
            XCTAssertEqual(error as? ComputeDispatcherError, .invalidBuffer)
        }
    }

    func testBatchMismatchedSizesThrows() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        let dispatcher = ComputeDispatcher()

        var payloads: [UInt64] = [1, 2, 3]  // 3 elementos, 2 batches * 3 = 6 esperados
        var results:  [UInt64] = [0, 0]

        XCTAssertThrowsError(
            try payloads.withUnsafeMutableBufferPointer { pPtr in
                try results.withUnsafeMutableBufferPointer { rPtr in
                    try dispatcher.dispatchBatch(
                        pipeline: pipeline, payloads: pPtr, results: rPtr,
                        numBatches: 2, elementsPerBatch: 3
                    )
                }
            }
        )
    }
}
