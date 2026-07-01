import XCTest
@testable import MetalJIT

// ==========================================================================
// MetalJITAsyncTests — Tests de despacho asincrono
// ==========================================================================

final class MetalJITAsyncTests: XCTestCase {

    var compiler: JITCompiler!
    var dispatcher: ComputeDispatcher!
    var pipeline: Pipeline!

    override func setUp() {
        compiler   = JITCompiler()
        dispatcher = ComputeDispatcher()
        pipeline   = try! compiler.compile()
    }

    override func tearDown() {
        try? pipeline?.destroy()
        pipeline   = nil
        dispatcher = nil
        compiler   = nil
    }

    // MARK: - Async UInt64

    @available(macOS 10.15, *)
    func testAsyncDispatchUInt64() async throws {
        var input:  [UInt64] = [10, 20, 30, 40, 50]
        var output: [UInt64] = [0, 0, 0, 0, 0]

        try await input.withUnsafeMutableBufferPointer { inPtr in
            try await output.withUnsafeMutableBufferPointer { outPtr in
                try await dispatcher.dispatchAsync(
                    pipeline: pipeline, input: inPtr, output: outPtr
                )
            }
        }
        XCTAssertEqual(output, input)
    }

    @available(macOS 10.15, *)
    func testAsyncDispatchLarge() async throws {
        let count = 1000
        var input  = [UInt64](repeating: 1, count: count)
        var output = [UInt64](repeating: 0, count: count)

        try await input.withUnsafeMutableBufferPointer { inPtr in
            try await output.withUnsafeMutableBufferPointer { outPtr in
                try await dispatcher.dispatchAsync(
                    pipeline: pipeline, input: inPtr, output: outPtr
                )
            }
        }
        XCTAssertEqual(output, input)
    }

    // MARK: - Async Float64

    @available(macOS 10.15, *)
    func testAsyncDispatchFloat64() async throws {
        var input:  [Float64] = [3.14159, 2.71828, 1.41421]
        var output: [Float64] = [0, 0, 0]

        try await input.withUnsafeMutableBufferPointer { inPtr in
            try await output.withUnsafeMutableBufferPointer { outPtr in
                try await dispatcher.dispatchFloat64Async(
                    pipeline: pipeline, input: inPtr, output: outPtr
                )
            }
        }
        XCTAssertEqual(output, input)
    }

    // MARK: - Async Batch

    @available(macOS 10.15, *)
    func testAsyncBatchDispatch() async throws {
        var payloads: [UInt64] = [
            1, 2, 3, 4,
            10, 20, 30, 40,
            100, 200, 300, 400
        ]
        var results: [UInt64] = [0, 0, 0]

        try await payloads.withUnsafeMutableBufferPointer { p in
            try await results.withUnsafeMutableBufferPointer { r in
                try await dispatcher.dispatchBatchAsync(
                    pipeline: pipeline, payloads: p, results: r,
                    numBatches: 3, elementsPerBatch: 4
                )
            }
        }
        XCTAssertEqual(results, [1, 10, 100])
    }

    // MARK: - Async con error

    @available(macOS 10.15, *)
    func testAsyncDispatchOnDestroyedPipelineThrows() async throws {
        let p = try compiler.compile()
        try p.destroy()

        var input:  [UInt64] = [1]
        var output: [UInt64] = [0]

        do {
            try await input.withUnsafeMutableBufferPointer { inPtr in
                try await output.withUnsafeMutableBufferPointer { outPtr in
                    try await dispatcher.dispatchAsync(
                        pipeline: p, input: inPtr, output: outPtr
                    )
                }
            }
            XCTFail("Debio lanzar error")
        } catch {
            // Esperado
        }
    }

    @available(macOS 10.15, *)
    func testAsyncDispatchEmptyBufferThrows() async throws {
        var empty: [UInt64] = []
        var out:   [UInt64] = []

        do {
            try await empty.withUnsafeMutableBufferPointer { inPtr in
                try await out.withUnsafeMutableBufferPointer { outPtr in
                    try await dispatcher.dispatchAsync(
                        pipeline: pipeline, input: inPtr, output: outPtr
                    )
                }
            }
            XCTFail("Debio lanzar error por buffer vacio")
        } catch let error as ComputeDispatcherError {
            XCTAssertEqual(error, .invalidBuffer)
        }
    }

    // MARK: - Async Bridge directo

    @available(macOS 10.15, *)
    func testBridgeAsyncDispatch() async throws {
        let handle = try MetalJITBridge.compilePipeline(withShader: nil, kernelName: nil)
        defer { try? MetalJITBridge.destroyPipeline(handle) }

        var input:  [UInt64] = [42]
        var output: [UInt64] = [0]

        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            input.withUnsafeMutableBufferPointer { inPtr in
                output.withUnsafeMutableBufferPointer { outPtr in
                    MetalJITBridge.dispatchAsync(
                        withPipeline: handle,
                        input: inPtr.baseAddress,
                        output: outPtr.baseAddress,
                        elementCount: 1,
                        dataType: 0
                    ) { error in
                        if let error = error {
                            cont.resume(throwing: error)
                        } else {
                            cont.resume()
                        }
                    }
                }
            }
        }

        XCTAssertEqual(output, [42])
    }

    @available(macOS 10.15, *)
    func testBridgeAsyncBatchDispatch() async throws {
        let handle = try MetalJITBridge.compilePipeline(withShader: nil, kernelName: nil)
        defer { try? MetalJITBridge.destroyPipeline(handle) }

        var payloads: [UInt64] = [1, 2, 3,  10, 20, 30]
        var results:  [UInt64] = [0, 0]

        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            payloads.withUnsafeMutableBufferPointer { p in
                results.withUnsafeMutableBufferPointer { r in
                    MetalJITBridge.dispatchBatchAsync(
                        withPipeline: handle,
                        payloads: p.baseAddress,
                        results: r.baseAddress,
                        numBatches: 2,
                        elementsPerPayload: 3,
                        dataType: 0
                    ) { error in
                        if let error = error {
                            cont.resume(throwing: error)
                        } else {
                            cont.resume()
                        }
                    }
                }
            }
        }

        XCTAssertEqual(results, [1, 10])
    }

    @available(macOS 10.15, *)
    func testBridgeAsyncDispatchError() async throws {
        // Usar handle invalido debe producir error
        let fake = MetalJITBridge.wrapExistingHandle(99999)

        if let f = fake {
            var input:  [UInt64] = [1]
            var output: [UInt64] = [0]

            do {
                try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                    input.withUnsafeMutableBufferPointer { inPtr in
                        output.withUnsafeMutableBufferPointer { outPtr in
                            MetalJITBridge.dispatchAsync(
                                withPipeline: f,
                                input: inPtr.baseAddress,
                                output: outPtr.baseAddress,
                                elementCount: 1,
                                dataType: 0
                            ) { error in
                                if let error = error {
                                    cont.resume(throwing: error)
                                } else {
                                    cont.resume()
                                }
                            }
                        }
                    }
                }
                XCTFail("Debio lanzar error")
            } catch {
                // Esperado: handle invalido
            }
        }
    }
}
