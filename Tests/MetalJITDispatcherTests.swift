import XCTest
@testable import MetalJIT

// ==========================================================================
// MetalJITDispatcherTests — Tests del despachador Swift
// ==========================================================================

final class MetalJITDispatcherTests: XCTestCase {

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

    // MARK: - UInt64

    func testDispatchUInt64() throws {
        var input:  [UInt64] = [10, 20, 30, 40, 50]
        var output: [UInt64] = [0, 0, 0, 0, 0]
        try input.withUnsafeMutableBufferPointer { i in
            try output.withUnsafeMutableBufferPointer { o in
                try dispatcher.dispatch(pipeline: pipeline, input: i, output: o)
            }
        }
        XCTAssertEqual(output, input)
    }

    // MARK: - Float32

    func testDispatchFloat32() throws {
        var input:  [Float32] = [1.5, 2.5, 3.5]
        var output: [Float32] = [0, 0, 0]
        try input.withUnsafeMutableBufferPointer { i in
            try output.withUnsafeMutableBufferPointer { o in
                try dispatcher.dispatchFloat32(pipeline: pipeline, input: i, output: o)
            }
        }
        XCTAssertEqual(output, input)
    }

    // MARK: - Float64

    func testDispatchFloat64() throws {
        var input:  [Float64] = [3.1415926535, 2.7182818284, 1.4142135623]
        var output: [Float64] = [0, 0, 0]
        try input.withUnsafeMutableBufferPointer { i in
            try output.withUnsafeMutableBufferPointer { o in
                try dispatcher.dispatchFloat64(pipeline: pipeline, input: i, output: o)
            }
        }
        XCTAssertEqual(output, input)
    }

    // MARK: - Int32

    func testDispatchInt32() throws {
        var input:  [Int32] = [-100, 0, 100, Int32.max, Int32.min]
        var output: [Int32] = [0, 0, 0, 0, 0]
        try input.withUnsafeMutableBufferPointer { i in
            try output.withUnsafeMutableBufferPointer { o in
                try dispatcher.dispatchInt32(pipeline: pipeline, input: i, output: o)
            }
        }
        XCTAssertEqual(output, input)
    }

    // MARK: - Float16

    func testDispatchFloat16() throws {
        var input:  [Float16] = [1.0, 2.0, 3.0]
        var output: [Float16] = [0, 0, 0]
        try input.withUnsafeMutableBufferPointer { i in
            try output.withUnsafeMutableBufferPointer { o in
                try dispatcher.dispatchFloat16(pipeline: pipeline, input: i, output: o)
            }
        }
        XCTAssertEqual(output, [1.0, 2.0, 3.0])
    }

    // MARK: - Batch

    func testDispatchBatchUInt64() throws {
        var payloads: [UInt64] = [
            1, 2, 3, 4,
            10, 20, 30, 40,
            100, 200, 300, 400
        ]
        var results: [UInt64] = [0, 0, 0]
        try payloads.withUnsafeMutableBufferPointer { p in
            try results.withUnsafeMutableBufferPointer { r in
                try dispatcher.dispatchBatch(
                    pipeline: pipeline, payloads: p, results: r,
                    numBatches: 3, elementsPerBatch: 4
                )
            }
        }
        XCTAssertEqual(results, [1, 10, 100])
    }

    func testDispatchBatchFloat32() throws {
        var payloads: [Float32] = [1.5, 2.5,  10.5, 20.5]
        var results:  [Float32] = [0, 0]
        try payloads.withUnsafeMutableBufferPointer { p in
            try results.withUnsafeMutableBufferPointer { r in
                try dispatcher.dispatchBatchFloat32(
                    pipeline: pipeline, payloads: p, results: r,
                    numBatches: 2, elementsPerBatch: 2
                )
            }
        }
        XCTAssertEqual(results, [1.5, 10.5])
    }

    func testDispatchBatchFloat64() throws {
        var payloads: [Float64] = [1.1, 2.2,  3.3, 4.4]
        var results:  [Float64] = [0, 0]
        try payloads.withUnsafeMutableBufferPointer { p in
            try results.withUnsafeMutableBufferPointer { r in
                try dispatcher.dispatchBatchFloat64(
                    pipeline: pipeline, payloads: p, results: r,
                    numBatches: 2, elementsPerBatch: 2
                )
            }
        }
        XCTAssertEqual(results, [1.1, 3.3])
    }

    func testDispatchBatchSingleElement() throws {
        var payloads: [UInt64] = [42, 99, 7]
        var results:  [UInt64] = [0, 0, 0]
        try payloads.withUnsafeMutableBufferPointer { p in
            try results.withUnsafeMutableBufferPointer { r in
                try dispatcher.dispatchBatch(
                    pipeline: pipeline, payloads: p, results: r,
                    numBatches: 3, elementsPerBatch: 1
                )
            }
        }
        XCTAssertEqual(results, [42, 99, 7])
    }

    // MARK: - Validacion de errores

    func testDispatchEmptyBufferThrows() {
        var empty: [UInt64] = []
        var out:   [UInt64] = []
        XCTAssertThrowsError(
            try empty.withUnsafeMutableBufferPointer { i in
                try out.withUnsafeMutableBufferPointer { o in
                    try dispatcher.dispatch(pipeline: pipeline, input: i, output: o)
                }
            }
        ) { error in
            XCTAssertEqual(error as? ComputeDispatcherError, .invalidBuffer)
        }
    }

    func testDispatchMismatchedBufferThrows() {
        var input:  [UInt64] = [1, 2, 3]
        var output: [UInt64] = [0, 0]
        XCTAssertThrowsError(
            try input.withUnsafeMutableBufferPointer { i in
                try output.withUnsafeMutableBufferPointer { o in
                    try dispatcher.dispatch(pipeline: pipeline, input: i, output: o)
                }
            }
        ) { error in
            XCTAssertEqual(error as? ComputeDispatcherError, .invalidBuffer)
        }
    }

    func testBatchInvalidSizesThrows() {
        var payloads: [UInt64] = [1, 2]
        var results:  [UInt64] = [0]
        XCTAssertThrowsError(
            try payloads.withUnsafeMutableBufferPointer { p in
                try results.withUnsafeMutableBufferPointer { r in
                    try dispatcher.dispatchBatch(
                        pipeline: pipeline, payloads: p, results: r,
                        numBatches: 2, elementsPerBatch: 2
                    )
                }
            }
        )
    }

    // MARK: - Dispatch sin pipeline (destruido)

    func testDispatchOnDestroyedPipelineThrows() throws {
        let p = try compiler.compile()
        try p.destroy()

        var input:  [UInt64] = [1]
        var output: [UInt64] = [0]
        XCTAssertThrowsError(
            try input.withUnsafeMutableBufferPointer { i in
                try output.withUnsafeMutableBufferPointer { o in
                    try dispatcher.dispatch(pipeline: p, input: i, output: o)
                }
            }
        )
    }
}
