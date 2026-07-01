import XCTest
@testable import MetalJIT

// ==========================================================================
// MetalJITBindingTests — Tests de los bindings C (MetalJITBinding.h)
// ==========================================================================

final class MetalJITBindingTests: XCTestCase {

    var compiler: JITCompiler!
    var pipeline: Pipeline!

    override func setUp() {
        compiler = JITCompiler()
        pipeline = try! compiler.compile()
    }

    override func tearDown() {
        try? pipeline?.destroy()
        pipeline = nil
        compiler = nil
    }

    // MARK: - Wrappers tipados UInt64

    func testDispatchUInt64() {
        let handle = pipeline.handle.handle
        var input:  [UInt64] = [10, 20, 30, 40, 50]
        var output: [UInt64] = [0, 0, 0, 0, 0]

        let result = input.withUnsafeMutableBufferPointer { inPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                mjit_dispatch_uint64(Int32(handle), inPtr.baseAddress, outPtr.baseAddress, 5)
            }
        }
        XCTAssertEqual(result, 0)
        XCTAssertEqual(output, [10, 20, 30, 40, 50])
    }

    // MARK: - Wrappers tipados Float32

    func testDispatchFloat32() {
        let handle = pipeline.handle.handle
        var input:  [Float] = [1.5, 2.5, 3.5]
        var output: [Float] = [0, 0, 0]

        let result = input.withUnsafeMutableBufferPointer { inPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                mjit_dispatch_float32(Int32(handle), inPtr.baseAddress, outPtr.baseAddress, 3)
            }
        }
        XCTAssertEqual(result, 0)
        XCTAssertEqual(output, [1.5, 2.5, 3.5])
    }

    // MARK: - Wrappers tipados Float64

    func testDispatchFloat64() {
        let handle = pipeline.handle.handle
        var input:  [Double] = [3.14159, 2.71828, 1.41421]
        var output: [Double] = [0, 0, 0]

        let result = input.withUnsafeMutableBufferPointer { inPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                mjit_dispatch_float64(Int32(handle), inPtr.baseAddress, outPtr.baseAddress, 3)
            }
        }
        XCTAssertEqual(result, 0)
        XCTAssertEqual(output, input)
    }

    // MARK: - Wrappers tipados Int32

    func testDispatchInt32() {
        let handle = pipeline.handle.handle
        var input:  [Int32] = [-42, 0, 42]
        var output: [Int32] = [0, 0, 0]

        let result = input.withUnsafeMutableBufferPointer { inPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                mjit_dispatch_int32(Int32(handle), inPtr.baseAddress, outPtr.baseAddress, 3)
            }
        }
        XCTAssertEqual(result, 0)
        XCTAssertEqual(output, [-42, 0, 42])
    }

    // MARK: - Wrappers tipados Int64

    func testDispatchInt64() {
        let handle = pipeline.handle.handle
        var input:  [Int64] = [Int64.max, Int64.min, 0]
        var output: [Int64] = [0, 0, 0]

        let result = input.withUnsafeMutableBufferPointer { inPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                mjit_dispatch_int64(Int32(handle), inPtr.baseAddress, outPtr.baseAddress, 3)
            }
        }
        XCTAssertEqual(result, 0)
        XCTAssertEqual(output, [Int64.max, Int64.min, 0])
    }

    // MARK: - Batch wrappers tipados

    func testDispatchBatchUInt64() {
        let handle = pipeline.handle.handle
        var payloads: [UInt64] = [1, 2, 3,  10, 20, 30]
        var results:  [UInt64] = [0, 0]

        let result = payloads.withUnsafeMutableBufferPointer { p in
            results.withUnsafeMutableBufferPointer { r in
                mjit_dispatch_batch_uint64(Int32(handle), p.baseAddress, r.baseAddress, 2, 3)
            }
        }
        XCTAssertEqual(result, 0)
        XCTAssertEqual(results, [1, 10])
    }

    func testDispatchBatchFloat32() {
        let handle = pipeline.handle.handle
        var payloads: [Float] = [1.5, 2.5,  10.5, 20.5]
        var results:  [Float] = [0, 0]

        let result = payloads.withUnsafeMutableBufferPointer { p in
            results.withUnsafeMutableBufferPointer { r in
                mjit_dispatch_batch_float32(Int32(handle), p.baseAddress, r.baseAddress, 2, 2)
            }
        }
        XCTAssertEqual(result, 0)
        XCTAssertEqual(results, [1.5, 10.5])
    }

    func testDispatchBatchFloat64() {
        let handle = pipeline.handle.handle
        var payloads: [Double] = [1.1, 2.2,  3.3, 4.4]
        var results:  [Double] = [0, 0]

        let result = payloads.withUnsafeMutableBufferPointer { p in
            results.withUnsafeMutableBufferPointer { r in
                mjit_dispatch_batch_float64(Int32(handle), p.baseAddress, r.baseAddress, 2, 2)
            }
        }
        XCTAssertEqual(result, 0)
        XCTAssertEqual(results, [1.1, 3.3])
    }

    // MARK: - Utilidades

    func testDataTypeSize() {
        XCTAssertEqual(mjit_data_type_size(0), 8)   // UInt64
        XCTAssertEqual(mjit_data_type_size(1), 4)   // Float32
        XCTAssertEqual(mjit_data_type_size(2), 8)   // Float64
        XCTAssertEqual(mjit_data_type_size(3), 4)   // Int32
        XCTAssertEqual(mjit_data_type_size(4), 8)   // Int64
        XCTAssertEqual(mjit_data_type_size(5), 2)   // Float16
    }

    func testDataTypeName() {
        XCTAssertEqual(String(cString: mjit_data_type_name(0)), "UInt64")
        XCTAssertEqual(String(cString: mjit_data_type_name(1)), "Float32")
        XCTAssertEqual(String(cString: mjit_data_type_name(2)), "Float64")
        XCTAssertEqual(String(cString: mjit_data_type_name(5)), "Float16")
    }

    // MARK: - MJITDispatchResult

    func testDispatchResultSuccess() {
        let res = mjit_make_result(0, 100, MJIT_TYPE_UINT64)
        XCTAssertEqual(res.error_code, 0)
        XCTAssertEqual(res.elements_processed, 100)
        let name = String(cString: res.data_type_name!)
        XCTAssertEqual(name, "UInt64")
    }

    func testDispatchResultError() {
        let res = mjit_make_result(103, 50, MJIT_TYPE_FLOAT64)
        XCTAssertEqual(res.error_code, 103)
        XCTAssertEqual(res.elements_processed, 0)  // error => 0 processed
    }

    // MARK: - Genérico tipado via mjit_dispatch

    func testDispatchGenericUInt64() {
        let handle = pipeline.handle.handle
        var input:  [UInt64] = [7, 8, 9]
        var output: [UInt64] = [0, 0, 0]

        let result = input.withUnsafeMutableBufferPointer { inPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                mjit_dispatch(Int32(handle), inPtr.baseAddress, outPtr.baseAddress, 3, MJIT_TYPE_UINT64)
            }
        }
        XCTAssertEqual(result, 0)
        XCTAssertEqual(output, [7, 8, 9])
    }

    func testDispatchGenericFloat64() {
        let handle = pipeline.handle.handle
        var input:  [Double] = [3.14, 2.71]
        var output: [Double] = [0, 0]

        let result = input.withUnsafeMutableBufferPointer { inPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                mjit_dispatch(Int32(handle), inPtr.baseAddress, outPtr.baseAddress, 2, MJIT_TYPE_FLOAT64)
            }
        }
        XCTAssertEqual(result, 0)
        XCTAssertEqual(output, [3.14, 2.71])
    }

    // MARK: - Error handling

    func testDispatchInvalidHandle() {
        var input:  [UInt64] = [1]
        var output: [UInt64] = [0]

        let result = input.withUnsafeMutableBufferPointer { inPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                mjit_dispatch(99999, inPtr.baseAddress, outPtr.baseAddress, 1, MJIT_TYPE_UINT64)
            }
        }
        XCTAssertNotEqual(result, 0)
    }

    func testDispatchNullBuffer() {
        let handle = pipeline.handle.handle
        let result = mjit_dispatch(Int32(handle), nil, nil, 0, MJIT_TYPE_UINT64)
        XCTAssertNotEqual(result, 0)
    }
}
