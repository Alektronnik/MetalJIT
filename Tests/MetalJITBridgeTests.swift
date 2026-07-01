import XCTest
@testable import MetalJIT

// ==========================================================================
// MetalJITBridgeTests — Tests del puente ObjC directamente
// ==========================================================================

final class MetalJITBridgeTests: XCTestCase {

    // MARK: - Compilacion via bridge

    func testBridgeCompileCPU() throws {
        let handle = try MetalJITBridge.compilePipeline(withShader: nil, kernelName: nil)
        XCTAssertNotNil(handle)
        XCTAssertGreaterThan(handle.handle, 0)
        try MetalJITBridge.destroyPipeline(handle)
    }

    func testBridgeCompileGPU() throws {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void test_kernel(device uint64_t* out [[buffer(0)]],
                                uint id [[thread_position_in_grid]]) {
            out[id] = id;
        }
        """
        let handle = try MetalJITBridge.compilePipeline(withShader: shader, kernelName: "test_kernel")
        XCTAssertNotNil(handle)
        XCTAssertGreaterThan(handle.handle, 0)
        try MetalJITBridge.destroyPipeline(handle)
    }

    func testBridgeCompileInvalidShaderReturnsNil() throws {
        let handle = try? MetalJITBridge.compilePipeline(
            withShader: "basura_total", kernelName: "nada"
        )
        XCTAssertNil(handle)
    }

    // MARK: - Destruccion

    func testBridgeDestroyValidHandle() throws {
        let handle = try MetalJITBridge.compilePipeline(withShader: nil, kernelName: nil)
        XCTAssertNoThrow(try MetalJITBridge.destroyPipeline(handle))
    }

    func testBridgeDestroyInvalidHandleThrows() {
        // Handle 99999 no deberia existir
        let fake = MetalJITPipelineHandle()
        // No podemos settear handle directamente, asi que usamos wrap
        let wrapped = MetalJITBridge.wrapExistingHandle(99999)
        if let w = wrapped {
            XCTAssertThrowsError(try MetalJITBridge.destroyPipeline(w))
        }
    }

    // MARK: - Despacho via bridge

    func testBridgeDispatchUInt64() throws {
        let handle = try MetalJITBridge.compilePipeline(withShader: nil, kernelName: nil)
        defer { try? MetalJITBridge.destroyPipeline(handle) }

        var input:  [UInt64] = [10, 20, 30]
        var output: [UInt64] = [0, 0, 0]

        try input.withUnsafeMutableBufferPointer { inPtr in
            try output.withUnsafeMutableBufferPointer { outPtr in
                try MetalJITBridge.dispatch(
                    withPipeline: handle, input: inPtr.baseAddress,
                    output: outPtr.baseAddress, elementCount: 3, dataType: 0
                )
            }
        }
        XCTAssertEqual(output, [10, 20, 30])
    }

    func testBridgeDispatchFloat64() throws {
        let handle = try MetalJITBridge.compilePipeline(withShader: nil, kernelName: nil)
        defer { try? MetalJITBridge.destroyPipeline(handle) }

        var input:  [Float64] = [1.5, 2.5]
        var output: [Float64] = [0, 0]

        try input.withUnsafeMutableBufferPointer { inPtr in
            try output.withUnsafeMutableBufferPointer { outPtr in
                try MetalJITBridge.dispatch(
                    withPipeline: handle, input: inPtr.baseAddress,
                    output: outPtr.baseAddress, elementCount: 2, dataType: 2
                )
            }
        }
        XCTAssertEqual(output, [1.5, 2.5])
    }

    func testBridgeDispatchBatch() throws {
        let handle = try MetalJITBridge.compilePipeline(withShader: nil, kernelName: nil)
        defer { try? MetalJITBridge.destroyPipeline(handle) }

        var payloads: [UInt64] = [1, 2, 3,  10, 20, 30]
        var results:  [UInt64] = [0, 0]

        try payloads.withUnsafeMutableBufferPointer { pPtr in
            try results.withUnsafeMutableBufferPointer { rPtr in
                try MetalJITBridge.dispatchBatch(
                    withPipeline: handle, payloads: pPtr.baseAddress,
                    results: rPtr.baseAddress, numBatches: 2,
                    elementsPerPayload: 3, dataType: 0
                )
            }
        }
        XCTAssertEqual(results, [1, 10])
    }

    // MARK: - Built-in CPU fallback

    func testBridgeSetBuiltInCPUFallback() throws {
        let handle = try MetalJITBridge.compilePipeline(withShader: nil, kernelName: nil)
        defer { try? MetalJITBridge.destroyPipeline(handle) }

        // Registrar fallback crypto (tipo 0)
        XCTAssertNoThrow(try MetalJITBridge.setBuiltInCPUFallback(handle, kernelType: 0))
    }

    func testBridgeSetBuiltInCPUFallbackInvalidType() throws {
        let handle = try MetalJITBridge.compilePipeline(withShader: nil, kernelName: nil)
        defer { try? MetalJITBridge.destroyPipeline(handle) }

        XCTAssertThrowsError(try MetalJITBridge.setBuiltInCPUFallback(handle, kernelType: 99))
    }

    // MARK: - Error propagation

    func testBridgeDispatchWithNullBufferReturnsError() {
        let handle = try? MetalJITBridge.compilePipeline(withShader: nil, kernelName: nil)
        guard let h = handle else { return }
        defer { try? MetalJITBridge.destroyPipeline(h) }

        var error: NSError?
        let result = MetalJITBridge.dispatch(
            withPipeline: h, input: nil, output: nil,
            elementCount: 10, dataType: 0, error: &error
        )
        XCTAssertFalse(result)
        XCTAssertNotNil(error)
    }
}
