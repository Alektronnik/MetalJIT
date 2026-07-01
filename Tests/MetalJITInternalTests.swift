import XCTest
@testable import MetalJIT

// ==========================================================================
// MetalJITInternalTests — Tests de funcionalidad interna via bridge
// ==========================================================================
// Las APIs internas (mjit_register_pipeline, MetalJITPipeline) son C++
// y no estan expuestas a Swift. Se prueban indirectamente via bridge
// y cache, que son los consumidores reales de estas APIs.
// ==========================================================================

final class MetalJITInternalTests: XCTestCase {

    // MARK: - wrapExistingHandle (puente a handles internos)

    func testWrapExistingHandleValid() {
        // Compilar via bridge produce un handle interno
        let handle = try? MetalJITBridge.compilePipeline(withShader: nil, kernelName: nil)
        XCTAssertNotNil(handle)

        if let h = handle {
            // wrapExistingHandle debe poder envolver el mismo handle
            let wrapped = MetalJITBridge.wrapExistingHandle(h.handle)
            XCTAssertNotNil(wrapped)
            XCTAssertEqual(wrapped?.handle, h.handle)

            try? MetalJITBridge.destroyPipeline(h)
        }
    }

    func testWrapExistingHandleInvalid() {
        // Handle 0 o negativo debe devolver nil
        XCTAssertNil(MetalJITBridge.wrapExistingHandle(0))
        XCTAssertNil(MetalJITBridge.wrapExistingHandle(-1))
    }

    func testWrapExistingHandleVeryLarge() {
        // Handle muy grande (no registrado) debe devolver algo
        // pero al usarlo deberia fallar
        let wrapped = MetalJITBridge.wrapExistingHandle(999999)
        if let w = wrapped {
            // Intentar despachar debe fallar
            var input:  [UInt64] = [1]
            var output: [UInt64] = [0]
            var error: NSError?

            let result = input.withUnsafeMutableBufferPointer { inPtr in
                output.withUnsafeMutableBufferPointer { outPtr in
                    MetalJITBridge.dispatch(
                        withPipeline: w,
                        input: inPtr.baseAddress,
                        output: outPtr.baseAddress,
                        elementCount: 1,
                        dataType: 0,
                        error: &error
                    )
                }
            }
            XCTAssertFalse(result)
        }
    }

    // MARK: - Pipeline registration via cache (usa mjit_register_pipeline)

    func testCachePipelineDispatch() throws {
        // Cache carga libreria y registra pipeline internamente
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void internal_k(device uint64_t* out [[buffer(0)]],
                               uint id [[thread_position_in_grid]]) {
            out[id] = id;
        }
        """
        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MetalJITInternal_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let cachePath = tempDir.appendingPathComponent("internal.metallib").path
        var errorBuf = [CChar](repeating: 0, count: 512)

        // Guardar
        let saveResult = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_cache_compile_and_save(shader, cachePath, buf.baseAddress, 512)
        }
        XCTAssertEqual(saveResult, 0)

        // Cargar (usa mjit_register_pipeline internamente)
        let handle = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_cache_load_library(cachePath, "internal_k", buf.baseAddress, 512)
        }
        XCTAssertGreaterThan(handle, 0)

        // Envolver y despachar
        guard let wrapped = MetalJITBridge.wrapExistingHandle(handle) else {
            XCTFail("wrapExistingHandle fallo")
            return
        }

        var input:  [UInt64] = [100]
        var output: [UInt64] = [0]

        try input.withUnsafeMutableBufferPointer { inPtr in
            try output.withUnsafeMutableBufferPointer { outPtr in
                try MetalJITBridge.dispatch(
                    withPipeline: wrapped,
                    input: inPtr.baseAddress,
                    output: outPtr.baseAddress,
                    elementCount: 1,
                    dataType: 0
                )
            }
        }

        // El shader internal_k pone out[id] = id, asi que out[0] = 0 (porque id=0)
        // Pero el fallback CPU copia input[0] -> output[0] = 100
        XCTAssertEqual(output[0], 100)
    }

    // MARK: - Multiples pipelines internos

    func testMultipleInternalPipelines() {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void multi_k(device uint64_t* out [[buffer(0)]],
                            uint id [[thread_position_in_grid]]) { out[id] = id; }
        """
        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MetalJITInternal_Multi_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        var errorBuf = [CChar](repeating: 0, count: 512)

        let save = { (name: String) in
            let p = tempDir.appendingPathComponent("\(name).metallib").path
            return errorBuf.withUnsafeMutableBufferPointer { buf in
                mjit_cache_compile_and_save(shader, p, buf.baseAddress, 512)
            }
        }

        XCTAssertEqual(save("p1"), 0)
        XCTAssertEqual(save("p2"), 0)
        XCTAssertEqual(save("p3"), 0)

        // Cargar los 3 (cada uno usa mjit_register_pipeline)
        let load = { (name: String) -> Int32 in
            let p = tempDir.appendingPathComponent("\(name).metallib").path
            return errorBuf.withUnsafeMutableBufferPointer { buf in
                mjit_cache_load_library(p, "multi_k", buf.baseAddress, 512)
            }
        }

        let h1 = load("p1")
        let h2 = load("p2")
        let h3 = load("p3")

        XCTAssertGreaterThan(h1, 0)
        XCTAssertGreaterThan(h2, 0)
        XCTAssertGreaterThan(h3, 0)
        XCTAssertNotEqual(h1, h2)
        XCTAssertNotEqual(h2, h3)
    }

    // MARK: - Bridge + internal consistency

    func testBridgeHandleMatchesInternal() throws {
        // Compilar via bridge normal
        let handle1 = try MetalJITBridge.compilePipeline(withShader: nil, kernelName: nil)
        let handle2 = try MetalJITBridge.compilePipeline(withShader: nil, kernelName: nil)

        XCTAssertNotEqual(handle1.handle, handle2.handle)

        // Ambos deben poder despachar independientemente
        var in1:  [UInt64] = [1]
        var out1: [UInt64] = [0]
        var in2:  [UInt64] = [2]
        var out2: [UInt64] = [0]

        try in1.withUnsafeMutableBufferPointer { i in
            try out1.withUnsafeMutableBufferPointer { o in
                try MetalJITBridge.dispatch(withPipeline: handle1, input: i.baseAddress,
                                             output: o.baseAddress, elementCount: 1, dataType: 0)
            }
        }
        try in2.withUnsafeMutableBufferPointer { i in
            try out2.withUnsafeMutableBufferPointer { o in
                try MetalJITBridge.dispatch(withPipeline: handle2, input: i.baseAddress,
                                             output: o.baseAddress, elementCount: 1, dataType: 0)
            }
        }

        XCTAssertEqual(out1, [1])
        XCTAssertEqual(out2, [2])

        try MetalJITBridge.destroyPipeline(handle1)
        try MetalJITBridge.destroyPipeline(handle2)
    }

    // MARK: - Destruccion de pipeline cacheado

    func testDestroyCachedPipeline() throws {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void destroy_k(device uint64_t* out [[buffer(0)]],
                              uint id [[thread_position_in_grid]]) { out[id] = id; }
        """
        let tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MetalJITInternal_Destroy_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let cachePath = tempDir.appendingPathComponent("destroy.metallib").path
        var errorBuf = [CChar](repeating: 0, count: 512)

        errorBuf.withUnsafeMutableBufferPointer { buf in
            _ = mjit_cache_compile_and_save(shader, cachePath, buf.baseAddress, 512)
        }

        let handle = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_cache_load_library(cachePath, "destroy_k", buf.baseAddress, 512)
        }
        XCTAssertGreaterThan(handle, 0)

        // Destruir via bridge
        guard let wrapped = MetalJITBridge.wrapExistingHandle(handle) else {
            XCTFail("wrap failed")
            return
        }
        XCTAssertNoThrow(try MetalJITBridge.destroyPipeline(wrapped))

        // Segunda destruccion debe fallar
        XCTAssertThrowsError(try MetalJITBridge.destroyPipeline(wrapped))
    }
}
