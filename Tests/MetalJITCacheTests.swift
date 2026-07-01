import XCTest
@testable import MetalJIT

// ==========================================================================
// MetalJITCacheTests — Tests del cache de compilacion de shaders
// ==========================================================================

final class MetalJITCacheTests: XCTestCase {

    var tempDir: URL!

    override func setUp() {
        tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MetalJITCacheTests_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: tempDir)
        tempDir = nil
    }

    private func path(_ name: String) -> String {
        tempDir.appendingPathComponent("\(name).metallib").path
    }

    // MARK: - Guardado

    func testCompileAndSaveValidShader() {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void test_kernel(device uint64_t* out [[buffer(0)]],
                                uint id [[thread_position_in_grid]]) {
            out[id] = id;
        }
        """
        let cachePath = path("valid_shader")
        var errorBuf = [CChar](repeating: 0, count: 512)

        let result = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_cache_compile_and_save(shader, cachePath, buf.baseAddress, 512)
        }
        XCTAssertEqual(result, 0, "Guardado debe ser exitoso")
        XCTAssertTrue(FileManager.default.fileExists(atPath: cachePath), "Archivo debe existir")
    }

    func testCompileAndSaveInvalidShader() {
        let shader = "basura_no_es_msl_valido"
        let cachePath = path("invalid_shader")
        var errorBuf = [CChar](repeating: 0, count: 512)

        let result = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_cache_compile_and_save(shader, cachePath, buf.baseAddress, 512)
        }
        XCTAssertNotEqual(result, 0, "Shader invalido debe fallar")
    }

    // MARK: - Existencia

    func testCacheExists() {
        let cachePath = path("exists_test")

        // No existe aun
        XCTAssertEqual(mjit_cache_exists(cachePath), 0)

        // Guardar shader valido
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void k(device uint64_t* out [[buffer(0)]],
                      uint id [[thread_position_in_grid]]) { out[id] = 0; }
        """
        var errorBuf = [CChar](repeating: 0, count: 512)
        errorBuf.withUnsafeMutableBufferPointer { buf in
            _ = mjit_cache_compile_and_save(shader, cachePath, buf.baseAddress, 512)
        }

        // Ahora debe existir
        XCTAssertEqual(mjit_cache_exists(cachePath), 1)
    }

    func testCacheExistsFileNotExists() {
        XCTAssertEqual(mjit_cache_exists("/tmp/archivo_que_no_existe_12345.metallib"), 0)
    }

    // MARK: - Carga

    func testLoadCachedLibrary() {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void cached_kernel(device uint64_t* out [[buffer(0)]],
                                  uint id [[thread_position_in_grid]]) {
            out[id] = id;
        }
        """
        let cachePath = path("load_test")
        var errorBuf = [CChar](repeating: 0, count: 512)

        // Guardar
        let saveResult = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_cache_compile_and_save(shader, cachePath, buf.baseAddress, 512)
        }
        XCTAssertEqual(saveResult, 0)

        // Cargar
        let handle = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_cache_load_library(cachePath, "cached_kernel", buf.baseAddress, 512)
        }
        XCTAssertGreaterThan(handle, 0, "Carga debe devolver handle > 0")
    }

    func testLoadNonexistentCache() {
        var errorBuf = [CChar](repeating: 0, count: 512)

        let handle = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_cache_load_library("/tmp/no_existe_xyz.metallib", "k", buf.baseAddress, 512)
        }
        XCTAssertLessThan(handle, 0, "Cache inexistente debe devolver error")
    }

    func testLoadWrongKernelName() {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void kernel_a(device uint64_t* out [[buffer(0)]],
                             uint id [[thread_position_in_grid]]) { out[id] = 0; }
        """
        let cachePath = path("wrong_kernel")
        var errorBuf = [CChar](repeating: 0, count: 512)

        // Guardar con kernel_a
        errorBuf.withUnsafeMutableBufferPointer { buf in
            _ = mjit_cache_compile_and_save(shader, cachePath, buf.baseAddress, 512)
        }

        // Cargar con kernel_b (no existe)
        let handle = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_cache_load_library(cachePath, "kernel_b", buf.baseAddress, 512)
        }
        XCTAssertLessThan(handle, 0, "Kernel inexistente debe fallar")
    }

    // MARK: - Error propagation

    func testCacheErrorMessageOnFailure() {
        let cachePath = path("error_msg_test")
        var errorBuf = [CChar](repeating: 0, count: 512)

        let result = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_cache_compile_and_save("basura", cachePath, buf.baseAddress, 512)
        }
        XCTAssertNotEqual(result, 0)
        let msg = String(cString: errorBuf)
        XCTAssertFalse(msg.isEmpty, "Debe haber mensaje de error")
    }

    // MARK: - Multiples shaders

    func testCacheMultipleShaders() {
        let shader1 = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void k1(device uint64_t* out [[buffer(0)]],
                       uint id [[thread_position_in_grid]]) { out[id] = 1; }
        """
        let shader2 = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void k2(device uint64_t* out [[buffer(0)]],
                       uint id [[thread_position_in_grid]]) { out[id] = 2; }
        """
        var errorBuf = [CChar](repeating: 0, count: 512)

        let r1 = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_cache_compile_and_save(shader1, path("multi1"), buf.baseAddress, 512)
        }
        let r2 = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_cache_compile_and_save(shader2, path("multi2"), buf.baseAddress, 512)
        }

        XCTAssertEqual(r1, 0)
        XCTAssertEqual(r2, 0)
        XCTAssertEqual(mjit_cache_exists(path("multi1")), 1)
        XCTAssertEqual(mjit_cache_exists(path("multi2")), 1)
    }

    func testLoadMultipleCachedLibraries() {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void multik(device uint64_t* out [[buffer(0)]],
                           uint id [[thread_position_in_grid]]) { out[id] = id; }
        """
        var errorBuf = [CChar](repeating: 0, count: 512)

        // Guardar dos copias
        errorBuf.withUnsafeMutableBufferPointer { buf in
            _ = mjit_cache_compile_and_save(shader, path("libA"), buf.baseAddress, 512)
            _ = mjit_cache_compile_and_save(shader, path("libB"), buf.baseAddress, 512)
        }

        // Cargar ambas
        let h1 = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_cache_load_library(path("libA"), "multik", buf.baseAddress, 512)
        }
        let h2 = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_cache_load_library(path("libB"), "multik", buf.baseAddress, 512)
        }

        XCTAssertGreaterThan(h1, 0)
        XCTAssertGreaterThan(h2, 0)
        XCTAssertNotEqual(h1, h2, "Handles deben ser distintos")
    }
}
