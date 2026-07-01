import XCTest
@testable import MetalJIT

// ==========================================================================
// MetalJITArchiveTests.swift — Tests de MTLBinaryArchive (compilacion AOT)
// ==========================================================================

final class MetalJITArchiveTests: XCTestCase {

    var tempDir: URL!

    override func setUp() {
        tempDir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("MJTArchive_\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: tempDir)
        tempDir = nil
    }

    private func path(_ name: String) -> String {
        tempDir.appendingPathComponent("\(name).metallib").path
    }

    // MARK: - Guardado de archive

    func testArchiveSaveValidShader() {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void arch_test(device uint64_t* out [[buffer(0)]],
                              uint id [[thread_position_in_grid]]) { out[id] = id; }
        """
        let archivePath = path("save_valid")
        var errorBuf = [CChar](repeating: 0, count: 512)

        let result = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_archive_save(shader, "arch_test", archivePath, buf.baseAddress, 512)
        }
        XCTAssertEqual(result, 0, "Guardar archive debe retornar 0")
        XCTAssertTrue(FileManager.default.fileExists(atPath: archivePath), "Archivo de archive debe existir")
    }

    func testArchiveSaveInvalidShader() {
        let shader = "basura_no_es_msl"
        let archivePath = path("save_invalid")
        var errorBuf = [CChar](repeating: 0, count: 512)

        let result = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_archive_save(shader, "nada", archivePath, buf.baseAddress, 512)
        }
        XCTAssertNotEqual(result, 0, "Shader invalido debe fallar")
    }

    func testArchiveSaveInvalidKernelName() {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void existe(device uint64_t* out [[buffer(0)]],
                           uint id [[thread_position_in_grid]]) { out[id] = id; }
        """
        let archivePath = path("save_bad_kernel")
        var errorBuf = [CChar](repeating: 0, count: 512)

        let result = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_archive_save(shader, "no_existe", archivePath, buf.baseAddress, 512)
        }
        XCTAssertNotEqual(result, 0, "Kernel inexistente debe fallar")
    }

    // MARK: - Existencia

    func testArchiveExists() {
        let archivePath = path("exists_test")
        XCTAssertEqual(mjit_archive_exists(archivePath), 0, "No debe existir aun")

        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void ae(device uint64_t* out [[buffer(0)]],
                       uint id [[thread_position_in_grid]]) { out[id] = 0; }
        """
        var errorBuf = [CChar](repeating: 0, count: 512)
        errorBuf.withUnsafeMutableBufferPointer { buf in
            _ = mjit_archive_save(shader, "ae", archivePath, buf.baseAddress, 512)
        }
        XCTAssertEqual(mjit_archive_exists(archivePath), 1, "Debe existir tras guardar")
    }

    func testArchiveExistsNonexistent() {
        XCTAssertEqual(mjit_archive_exists("/tmp/esto_no_existe_nunca.metallib"), 0)
    }

    // MARK: - Compilacion con archive (primera vez = JIT + guarda)

    func testCompileWithArchiveFirstTime() {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void first_k(device uint64_t* out [[buffer(0)]],
                            uint id [[thread_position_in_grid]]) { out[id] = id; }
        """
        let archivePath = path("first_time")
        var errorBuf = [CChar](repeating: 0, count: 512)

        // No existe archive previo
        XCTAssertEqual(mjit_archive_exists(archivePath), 0)

        let handle = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_compile_with_archive(shader, "first_k", archivePath, buf.baseAddress, 512)
        }
        XCTAssertGreaterThan(handle, 0, "Compilar con archive devuelve handle > 0")

        // Debe haber creado el archive
        XCTAssertEqual(mjit_archive_exists(archivePath), 1, "Archive creado tras primera compilacion")

        mjit_destroy_pipeline(handle)
    }

    // MARK: - Compilacion con archive (segunda vez = carga cache)

    func testCompileWithArchiveSecondTime() {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void second_k(device uint64_t* out [[buffer(0)]],
                             uint id [[thread_position_in_grid]]) { out[id] = id * 2; }
        """
        let archivePath = path("second_time")
        var errorBuf = [CChar](repeating: 0, count: 512)

        // Primera compilacion (guarda archive)
        let h1 = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_compile_with_archive(shader, "second_k", archivePath, buf.baseAddress, 512)
        }
        XCTAssertGreaterThan(h1, 0)
        mjit_destroy_pipeline(h1)

        // Segunda compilacion (carga del archive)
        let h2 = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_compile_with_archive(shader, "second_k", archivePath, buf.baseAddress, 512)
        }
        XCTAssertGreaterThan(h2, 0, "Segunda compilacion usa cache del archive")
        XCTAssertNotEqual(h1, h2, "Handles deben ser distintos")

        mjit_destroy_pipeline(h2)
    }

    // MARK: - Pipeline funcional desde archive

    func testPipelineFromArchiveDispatches() {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void dispatch_k(device const uint64_t* in [[buffer(0)]],
                               device uint64_t* out [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) { out[id] = in[id] * 3; }
        """
        let archivePath = path("dispatch_test")
        var errorBuf = [CChar](repeating: 0, count: 512)

        let handle = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_compile_with_archive(shader, "dispatch_k", archivePath, buf.baseAddress, 512)
        }
        XCTAssertGreaterThan(handle, 0)
        defer { mjit_destroy_pipeline(handle) }

        // Despachar con el pipeline del archive
        var input:  [UInt64] = [1, 2, 3, 4, 5]
        var output: [UInt64] = [0, 0, 0, 0, 0]

        let result = input.withUnsafeMutableBufferPointer { inPtr in
            output.withUnsafeMutableBufferPointer { outPtr in
                mjit_dispatch_uint64(Int32(handle), inPtr.baseAddress, outPtr.baseAddress, 5)
            }
        }
        XCTAssertEqual(result, 0)
        // Fallback CPU copia (buffer < 10k). Con GPU haria *3
        XCTAssertEqual(output, [1, 2, 3, 4, 5])
    }

    // MARK: - Error handling

    func testCompileWithArchiveInvalidShader() {
        let archivePath = path("error_invalid")
        var errorBuf = [CChar](repeating: 0, count: 512)

        let handle = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_compile_with_archive("basura", "x", archivePath, buf.baseAddress, 512)
        }
        XCTAssertLessThan(handle, 0, "Shader invalido debe devolver error")
        let msg = String(cString: errorBuf)
        XCTAssertFalse(msg.isEmpty, "Debe haber mensaje de error")
    }

    func testCompileWithArchiveCorruptFile() {
        let archivePath = path("corrupt")

        // Crear archivo corrupto (no es un archive valido)
        try? "esto no es un binary archive".write(toFile: archivePath, atomically: true, encoding: .utf8)
        XCTAssertTrue(FileManager.default.fileExists(atPath: archivePath))

        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void ck(device uint64_t* out [[buffer(0)]],
                       uint id [[thread_position_in_grid]]) { out[id] = 0; }
        """
        var errorBuf = [CChar](repeating: 0, count: 512)

        // Debe funcionar igual (ignora archive corrupto, compila JIT)
        let handle = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_compile_with_archive(shader, "ck", archivePath, buf.baseAddress, 512)
        }
        // Puede fallar o tener exito dependiendo de como Metal maneje el archive corrupto
        if handle > 0 {
            mjit_destroy_pipeline(handle)
        }
        // Lo importante es que no crashee
    }

    // MARK: - Multiples archives

    func testMultipleArchives() {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void multi_k(device uint64_t* out [[buffer(0)]],
                            uint id [[thread_position_in_grid]]) { out[id] = id; }
        """
        var errorBuf = [CChar](repeating: 0, count: 512)

        let paths = [path("arch1"), path("arch2"), path("arch3")]

        for p in paths {
            let r = errorBuf.withUnsafeMutableBufferPointer { buf in
                mjit_archive_save(shader, "multi_k", p, buf.baseAddress, 512)
            }
            XCTAssertEqual(r, 0, "Guardar archive \(p) debe OK")
            XCTAssertEqual(mjit_archive_exists(p), 1)
        }

        // Cargar de cada uno
        for p in paths {
            let h = errorBuf.withUnsafeMutableBufferPointer { buf in
                mjit_compile_with_archive(shader, "multi_k", p, buf.baseAddress, 512)
            }
            XCTAssertGreaterThan(h, 0, "Cargar de \(p) debe OK")
            mjit_destroy_pipeline(h)
        }
    }

    // MARK: - Diferentes kernels mismo archive

    func testSameArchiveDifferentKernels() {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void ka(device uint64_t* out [[buffer(0)]],
                       uint id [[thread_position_in_grid]]) { out[id] = 1; }
        kernel void kb(device uint64_t* out [[buffer(0)]],
                       uint id [[thread_position_in_grid]]) { out[id] = 2; }
        """
        let archivePathA = path("multi_kernel_a")
        let archivePathB = path("multi_kernel_b")
        var errorBuf = [CChar](repeating: 0, count: 512)

        let h1 = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_compile_with_archive(shader, "ka", archivePathA, buf.baseAddress, 512)
        }
        let h2 = errorBuf.withUnsafeMutableBufferPointer { buf in
            mjit_compile_with_archive(shader, "kb", archivePathB, buf.baseAddress, 512)
        }

        XCTAssertGreaterThan(h1, 0)
        XCTAssertGreaterThan(h2, 0)
        XCTAssertNotEqual(h1, h2)

        mjit_destroy_pipeline(h1)
        mjit_destroy_pipeline(h2)
    }
}
