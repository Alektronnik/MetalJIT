import XCTest
@testable import MetalJIT

// ==========================================================================
// MetalJITCompilerTests — Tests del compilador JIT Swift
// ==========================================================================

final class MetalJITCompilerTests: XCTestCase {

    // MARK: - Compilacion basica

    func testCompilerCPU() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        XCTAssertGreaterThan(pipeline.handle.handle, 0)
    }

    func testCompilerGPU() throws {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void identidad(device const uint64_t* in  [[buffer(0)]],
                              device uint64_t* out        [[buffer(1)]],
                              uint id [[thread_position_in_grid]]) {
            out[id] = in[id];
        }
        """
        let compiler = JITCompiler()
        let pipeline = try compiler.compile(shaderSource: shader, kernelName: "identidad")
        XCTAssertGreaterThan(pipeline.handle.handle, 0)
    }

    func testCompilerNilShaderIsCPU() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile(shaderSource: nil, kernelName: nil)
        XCTAssertGreaterThan(pipeline.handle.handle, 0)
    }

    // MARK: - Errores de compilacion

    func testCompilerErrorPropagation() throws {
        let compiler = JITCompiler()
        let invalid = "basura no es MSL valido"
        XCTAssertThrowsError(try compiler.compile(shaderSource: invalid, kernelName: "x")) { error in
            guard case JITCompilerError.compilationFailed(let msg) = error else {
                XCTFail("Esperaba compilationFailed con mensaje")
                return
            }
            XCTAssertFalse(msg.isEmpty)
        }
    }

    func testCompilerMissingKernelError() throws {
        let shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void kernel_a(device uint64_t* out [[buffer(0)]],
                             uint id [[thread_position_in_grid]]) {
            out[id] = id;
        }
        """
        let compiler = JITCompiler()
        XCTAssertThrowsError(try compiler.compile(shaderSource: shader, kernelName: "kernel_b")) { error in
            guard case JITCompilerError.compilationFailed = error else {
                XCTFail("Esperaba compilationFailed")
                return
            }
        }
    }

    // MARK: - Multiples pipelines

    func testCompilerMultiplePipelines() throws {
        let compiler = JITCompiler()
        let p1 = try compiler.compile()
        let p2 = try compiler.compile()
        let p3 = try compiler.compile()

        XCTAssertNotEqual(p1.handle.handle, p2.handle.handle)
        XCTAssertNotEqual(p2.handle.handle, p3.handle.handle)
        XCTAssertNotEqual(p1.handle.handle, p3.handle.handle)
    }

    func testCompilerPipelineDestroy() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        XCTAssertNoThrow(try pipeline.destroy())
    }

    func testCompilerPipelineDoubleDestroy() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        try pipeline.destroy()
        // Segunda destruccion debe lanzar error
        XCTAssertThrowsError(try pipeline.destroy())
    }

    // MARK: - Built-in kernels

    func testCompileBuiltInCrypto() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile(builtIn: .crypto)
        XCTAssertGreaterThan(pipeline.handle.handle, 0)
    }

    func testCompileBuiltInTensor() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile(builtIn: .tensor)
        XCTAssertGreaterThan(pipeline.handle.handle, 0)
    }

    func testCompileBuiltInLogic() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile(builtIn: .logic)
        XCTAssertGreaterThan(pipeline.handle.handle, 0)
    }

    func testCompileBuiltInPhysics() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile(builtIn: .physics)
        XCTAssertGreaterThan(pipeline.handle.handle, 0)
    }

    func testCompileBuiltInTopology() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile(builtIn: .topology)
        XCTAssertGreaterThan(pipeline.handle.handle, 0)
    }

    func testAllBuiltInKernelsCompile() throws {
        let compiler = JITCompiler()
        for kernel in BuiltInKernel.allCases {
            let pipeline = try compiler.compile(builtIn: kernel)
            XCTAssertGreaterThan(pipeline.handle.handle, 0, "Kernel \(kernel.rawValue) fallo")
        }
    }

    // MARK: - Pipeline properties

    func testPipelineHandleIsPubliclyAccessible() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile()
        let handleValue = pipeline.handle.handle
        XCTAssertGreaterThan(handleValue, 0)
    }
}
