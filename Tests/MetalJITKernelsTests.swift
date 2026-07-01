import XCTest
@testable import MetalJIT

// ==========================================================================
// MetalJITKernelsTests — Tests de los kernels predefinidos
// ==========================================================================

final class MetalJITKernelsTests: XCTestCase {

    // MARK: - Compilacion

    func testAllKernelsCompile() throws {
        let compiler = JITCompiler()
        for kernel in BuiltInKernel.allCases {
            let pipeline = try compiler.compile(builtIn: kernel)
            XCTAssertGreaterThan(pipeline.handle.handle, 0, "\(kernel.rawValue)")
            try? pipeline.destroy()
        }
    }

    // MARK: - Metadatos

    func testKernelMetadata() {
        for kernel in BuiltInKernel.allCases {
            XCTAssertFalse(kernel.kernelName.isEmpty, "\(kernel.rawValue) kernelName")
            XCTAssertGreaterThan(kernel.defaultFaceCount, 0, "\(kernel.rawValue) faceCount")
            XCTAssertFalse(kernel.description.isEmpty, "\(kernel.rawValue) description")
            XCTAssertFalse(kernel.mslSource.isEmpty, "\(kernel.rawValue) mslSource")
        }
    }

    func testKernelNames() {
        XCTAssertEqual(BuiltInKernel.crypto.kernelName,    "mjit_crypto")
        XCTAssertEqual(BuiltInKernel.tensor.kernelName,    "mjit_tensor")
        XCTAssertEqual(BuiltInKernel.logic.kernelName,     "mjit_logic")
        XCTAssertEqual(BuiltInKernel.physics.kernelName,   "mjit_physics")
        XCTAssertEqual(BuiltInKernel.topology.kernelName,  "mjit_topology")
    }

    func testDefaultFaceCounts() {
        XCTAssertEqual(BuiltInKernel.crypto.defaultFaceCount,    6)
        XCTAssertEqual(BuiltInKernel.tensor.defaultFaceCount,    9)
        XCTAssertEqual(BuiltInKernel.logic.defaultFaceCount,     3)
        XCTAssertEqual(BuiltInKernel.physics.defaultFaceCount,   6)
        XCTAssertEqual(BuiltInKernel.topology.defaultFaceCount,  6)
    }

    func testNativeDataTypes() {
        XCTAssertEqual(BuiltInKernel.crypto.nativeDataType,   .uint64)
        XCTAssertEqual(BuiltInKernel.tensor.nativeDataType,   .float64)
        XCTAssertEqual(BuiltInKernel.logic.nativeDataType,    .uint64)
        XCTAssertEqual(BuiltInKernel.physics.nativeDataType,  .float64)
        XCTAssertEqual(BuiltInKernel.topology.nativeDataType, .float64)
    }

    func testMSLSourceContainsKernelKeyword() {
        for kernel in BuiltInKernel.allCases {
            XCTAssertTrue(kernel.mslSource.contains("kernel void"),
                          "\(kernel.rawValue) debe contener 'kernel void'")
        }
    }

    func testMSLSourceContainsMetalStdlib() {
        for kernel in BuiltInKernel.allCases {
            XCTAssertTrue(kernel.mslSource.contains("#include <metal_stdlib>"),
                          "\(kernel.rawValue) debe incluir metal_stdlib")
        }
    }

    // MARK: - Kernel count

    func testAllCasesCount() {
        XCTAssertEqual(BuiltInKernel.allCases.count, 5)
    }

    func testRawValues() {
        let rawValues = BuiltInKernel.allCases.map { $0.rawValue }
        XCTAssertEqual(rawValues, ["crypto", "tensor", "logic", "physics", "topology"])
    }

    // MARK: - Despacho con fallback CPU

    func testCryptoKernelBatchDispatch() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile(builtIn: .crypto)
        let dispatcher = ComputeDispatcher()

        // 2 lotes de 6 elementos cada uno
        var payloads: [UInt64] = [
            1, 2, 3, 4, 5, 6,
            10, 20, 30, 40, 50, 60
        ]
        var results: [UInt64] = [0, 0]

        try payloads.withUnsafeMutableBufferPointer { p in
            try results.withUnsafeMutableBufferPointer { r in
                try dispatcher.dispatchBatch(
                    pipeline: pipeline, payloads: p, results: r,
                    numBatches: 2, elementsPerBatch: 6
                )
            }
        }

        // El fallback CPU crypto produce hashes no-cero
        XCTAssertNotEqual(results[0], 0, "Crypto hash no debe ser 0")
        XCTAssertNotEqual(results[1], 0, "Crypto hash no debe ser 0")
        XCTAssertNotEqual(results[0], results[1], "Hashes distintos para datos distintos")
    }

    func testLogicKernelBatchDispatch() throws {
        let compiler = JITCompiler()
        let pipeline = try compiler.compile(builtIn: .logic)
        let dispatcher = ComputeDispatcher()

        // 2 lotes de 3 elementos: primero con 0, segundo todo F
        var payloads: [UInt64] = [
            0xFFFFFFFFFFFFFFFF, 0x0000000000000000, 0xFFFFFFFFFFFFFFFF,
            0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF
        ]
        var results: [UInt64] = [0, 0]

        try payloads.withUnsafeMutableBufferPointer { p in
            try results.withUnsafeMutableBufferPointer { r in
                try dispatcher.dispatchBatch(
                    pipeline: pipeline, payloads: p, results: r,
                    numBatches: 2, elementsPerBatch: 3
                )
            }
        }

        // Primer lote: AND con 0 = 0
        XCTAssertEqual(results[0], 0, "AND con cero debe ser 0")
        // Segundo lote: AND de todo F = F
        XCTAssertEqual(results[1], 0xFFFFFFFFFFFFFFFF, "AND de todo F debe ser F")
    }
}
