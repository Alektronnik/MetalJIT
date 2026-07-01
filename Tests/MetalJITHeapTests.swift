import XCTest
@testable import MetalJIT

// ==========================================================================
// MetalJITHeapTests — Tests del pool de buffers GPU (MTLHeap)
// ==========================================================================

final class MetalJITHeapTests: XCTestCase {

    // MARK: - Creacion y destruccion

    func testCreateHeap() {
        let handle = mjit_heap_create(1024 * 1024)  // 1 MB
        XCTAssertGreaterThan(handle, 0, "Heap handle debe ser > 0")
        XCTAssertEqual(mjit_heap_destroy(handle), 0)
    }

    func testCreateMultipleHeaps() {
        let h1 = mjit_heap_create(1024)
        let h2 = mjit_heap_create(2048)
        let h3 = mjit_heap_create(4096)

        XCTAssertGreaterThan(h1, 0)
        XCTAssertGreaterThan(h2, 0)
        XCTAssertGreaterThan(h3, 0)
        XCTAssertNotEqual(h1, h2)
        XCTAssertNotEqual(h2, h3)

        XCTAssertEqual(mjit_heap_destroy(h1), 0)
        XCTAssertEqual(mjit_heap_destroy(h2), 0)
        XCTAssertEqual(mjit_heap_destroy(h3), 0)
    }

    func testCreateHeapSizeZero() {
        // Size 0 podria fallar o crear heap vacio
        let handle = mjit_heap_create(0)
        if handle > 0 {
            XCTAssertEqual(mjit_heap_destroy(handle), 0)
        }
    }

    func testDestroyInvalidHeap() {
        XCTAssertNotEqual(mjit_heap_destroy(99999), 0)
    }

    // MARK: - Allocate

    func testAllocateFromHeap() {
        let handle = mjit_heap_create(1024 * 1024)
        defer { mjit_heap_destroy(handle) }

        let ptr = mjit_heap_allocate(handle, 1024, 256)
        XCTAssertNotNil(ptr)
    }

    func testAllocateMultipleBuffers() {
        let handle = mjit_heap_create(1024 * 1024)
        defer { mjit_heap_destroy(handle) }

        let p1 = mjit_heap_allocate(handle, 256, 256)
        let p2 = mjit_heap_allocate(handle, 512, 256)
        let p3 = mjit_heap_allocate(handle, 128, 256)

        XCTAssertNotNil(p1)
        XCTAssertNotNil(p2)
        XCTAssertNotNil(p3)
    }

    func testAllocateExceedingCapacity() {
        let handle = mjit_heap_create(256)  // heap muy pequeno
        defer { mjit_heap_destroy(handle) }

        // Intentar asignar mas de lo que cabe
        let ptr = mjit_heap_allocate(handle, 1024, 256)
        // Puede devolver nil o un puntero (depende del driver)
        // Lo importante es que no crashee
        _ = ptr
    }

    func testAllocateWithAlignment() {
        let handle = mjit_heap_create(1024 * 1024)
        defer { mjit_heap_destroy(handle) }

        // Diferentes alineaciones
        let p1 = mjit_heap_allocate(handle, 100, 16)
        let p2 = mjit_heap_allocate(handle, 100, 64)
        let p3 = mjit_heap_allocate(handle, 100, 256)
        let p4 = mjit_heap_allocate(handle, 100, 4096)

        // Verificar que no crashea con diferentes alineaciones
        XCTAssertNotNil(p1)
        XCTAssertNotNil(p2)
        XCTAssertNotNil(p3)
        XCTAssertNotNil(p4)
    }

    // MARK: - Estadisticas

    func testHeapStats() {
        let handle = mjit_heap_create(1024 * 1024)
        defer { mjit_heap_destroy(handle) }

        var total: size_t = 0
        var used: size_t = 0

        XCTAssertEqual(mjit_heap_stats(handle, &total, &used), 0)
        XCTAssertEqual(total, 1024 * 1024)
        XCTAssertEqual(used, 0)

        // Allocate y verificar uso
        _ = mjit_heap_allocate(handle, 512, 256)
        XCTAssertEqual(mjit_heap_stats(handle, &total, &used), 0)
        XCTAssertGreaterThan(used, 0)
    }

    func testHeapStatsInvalidHandle() {
        var total: size_t = 0
        var used: size_t = 0
        XCTAssertNotEqual(mjit_heap_stats(99999, &total, &used), 0)
    }

    // MARK: - Reset

    func testHeapReset() {
        let handle = mjit_heap_create(1024 * 1024)
        defer { mjit_heap_destroy(handle) }

        _ = mjit_heap_allocate(handle, 8192, 256)
        _ = mjit_heap_allocate(handle, 4096, 256)

        var used: size_t = 0
        var total: size_t = 0
        mjit_heap_stats(handle, &total, &used)
        XCTAssertGreaterThan(used, 0, "Debe haber memoria usada tras allocate")

        XCTAssertEqual(mjit_heap_reset(handle), 0)

        mjit_heap_stats(handle, &total, &used)
        XCTAssertEqual(used, 0, "Tras reset, used debe ser 0")
    }

    func testHeapReuseAfterReset() {
        let handle = mjit_heap_create(1024 * 1024)
        defer { mjit_heap_destroy(handle) }

        // Primer uso
        let p1 = mjit_heap_allocate(handle, 256, 256)
        XCTAssertNotNil(p1)
        XCTAssertEqual(mjit_heap_reset(handle), 0)

        // Reuso tras reset
        let p2 = mjit_heap_allocate(handle, 256, 256)
        XCTAssertNotNil(p2)
    }

    // MARK: - Estres

    func testHeapManySmallAllocations() {
        let handle = mjit_heap_create(1024 * 1024)
        defer { mjit_heap_destroy(handle) }

        for _ in 0..<100 {
            _ = mjit_heap_allocate(handle, 64, 256)
        }
        // No debe crashear
    }

    func testHeapCreateDestroyCycle() {
        for _ in 0..<10 {
            let handle = mjit_heap_create(64 * 1024)
            if handle > 0 {
                _ = mjit_heap_allocate(handle, 1024, 256)
                XCTAssertEqual(mjit_heap_destroy(handle), 0)
            }
        }
    }
}
