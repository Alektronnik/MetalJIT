// ==========================================================================
// MetalJITHeap.mm — Implementacion del pool de buffers GPU
// ==========================================================================

#import "MetalJITHeap.h"
#include <unordered_map>
#include <mutex>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

static id<MTLDevice> _heap_device() {
    return MTLCreateSystemDefaultDevice();
}
#endif

struct HeapData {
#ifdef __OBJC__
    id<MTLHeap> metalHeap;
    NSMutableArray<id<MTLBuffer>>* liveBuffers;  // retiene buffers para que punteros no queden dangling
#endif
    size_t capacity;
    size_t used;      // bytes consumidos (para stats)
    size_t offset;    // offset real dentro del MTLHeap para la proxima asignacion
};

static std::mutex g_heap_mutex;
static std::unordered_map<int, HeapData*> g_heaps;
static std::atomic<int> g_next_heap_id{1};

extern "C" {

MJITHeapHandle mjit_heap_create(size_t size_bytes) {
    std::lock_guard<std::mutex> lock(g_heap_mutex);

    auto* h = new HeapData();
    h->capacity = size_bytes;
    h->used     = 0;
    h->offset   = 0;
#ifdef __OBJC__
    h->liveBuffers = [[NSMutableArray alloc] init];
#endif

#ifdef __OBJC__
    id<MTLDevice> device = _heap_device();
    if (!device) { delete h; return MJIT_INVALID_HEAP; }

    MTLHeapDescriptor* desc = [[MTLHeapDescriptor alloc] init];
    desc.size   = size_bytes;
    desc.storageMode = MTLStorageModeShared;

    h->metalHeap = [device newHeapWithDescriptor:desc];
    if (!h->metalHeap) { delete h; return MJIT_INVALID_HEAP; }
#endif

    int id = g_next_heap_id.fetch_add(1);
    g_heaps[id] = h;
    return id;
}

int mjit_heap_destroy(MJITHeapHandle handle) {
    std::lock_guard<std::mutex> lock(g_heap_mutex);
    auto it = g_heaps.find(handle);
    if (it == g_heaps.end()) return -1;
    delete it->second;
    g_heaps.erase(it);
    return 0;
}

void* mjit_heap_allocate(MJITHeapHandle handle, size_t size_bytes, size_t align) {
    std::lock_guard<std::mutex> lock(g_heap_mutex);
    auto it = g_heaps.find(handle);
    if (it == g_heaps.end()) return nullptr;

    HeapData* h = it->second;

#ifdef __OBJC__
    // Consultar a Metal el tamano y alineacion reales requeridos por el device
    MTLSizeAndAlign sa = [h->metalHeap.device heapBufferSizeAndAlignWithLength:size_bytes
                                                                       options:MTLResourceStorageModeShared];
    size_t metal_size  = sa.size;
    size_t metal_align = sa.align;

    // Alinear el offset actual al boundary que Metal exige
    size_t aligned_offset = h->offset;
    if (metal_align > 1) {
        aligned_offset = ((h->offset + metal_align - 1) / metal_align) * metal_align;
    }

    // Verificar capacidad con el tamano real que Metal usara
    if (aligned_offset + metal_size > h->capacity) {
        return nullptr;
    }

    id<MTLBuffer> buf = [h->metalHeap newBufferWithLength:metal_size
                                                   options:MTLResourceStorageModeShared
                                                     offset:aligned_offset];
    if (buf) {
        h->offset = aligned_offset + metal_size;
        h->used   = h->offset;
        [h->liveBuffers addObject:buf];
        return [buf contents];
    }
#else
    (void)align;
#endif
    return nullptr;
}

int mjit_heap_stats(MJITHeapHandle handle, size_t* total, size_t* used) {
    std::lock_guard<std::mutex> lock(g_heap_mutex);
    auto it = g_heaps.find(handle);
    if (it == g_heaps.end()) return -1;
    *total = it->second->capacity;
    *used  = it->second->used;
    return 0;
}

int mjit_heap_reset(MJITHeapHandle handle) {
    std::lock_guard<std::mutex> lock(g_heap_mutex);
    auto it = g_heaps.find(handle);
    if (it == g_heaps.end()) return -1;
#ifdef __OBJC__
    [it->second->liveBuffers removeAllObjects];
    size_t cap = it->second->capacity;
    id<MTLDevice> device = _heap_device();
    if (device) {
        MTLHeapDescriptor* desc = [[MTLHeapDescriptor alloc] init];
        desc.size = cap;
        desc.storageMode = MTLStorageModeShared;
        it->second->metalHeap = [device newHeapWithDescriptor:desc];
    }
#endif
    it->second->used = 0;
    it->second->offset = 0;
    return 0;
}

} // extern "C"
