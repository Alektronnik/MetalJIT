// ==========================================================================
// MetalJITHeap.h — Pool de buffers GPU (MTLHeap wrapper)
// ==========================================================================
// Permite asignar y reutilizar buffers GPU desde un heap pre-reservado,
// evitando la creacion/destruccion de MTLBuffer en bucles de alto rendimiento.
// ==========================================================================

#ifndef MetalJITHeap_h
#define MetalJITHeap_h

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int MJITHeapHandle;
#define MJIT_INVALID_HEAP 0

// Crea un heap GPU con capacidad total en bytes.
// Usa MTLResourceStorageModeShared para Zero-Copy en UMA.
MJITHeapHandle mjit_heap_create(size_t size_bytes);

// Destruye un heap y libera toda la memoria GPU asociada.
int mjit_heap_destroy(MJITHeapHandle handle);

// Asigna un buffer del heap. Retorna puntero al inicio del bloque.
// size_bytes: tamano solicitado.
// align: ignorado; el tamano y alineacion reales se obtienen del device
//        via heapBufferSizeAndAlignWithLength:options: (Metal).
// El buffer es valido hasta que el heap se destruye o se resetea.
void* mjit_heap_allocate(MJITHeapHandle handle, size_t size_bytes, size_t align);

// Estadisticas del heap: capacidad total y usada.
int mjit_heap_stats(MJITHeapHandle handle, size_t* total, size_t* used);

// Resetea el heap (libera todas las asignaciones sin destruir el heap).
int mjit_heap_reset(MJITHeapHandle handle);

#ifdef __cplusplus
}
#endif

#endif /* MetalJITHeap_h */
