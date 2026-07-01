// ==========================================================================
// MetalJITBinding.h — API C pura para MetalJIT
// ==========================================================================
// Consumible desde cualquier proyecto C99+ sin dependencias de ObjC ni Swift.
//
// Este header es incluido automaticamente por MetalJITCore.h (umbrella).
// No es necesario incluir ambos: con MetalJITCore.h basta.
//
// Uso:
//   #include "MetalJITCore.h"   // ya incluye MetalJITBinding.h
//   int handle = mjit_compile_pipeline(shader, "miKernel", errorBuf, 256);
//   mjit_dispatch(handle, input, output, count, MJIT_TYPE_FLOAT64);
//   mjit_destroy_pipeline(handle);
// ==========================================================================

#ifndef MetalJITBinding_h
#define MetalJITBinding_h

#if __has_include(<MetalJITCore/MetalJITCore.h>)
#include <MetalJITCore/MetalJITCore.h>
#else
#include "MetalJITCore.h"
#endif

// Este header extiende la API C pura con macros de conveniencia y tipos
// auxiliares. Incluye MetalJITCore.h para que los tipos base esten disponibles
// incluso si se incluye de forma independiente.

#ifdef __cplusplus
extern "C" {
#endif

// =====================================================================
// CONSTANTES DE TIPO DE DATO (alias legibles)
// =====================================================================
#define MJIT_DTYPE_UINT64  MJIT_TYPE_UINT64
#define MJIT_DTYPE_FLOAT32 MJIT_TYPE_FLOAT32
#define MJIT_DTYPE_FLOAT64 MJIT_TYPE_FLOAT64
#define MJIT_DTYPE_INT32   MJIT_TYPE_INT32
#define MJIT_DTYPE_INT64   MJIT_TYPE_INT64
#define MJIT_DTYPE_FLOAT16 MJIT_TYPE_FLOAT16

// =====================================================================
// MACROS DE CONVENIENCIA
// =====================================================================

// Despacho tipado seguro: evita casteos manuales de void*.
// Uso: MJIT_DISPATCH(handle, uint64_t, input, output, count)
#define MJIT_DISPATCH(handle, type, input, output, count) \
    mjit_dispatch((handle), (const void*)(input), (void*)(output), (count), MJIT_DTYPE_##type)

// Batch dispatch tipado seguro.
// Uso: MJIT_DISPATCH_BATCH(handle, uint64_t, payloads, results, batches, elemPerBatch)
#define MJIT_DISPATCH_BATCH(handle, type, payloads, results, numBatches, elemPerBatch) \
    mjit_dispatch_batch((handle), (const void*)(payloads), (void*)(results), \
                        (numBatches), (elemPerBatch), MJIT_DTYPE_##type)

// Compilacion con mensaje de error embebido.
// Uso: char err[256]; MJIT_COMPILE(shader, kernel, err)
#define MJIT_COMPILE(shader, kernel, errBuf) \
    mjit_compile_pipeline((shader), (kernel), (errBuf), sizeof(errBuf))

// =====================================================================
// ESTRUCTURAS AUXILIARES
// =====================================================================

// Resultado de una operacion de despacho: codigo de error + metadatos.
typedef struct {
    int error_code;
    int elements_processed;
    const char* data_type_name;
} MJITDispatchResult;

// Construye un resultado de despacho para retorno estructurado.
static inline MJITDispatchResult mjit_make_result(int error_code, int elements,
                                                   MJITDataType dtype) {
    MJITDispatchResult r;
    r.error_code        = error_code;
    r.elements_processed = (error_code == MJIT_SUCCESS) ? elements : 0;
    r.data_type_name    = mjit_data_type_name(dtype);
    return r;
}

// =====================================================================
// CONVENIENCIA: despachos con verificacion de tipos
// =====================================================================

// Despacho verificando consistencia de tipos en tiempo de compilacion.
// Wrapper estatico para uint64_t.
static inline int mjit_dispatch_uint64(int handle,
                                        const uint64_t* input,
                                        uint64_t* output,
                                        int count) {
    return mjit_dispatch(handle, input, output, count, MJIT_TYPE_UINT64);
}

// Wrapper estatico para float (32 bits).
static inline int mjit_dispatch_float32(int handle,
                                         const float* input,
                                         float* output,
                                         int count) {
    return mjit_dispatch(handle, input, output, count, MJIT_TYPE_FLOAT32);
}

// Wrapper estatico para double (64 bits).
static inline int mjit_dispatch_float64(int handle,
                                         const double* input,
                                         double* output,
                                         int count) {
    return mjit_dispatch(handle, input, output, count, MJIT_TYPE_FLOAT64);
}

// Wrapper estatico para int32_t.
static inline int mjit_dispatch_int32(int handle,
                                       const int32_t* input,
                                       int32_t* output,
                                       int count) {
    return mjit_dispatch(handle, input, output, count, MJIT_TYPE_INT32);
}

// Wrapper estatico para int64_t.
static inline int mjit_dispatch_int64(int handle,
                                       const int64_t* input,
                                       int64_t* output,
                                       int count) {
    return mjit_dispatch(handle, input, output, count, MJIT_TYPE_INT64);
}

// =====================================================================
// CONVENIENCIA: batch despachos tipados
// =====================================================================

static inline int mjit_dispatch_batch_uint64(int handle,
                                              const uint64_t* payloads,
                                              uint64_t* results,
                                              int numBatches,
                                              int elemPerBatch) {
    return mjit_dispatch_batch(handle, payloads, results,
                                numBatches, elemPerBatch, MJIT_TYPE_UINT64);
}

static inline int mjit_dispatch_batch_float32(int handle,
                                               const float* payloads,
                                               float* results,
                                               int numBatches,
                                               int elemPerBatch) {
    return mjit_dispatch_batch(handle, payloads, results,
                                numBatches, elemPerBatch, MJIT_TYPE_FLOAT32);
}

static inline int mjit_dispatch_batch_float64(int handle,
                                               const double* payloads,
                                               double* results,
                                               int numBatches,
                                               int elemPerBatch) {
    return mjit_dispatch_batch(handle, payloads, results,
                                numBatches, elemPerBatch, MJIT_TYPE_FLOAT64);
}

static inline int mjit_dispatch_batch_int32(int handle,
                                             const int32_t* payloads,
                                             int32_t* results,
                                             int numBatches,
                                             int elemPerBatch) {
    return mjit_dispatch_batch(handle, payloads, results,
                                numBatches, elemPerBatch, MJIT_TYPE_INT32);
}

#ifdef __cplusplus
}
#endif

#endif /* MetalJITBinding_h */
