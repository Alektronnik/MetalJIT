#ifndef MetalJITCore_h
#define MetalJITCore_h

#include <stdint.h>
#include <stddef.h>

// El umbrella header C puro no arrastra dependencias de Foundation.
// MetalJITBridge.h (ObjC) solo se incluye cuando __OBJC__ esta definido
// (Swift/ObjC), pero no en C/C++ puro.
#if __has_include(<MetalJITCore/MetalJITKernels.h>)
#import <MetalJITCore/MetalJITKernels.h>
#import <MetalJITCore/MetalJITHeap.h>
#import <MetalJITCore/MetalJITCache.h>
#import <MetalJITCore/MetalJITArchive.h>
#else
#import "MetalJITKernels.h"
#import "MetalJITHeap.h"
#import "MetalJITCache.h"
#import "MetalJITArchive.h"
#endif
#ifdef __OBJC__
#if __has_include(<MetalJITCore/MetalJITBridge.h>)
#import <MetalJITCore/MetalJITBridge.h>
#else
#import "MetalJITBridge.h"
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =====================================================================
// CODIGOS DE ERROR UNIFICADOS
// =====================================================================
typedef enum {
    MJIT_SUCCESS                = 0,
    MJIT_ERR_UNINITIALIZED      = 101,
    MJIT_ERR_COMPILATION_FAILED = 102,
    MJIT_ERR_INVALID_BUFFER     = 103,
    MJIT_ERR_INVALID_HANDLE     = 104,
    MJIT_ERR_OVERFLOW           = 105,
    MJIT_ERR_UNDERFLOW          = 106,
    MJIT_ERR_TYPE_MISMATCH      = 107
} MJITErrorCode;

// =====================================================================
// TIPOS DE DATO SOPORTADOS
// =====================================================================
typedef enum {
    MJIT_TYPE_UINT64 = 0,
    MJIT_TYPE_FLOAT32 = 1,
    MJIT_TYPE_FLOAT64 = 2,
    MJIT_TYPE_INT32   = 3,
    MJIT_TYPE_INT64   = 4,
    MJIT_TYPE_FLOAT16 = 5
} MJITDataType;

// =====================================================================
// MANEJADOR DE PIPELINE (handle opaco > 0)
// =====================================================================
typedef int MJITPipelineHandle;
#define MJIT_INVALID_HANDLE 0

// =====================================================================
// FUNCION DE COMPUTO CPU (fallback definido por el usuario)
// =====================================================================
typedef void (*MJITCPUComputeFunc)(const void* input,
                                   void* output,
                                   int element_count,
                                   void* user_data);

// =====================================================================
// CALLBACK DE COMPLETADO ASINCRONO
// =====================================================================
typedef void (*MJITCompletionCallback)(int error_code, void* user_data);

// =====================================================================
// API DE COMPILACION JIT
// =====================================================================

// Compila un shader Metal JIT y devuelve un handle de pipeline (>0).
// Si shader_src y kernel_name son NULL, devuelve un pipeline CPU-only.
// error_msg_out (si no es NULL, min 256 bytes) recibe el mensaje de error.
MJITPipelineHandle mjit_compile_pipeline(const char* shader_src,
                                          const char* kernel_name,
                                          char* error_msg_out,
                                          int error_msg_capacity);

// Destruye un pipeline liberando los recursos Metal asociados.
int mjit_destroy_pipeline(MJITPipelineHandle handle);

// Registra una funcion de computo CPU como fallback para un pipeline.
int mjit_set_cpu_fallback(MJITPipelineHandle handle,
                           MJITCPUComputeFunc func,
                           void* user_data);

// =====================================================================
// API DE DESPACHO
// =====================================================================

// Despacho unitario: un buffer de entrada, uno de salida.
// Auto-ruteo: N >= 10000 -> GPU,  N < 10000 -> CPU (GCD + fallback).
int mjit_dispatch(MJITPipelineHandle handle,
                  const void* input_buffer,
                  void* output_buffer,
                  int element_count,
                  MJITDataType data_type);

// Despacho por lotes: multiples payloads procesados en paralelo.
// payloads: buffer plano con (num_batches * elements_per_payload) elementos.
// results:  buffer con num_batches elementos (un resultado por lote).
// En GPU cada hilo procesa un lote: offset = id * elements_per_payload.
int mjit_dispatch_batch(MJITPipelineHandle handle,
                        const void* payloads,
                        void* results,
                        int num_batches,
                        int elements_per_payload,
                        MJITDataType data_type);

// Utilidad: tamano en bytes de un elemento del tipo dado.
int mjit_data_type_size(MJITDataType data_type);

// Utilidad: nombre legible del tipo de dato.
const char* mjit_data_type_name(MJITDataType data_type);

// =====================================================================
// API DE SELECCION DE DISPOSITIVO GPU
// =====================================================================

// Numero de GPUs disponibles en el sistema.
int mjit_device_count(void);

// Cambia el dispositivo activo. 0 = default del sistema.
// Afecta a las compilaciones y despachos posteriores.
int mjit_select_device(int index);

// Nombre del dispositivo (ej. "Apple M1 Pro", "AMD Radeon Pro 5500M").
const char* mjit_device_name(int index);

// Devuelve 1 si el dispositivo no tiene pantalla (headless).
int mjit_device_is_headless(int index);

// Devuelve 1 si el dispositivo es externo (eGPU).
int mjit_device_is_external(int index);

// RegistryID unico del dispositivo (util para identificarlo entre reinicios).
uint64_t mjit_device_registry_id(int index);

// =====================================================================
// API DE DESPACHO ASINCRONO
// =====================================================================

// Despacho asincrono: retorna inmediatamente. El callback se invoca
// cuando la GPU termina (o inmediatamente en ruta CPU).
// El callback recibe el codigo de error (MJIT_SUCCESS si OK).
int mjit_dispatch_async(MJITPipelineHandle handle,
                         const void* input_buffer,
                         void* output_buffer,
                         int element_count,
                         MJITDataType data_type,
                         MJITCompletionCallback callback,
                         void* user_data);

// Batch asincrono.
int mjit_dispatch_batch_async(MJITPipelineHandle handle,
                               const void* payloads,
                               void* results,
                               int num_batches,
                               int elements_per_payload,
                               MJITDataType data_type,
                               MJITCompletionCallback callback,
                               void* user_data);

#ifdef __cplusplus
}
#endif

// Bindings de conveniencia C (wrappers tipados, macros)
// Debe ir al final porque depende de los tipos ya declarados.
#if __has_include(<MetalJITCore/MetalJITBinding.h>)
#include <MetalJITCore/MetalJITBinding.h>
#else
#include "MetalJITBinding.h"
#endif

#endif /* MetalJITCore_h */
