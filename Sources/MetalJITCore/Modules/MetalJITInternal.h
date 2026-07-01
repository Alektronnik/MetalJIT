// ==========================================================================
// MetalJITInternal.h — Estado interno compartido (NO publico)
// ==========================================================================
// Forward declarations para modulos internos que necesitan registrar
// pipelines sin pasar por mjit_compile_pipeline (ej. MetalJITCache).
// ==========================================================================

#ifndef MetalJITInternal_h
#define MetalJITInternal_h

#include <stdint.h>
#include <unordered_map>
#include <mutex>
#include <atomic>

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#include "MetalJITCore.h"

// Estructura C++ con miembros ObjC (definicion completa)
struct MetalJITPipeline {
    int handle;
#ifdef __OBJC__
    id<MTLComputePipelineState> metalState;
    id<MTLLibrary>              metalLib;
#endif
    MJITCPUComputeFunc cpuFunc;
    void*              cpuUserData;
    MJITDataType       dataType;
    bool               isGPU;
};

extern std::mutex                                 g_mutex;
extern std::atomic<int>                           g_next_handle;
extern std::unordered_map<int, MetalJITPipeline*> g_pipelines;

int mjit_register_pipeline(MetalJITPipeline* pipeline);

#endif /* MetalJITInternal_h */
