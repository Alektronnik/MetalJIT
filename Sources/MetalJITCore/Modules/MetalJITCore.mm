#import "MetalJITCore.h"
#import "MetalJITInternal.h"
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstring>
#include <cmath>
#include <dispatch/dispatch.h>

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

static id<MTLDevice>       g_gpu         = nil;
static id<MTLCommandQueue> g_cmd_queue   = nil;
static NSArray<id<MTLDevice>>* g_all_devices = nil;
#endif

// =====================================================================
// ESTRUCTURA DE PIPELINE (definida en MetalJITInternal.h)
// =====================================================================

// =====================================================================
// ESTADO GLOBAL THREAD-SAFE (exportado para modulos internos)
// =====================================================================
std::mutex                                 g_mutex;
std::atomic<int>                           g_next_handle{1};
std::unordered_map<int, MetalJITPipeline*> g_pipelines;

// =====================================================================
// UTILIDADES INTERNAS
// =====================================================================
static inline int data_type_stride(MJITDataType t) {
    switch (t) {
        case MJIT_TYPE_UINT64:  return 8;
        case MJIT_TYPE_FLOAT32: return 4;
        case MJIT_TYPE_FLOAT64: return 8;
        case MJIT_TYPE_INT32:   return 4;
        case MJIT_TYPE_INT64:   return 8;
        case MJIT_TYPE_FLOAT16: return 2;
        default:                return 8;
    }
}

// =====================================================================
// IMPLEMENTACION DE LA API
// =====================================================================
extern "C" {

// ---------------------------------------------------------------------
// UTILIDADES
// ---------------------------------------------------------------------
int mjit_data_type_size(MJITDataType t) {
    return data_type_stride(t);
}

const char* mjit_data_type_name(MJITDataType t) {
    switch (t) {
        case MJIT_TYPE_UINT64:  return "UInt64";
        case MJIT_TYPE_FLOAT32: return "Float32";
        case MJIT_TYPE_FLOAT64: return "Float64";
        case MJIT_TYPE_INT32:   return "Int32";
        case MJIT_TYPE_INT64:   return "Int64";
        case MJIT_TYPE_FLOAT16: return "Float16";
        default:                return "Unknown";
    }
}

// ---------------------------------------------------------------------
// COMPILACION JIT
// ---------------------------------------------------------------------
MJITPipelineHandle mjit_compile_pipeline(const char* shader_src,
                                          const char* kernel_name,
                                          char* error_msg_out,
                                          int error_msg_capacity) {
    std::lock_guard<std::mutex> lock(g_mutex);

    auto* pipeline = new MetalJITPipeline();
    pipeline->handle    = g_next_handle.fetch_add(1);
    pipeline->cpuFunc   = nullptr;
    pipeline->cpuUserData = nullptr;
    pipeline->dataType  = MJIT_TYPE_UINT64;
    pipeline->isGPU     = false;

#ifdef __OBJC__
    if (!g_gpu) {
        g_gpu = MTLCreateSystemDefaultDevice();
        if (g_gpu) {
            g_cmd_queue = [g_gpu newCommandQueue];
        }
    }

    // Ruta GPU: usuario proporciono fuente y kernel
    if (shader_src && kernel_name) {
        if (!g_gpu) {
            if (error_msg_out && error_msg_capacity > 0) {
                snprintf(error_msg_out, error_msg_capacity,
                         "Metal GPU no disponible para compilacion JIT.");
            }
            delete pipeline;
            return -MJIT_ERR_UNINITIALIZED;
        }

        NSError* err = nil;
        NSString* src = [NSString stringWithUTF8String:shader_src];
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];

        id<MTLLibrary> lib = [g_gpu newLibraryWithSource:src
                                                  options:opts
                                                    error:&err];
        if (err || !lib) {
            if (error_msg_out && error_msg_capacity > 0) {
                const char* desc = [[err localizedDescription] UTF8String];
                snprintf(error_msg_out, error_msg_capacity, "%s", desc ? desc : "unknown");
            }
            delete pipeline;
            return -MJIT_ERR_COMPILATION_FAILED;
        }

        NSString* kname = [NSString stringWithUTF8String:kernel_name];
        id<MTLFunction> func = [lib newFunctionWithName:kname];
        if (!func) {
            if (error_msg_out && error_msg_capacity > 0) {
                snprintf(error_msg_out, error_msg_capacity,
                         "Kernel '%s' no encontrado en el shader.", kernel_name);
            }
            delete pipeline;
            return -MJIT_ERR_COMPILATION_FAILED;
        }

        id<MTLComputePipelineState> pso = [g_gpu newComputePipelineStateWithFunction:func
                                                                               error:&err];
        if (err || !pso) {
            if (error_msg_out && error_msg_capacity > 0) {
                const char* desc = [[err localizedDescription] UTF8String];
                snprintf(error_msg_out, error_msg_capacity, "%s", desc ? desc : "unknown");
            }
            delete pipeline;
            return -MJIT_ERR_COMPILATION_FAILED;
        }

        pipeline->metalState = pso;
        pipeline->metalLib   = lib;
        pipeline->isGPU      = true;
    }
    // Ruta CPU: ambos parametros son nullptr, pipeline solo CPU
    // (solo se permite si ambos son null; un parametro parcial es error)
    else if (shader_src || kernel_name) {
        if (error_msg_out && error_msg_capacity > 0) {
            snprintf(error_msg_out, error_msg_capacity,
                     "Debe proporcionar shader y kernel, o ambos nullptr para CPU-only.");
        }
        delete pipeline;
        return -MJIT_ERR_COMPILATION_FAILED;
    }
#else
    // Sin soporte ObjC, solo se permiten pipelines CPU
    if (shader_src || kernel_name) {
        if (error_msg_out && error_msg_capacity > 0) {
            snprintf(error_msg_out, error_msg_capacity,
                     "Compilacion JIT requiere Metal (soporte ObjC no disponible).");
        }
        delete pipeline;
        return -MJIT_ERR_UNINITIALIZED;
    }
#endif

    g_pipelines[pipeline->handle] = pipeline;
    return pipeline->handle;
}

// ---------------------------------------------------------------------
// DESTRUIR PIPELINE
// ---------------------------------------------------------------------
int mjit_destroy_pipeline(MJITPipelineHandle handle) {
    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_pipelines.find(handle);
    if (it == g_pipelines.end()) return MJIT_ERR_INVALID_HANDLE;

    delete it->second;
    g_pipelines.erase(it);
    return MJIT_SUCCESS;
}

// ---------------------------------------------------------------------
// REGISTRAR FALLBACK CPU
// ---------------------------------------------------------------------
int mjit_set_cpu_fallback(MJITPipelineHandle handle,
                           MJITCPUComputeFunc func,
                           void* user_data) {
    std::lock_guard<std::mutex> lock(g_mutex);

    auto it = g_pipelines.find(handle);
    if (it == g_pipelines.end()) return MJIT_ERR_INVALID_HANDLE;

    it->second->cpuFunc     = func;
    it->second->cpuUserData = user_data;
    return MJIT_SUCCESS;
}

// ---------------------------------------------------------------------
// DESPACHO UNITARIO
// ---------------------------------------------------------------------
int mjit_dispatch(MJITPipelineHandle handle,
                  const void* input_buffer,
                  void* output_buffer,
                  int element_count,
                  MJITDataType data_type) {

    if (!input_buffer || !output_buffer || element_count <= 0) {
        return MJIT_ERR_INVALID_BUFFER;
    }

    int stride = data_type_stride(data_type);
    size_t buffer_bytes = (size_t)element_count * stride;

    // Capturar estado bajo lock
    MetalJITPipeline* ppl = nullptr;
    MJITCPUComputeFunc   cpuFn = nullptr;
    void*                cpuCtx = nullptr;
    bool                 useGPU = false;
#ifdef __OBJC__
    id<MTLDevice>              gpu = nil;
    id<MTLCommandQueue>        queue = nil;
    id<MTLComputePipelineState> pso = nil;
    NSUInteger                 execWidth = 0;
#endif

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_pipelines.find(handle);
        if (it == g_pipelines.end()) return MJIT_ERR_INVALID_HANDLE;
        ppl = it->second;
        ppl->dataType = data_type;
        cpuFn = ppl->cpuFunc;
        cpuCtx = ppl->cpuUserData;
#ifdef __OBJC__
        if (element_count >= 10000 && g_gpu && ppl->isGPU) {
            useGPU = true;
            gpu = g_gpu;
            queue = g_cmd_queue;
            pso = ppl->metalState;
            execWidth = pso.threadExecutionWidth;
        }
#endif
    } // lock liberado aqui

#ifdef __OBJC__
    if (useGPU) {
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        id<MTLBuffer> mtlIn = [gpu newBufferWithBytesNoCopy:(void*)input_buffer
                                                   length:buffer_bytes
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        id<MTLBuffer> mtlOut = [gpu newBufferWithBytesNoCopy:output_buffer
                                                    length:buffer_bytes
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
        [enc setBuffer:mtlIn  offset:0 atIndex:0];
        [enc setBuffer:mtlOut offset:0 atIndex:1];
        MTLSize grid   = MTLSizeMake(element_count, 1, 1);
        MTLSize groups = MTLSizeMake(MIN((NSUInteger)element_count, execWidth), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:groups];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.error) {
            return MJIT_ERR_COMPILATION_FAILED;
        }
        return MJIT_SUCCESS;
    }
#endif

    // Ruta CPU (GCD + fallback con stride correcto, sin lock)
    if (cpuFn) {
        cpuFn(input_buffer, output_buffer, element_count, cpuCtx);
    } else {
        int s = stride;
        const char* src = (const char*)input_buffer;
        char* dst = (char*)output_buffer;
        dispatch_apply(element_count, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t i) {
            memcpy(dst + i * s, src + i * s, s);
        });
    }

    return MJIT_SUCCESS;
}

// ---------------------------------------------------------------------
// DESPACHO POR LOTES (BATCH)
// ---------------------------------------------------------------------
int mjit_dispatch_batch(MJITPipelineHandle handle,
                        const void* payloads,
                        void* results,
                        int num_batches,
                        int elements_per_payload,
                        MJITDataType data_type) {

    if (!payloads || !results || num_batches <= 0 || elements_per_payload <= 0) {
        return MJIT_ERR_INVALID_BUFFER;
    }

    int stride = data_type_stride(data_type);

    // Capturar estado bajo lock
    MetalJITPipeline* ppl = nullptr;
    MJITCPUComputeFunc   cpuFn = nullptr;
    void*                cpuCtx = nullptr;
    bool                 useGPU = false;
    int ep = elements_per_payload;
#ifdef __OBJC__
    id<MTLDevice>              gpu = nil;
    id<MTLCommandQueue>        queue = nil;
    id<MTLComputePipelineState> pso = nil;
    NSUInteger                 execWidth = 0;
#endif

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_pipelines.find(handle);
        if (it == g_pipelines.end()) return MJIT_ERR_INVALID_HANDLE;
        ppl = it->second;
        cpuFn = ppl->cpuFunc;
        cpuCtx = ppl->cpuUserData;
#ifdef __OBJC__
        if (num_batches >= 10000 && g_gpu && ppl->isGPU) {
            useGPU = true;
            gpu = g_gpu;
            queue = g_cmd_queue;
            pso = ppl->metalState;
            execWidth = pso.threadExecutionWidth;
        }
#endif
    } // lock liberado

#ifdef __OBJC__
    if (useGPU) {
        size_t payload_bytes = (size_t)num_batches * ep * stride;
        size_t result_bytes  = (size_t)num_batches * stride;
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        id<MTLBuffer> mtlIn = [gpu newBufferWithBytesNoCopy:(void*)payloads length:payload_bytes options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> mtlOut = [gpu newBufferWithBytesNoCopy:results length:result_bytes options:MTLResourceStorageModeShared deallocator:nil];
        [enc setBuffer:mtlIn  offset:0 atIndex:0];
        [enc setBuffer:mtlOut offset:0 atIndex:1];
        [enc setBytes:&ep length:sizeof(int) atIndex:2];
        MTLSize grid   = MTLSizeMake(num_batches, 1, 1);
        MTLSize groups = MTLSizeMake(MIN((NSUInteger)num_batches, execWidth), 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:groups];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.error) {
            return MJIT_ERR_COMPILATION_FAILED;
        }
        return MJIT_SUCCESS;
    }
#endif

    // Ruta CPU batch (sin lock)
    const char* src = (const char*)payloads;
    char*       dst = (char*)results;

    if (cpuFn) {
        dispatch_apply(num_batches, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t i) {
            cpuFn(src + i * ep * stride, dst + i * stride, ep, cpuCtx);
        });
    } else {
        int s = stride;
        dispatch_apply(num_batches, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t i) {
            memcpy(dst + i * s, src + i * ep * s, s);
        });
    }

    return MJIT_SUCCESS;
}

} // extern "C"

// =====================================================================
// DESPACHO ASINCRONO
// =====================================================================
extern "C" {

int mjit_dispatch_async(MJITPipelineHandle handle,
                         const void* input_buffer,
                         void* output_buffer,
                         int element_count,
                         MJITDataType data_type,
                         MJITCompletionCallback callback,
                         void* user_data) {

    if (!input_buffer || !output_buffer || element_count <= 0) {
        if (callback) callback(MJIT_ERR_INVALID_BUFFER, user_data);
        return MJIT_ERR_INVALID_BUFFER;
    }

    int stride = data_type_stride(data_type);
    size_t buffer_bytes = (size_t)element_count * stride;

    // Capturar estado bajo lock, liberar antes de GPU ops
    MJITCPUComputeFunc localCpuFn = nullptr;
    void* localCpuCtx = nullptr;
    bool useGPU = false;
#ifdef __OBJC__
    id<MTLDevice>              gpu = nil;
    id<MTLCommandQueue>        queue = nil;
    id<MTLComputePipelineState> pso = nil;
    NSUInteger                 execWidth = 0;
#endif

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_pipelines.find(handle);
        if (it == g_pipelines.end()) {
            if (callback) callback(MJIT_ERR_INVALID_HANDLE, user_data);
            return MJIT_ERR_INVALID_HANDLE;
        }
        MetalJITPipeline* ppl = it->second;
        localCpuFn = ppl->cpuFunc;
        localCpuCtx = ppl->cpuUserData;
#ifdef __OBJC__
        if (element_count >= 10000 && g_gpu && ppl->isGPU) {
            useGPU = true;
            gpu = g_gpu;
            queue = g_cmd_queue;
            pso = ppl->metalState;
            execWidth = pso.threadExecutionWidth;
        }
#endif
    } // lock liberado aqui

#ifdef __OBJC__
    if (useGPU) {
        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];

        id<MTLBuffer> mtlIn = [gpu newBufferWithBytesNoCopy:(void*)input_buffer
                                                     length:buffer_bytes
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
        id<MTLBuffer> mtlOut = [gpu newBufferWithBytesNoCopy:output_buffer
                                                      length:buffer_bytes
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];

        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> buf) {
            int code = buf.error ? MJIT_ERR_COMPILATION_FAILED : MJIT_SUCCESS;
            if (callback) callback(code, user_data);
            (void)mtlIn; (void)mtlOut;
        }];

        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:mtlIn  offset:0 atIndex:0];
        [enc setBuffer:mtlOut offset:0 atIndex:1];

        MTLSize grid   = MTLSizeMake(element_count, 1, 1);
        MTLSize groups = MTLSizeMake(MIN((NSUInteger)element_count, execWidth), 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:groups];
        [enc endEncoding];
        [cmdBuf commit];

        return MJIT_SUCCESS;
    }
#endif

    // Ruta CPU async
    int s = stride;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
        if (localCpuFn) {
            localCpuFn(input_buffer, output_buffer, element_count, localCpuCtx);
        } else {
            const char* src = (const char*)input_buffer;
            char* dst = (char*)output_buffer;
            dispatch_apply(element_count, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t i) {
                memcpy(dst + i * s, src + i * s, s);
            });
        }
        if (callback) callback(MJIT_SUCCESS, user_data);
    });

    return MJIT_SUCCESS;
}

int mjit_dispatch_batch_async(MJITPipelineHandle handle,
                               const void* payloads,
                               void* results,
                               int num_batches,
                               int elements_per_payload,
                               MJITDataType data_type,
                               MJITCompletionCallback callback,
                               void* user_data) {

    if (!payloads || !results || num_batches <= 0 || elements_per_payload <= 0) {
        if (callback) callback(MJIT_ERR_INVALID_BUFFER, user_data);
        return MJIT_ERR_INVALID_BUFFER;
    }

    int stride = data_type_stride(data_type);

    // Capturar estado bajo lock, liberar antes de GPU ops
    MJITCPUComputeFunc localCpuFn = nullptr;
    void* localCpuCtx = nullptr;
    bool useGPU = false;
    int ep = elements_per_payload;
#ifdef __OBJC__
    id<MTLDevice>              gpu = nil;
    id<MTLCommandQueue>        queue = nil;
    id<MTLComputePipelineState> pso = nil;
    NSUInteger                 execWidth = 0;
#endif

    {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_pipelines.find(handle);
        if (it == g_pipelines.end()) {
            if (callback) callback(MJIT_ERR_INVALID_HANDLE, user_data);
            return MJIT_ERR_INVALID_HANDLE;
        }
        MetalJITPipeline* ppl = it->second;
        localCpuFn = ppl->cpuFunc;
        localCpuCtx = ppl->cpuUserData;
#ifdef __OBJC__
        if (num_batches >= 10000 && g_gpu && ppl->isGPU) {
            useGPU = true;
            gpu = g_gpu;
            queue = g_cmd_queue;
            pso = ppl->metalState;
            execWidth = pso.threadExecutionWidth;
        }
#endif
    } // lock liberado

#ifdef __OBJC__
    if (useGPU) {
        size_t payload_bytes = (size_t)num_batches * ep * stride;
        size_t result_bytes  = (size_t)num_batches * stride;

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];

        id<MTLBuffer> mtlIn = [gpu newBufferWithBytesNoCopy:(void*)payloads
                                                     length:payload_bytes
                                                    options:MTLResourceStorageModeShared
                                                deallocator:nil];
        id<MTLBuffer> mtlOut = [gpu newBufferWithBytesNoCopy:results
                                                      length:result_bytes
                                                     options:MTLResourceStorageModeShared
                                                 deallocator:nil];

        [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> buf) {
            int code = buf.error ? MJIT_ERR_COMPILATION_FAILED : MJIT_SUCCESS;
            if (callback) callback(code, user_data);
            (void)mtlIn; (void)mtlOut;
        }];

        id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:mtlIn  offset:0 atIndex:0];
        [enc setBuffer:mtlOut offset:0 atIndex:1];
        [enc setBytes:&ep length:sizeof(int) atIndex:2];

        MTLSize grid   = MTLSizeMake(num_batches, 1, 1);
        MTLSize groups = MTLSizeMake(MIN((NSUInteger)num_batches, execWidth), 1, 1);

        [enc dispatchThreads:grid threadsPerThreadgroup:groups];
        [enc endEncoding];
        [cmdBuf commit];

        return MJIT_SUCCESS;
    }
#endif

    // CPU batch async
    const char* src = (const char*)payloads;
    char*       dst = (char*)results;
    int s = stride;

    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^{
        if (localCpuFn) {
            dispatch_apply(num_batches,
                           dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                           ^(size_t i) {
                const void* in  = src + i * ep * s;
                void*       out = dst + i * s;
                localCpuFn(in, out, ep, localCpuCtx);
            });
        } else {
            dispatch_apply(num_batches,
                           dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                           ^(size_t i) {
                memcpy(dst + i * s, src + i * ep * s, s);
            });
        }
        if (callback) callback(MJIT_SUCCESS, user_data);
    });

    return MJIT_SUCCESS;
}

} // extern "C"

// =====================================================================
// API DE DISPOSITIVO GPU
// =====================================================================
extern "C" {

static void _ensure_devices() {
#ifdef __OBJC__
    if (!g_all_devices) {
        g_all_devices = MTLCopyAllDevices();
        id<MTLDevice> def = MTLCreateSystemDefaultDevice();
        if (def && ![g_all_devices containsObject:def]) {
            g_all_devices = [@[def] arrayByAddingObjectsFromArray:g_all_devices];
        }
    }
#endif
}

int mjit_device_count(void) {
    _ensure_devices();
#ifdef __OBJC__
    return (int)[g_all_devices count];
#else
    return 0;
#endif
}

int mjit_select_device(int index) {
    std::lock_guard<std::mutex> lock(g_mutex);
#ifdef __OBJC__
    _ensure_devices();
    if (index < 0 || index >= (int)[g_all_devices count]) return MJIT_ERR_INVALID_BUFFER;
    g_gpu = [g_all_devices objectAtIndex:index];
    g_cmd_queue = [g_gpu newCommandQueue];
    return MJIT_SUCCESS;
#else
    return MJIT_ERR_UNINITIALIZED;
#endif
}

const char* mjit_device_name(int index) {
    _ensure_devices();
#ifdef __OBJC__
    if (index < 0 || index >= (int)[g_all_devices count]) return "unknown";
    return [[[g_all_devices objectAtIndex:index] name] UTF8String];
#else
    return "no-gpu";
#endif
}

int mjit_device_is_headless(int index) {
    _ensure_devices();
#ifdef __OBJC__
    if (index < 0 || index >= (int)[g_all_devices count]) return 0;
    return [[g_all_devices objectAtIndex:index] isHeadless] ? 1 : 0;
#else
    return 0;
#endif
}

int mjit_device_is_external(int index) {
    _ensure_devices();
#ifdef __OBJC__
    if (index < 0 || index >= (int)[g_all_devices count]) return 0;
    if (@available(macOS 10.15, *)) {
        return [[g_all_devices objectAtIndex:index] isRemovable] ? 1 : 0;
    }
    return 0;
#else
    return 0;
#endif
}

uint64_t mjit_device_registry_id(int index) {
    _ensure_devices();
#ifdef __OBJC__
    if (index < 0 || index >= (int)[g_all_devices count]) return 0;
    return [[g_all_devices objectAtIndex:index] registryID];
#else
    return 0;
#endif
}

} // extern "C"

// Registro interno de pipeline (C++ linkage, usa tipos ObjC)
int mjit_register_pipeline(MetalJITPipeline* pipeline) {
    std::lock_guard<std::mutex> lock(g_mutex);
    pipeline->handle = g_next_handle.fetch_add(1);
    g_pipelines[pipeline->handle] = pipeline;
    return pipeline->handle;
}
