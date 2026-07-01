// ==========================================================================
// MetalJITCache.mm — Cache de shaders Metal (fuente verificada + PSO cache)
// ==========================================================================
// Guarda fuente MSL verificada (compila y valida antes de guardar).
// La carga recompila JIT; en futuras versiones se integrara MTLBinaryArchive
// para cachear los PSO y evitar recompilacion de pipeline state.
//
// Nota: No existe API runtime para serializar un MTLLibrary a .metallib.
// La via oficial es usar herramientas offline (metal -c) o MTLBinaryArchive
// para acelerar la creacion de PSO, pero la compilacion de fuente MSL
// siempre es necesaria en modo JIT.
// ==========================================================================

#import "MetalJITCache.h"
#import "MetalJITCore.h"
#import "MetalJITInternal.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>

extern "C" {

int mjit_cache_compile_and_save(const char* shader_src,
                                 const char* cache_path,
                                 char* error_msg_out,
                                 int error_msg_capacity) {
    if (!shader_src || !cache_path) {
        if (error_msg_out && error_msg_capacity > 0) {
            snprintf(error_msg_out, error_msg_capacity,
                     "Parametros shader_src y cache_path son obligatorios.");
        }
        return -MJIT_ERR_INVALID_BUFFER;
    }

#ifdef __OBJC__
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        if (error_msg_out && error_msg_capacity > 0) {
            snprintf(error_msg_out, error_msg_capacity, "Metal GPU no disponible.");
        }
        return -MJIT_ERR_UNINITIALIZED;
    }

    NSString* path = [NSString stringWithUTF8String:cache_path];
    NSString* dir  = [path stringByDeletingLastPathComponent];

    // Crear directorio si no existe
    NSError* err = nil;
    [[NSFileManager defaultManager] createDirectoryAtPath:dir
                              withIntermediateDirectories:YES
                                               attributes:nil
                                                    error:&err];
    if (err) {
        if (error_msg_out && error_msg_capacity > 0) {
            snprintf(error_msg_out, error_msg_capacity, "%s",
                     [[err localizedDescription] UTF8String]);
        }
        return -MJIT_ERR_COMPILATION_FAILED;
    }

    // Verificar que el shader compila antes de guardarlo
    NSString* src = [NSString stringWithUTF8String:shader_src];
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [device newLibraryWithSource:src
                                               options:opts
                                                 error:&err];
    if (err || !lib) {
        if (error_msg_out && error_msg_capacity > 0) {
            const char* desc = [[err localizedDescription] UTF8String];
            snprintf(error_msg_out, error_msg_capacity, "%s", desc ? desc : "unknown");
        }
        return -MJIT_ERR_COMPILATION_FAILED;
    }

    // Guardar fuente verificada como texto
    BOOL ok = [src writeToFile:path atomically:YES encoding:NSUTF8StringEncoding error:&err];
    if (!ok || err) {
        if (error_msg_out && error_msg_capacity > 0) {
            snprintf(error_msg_out, error_msg_capacity, "%s",
                     [[err localizedDescription] UTF8String]);
        }
        return -MJIT_ERR_COMPILATION_FAILED;
    }
    return MJIT_SUCCESS;
#else
    (void)shader_src; (void)cache_path; (void)error_msg_out; (void)error_msg_capacity;
    return -MJIT_ERR_UNINITIALIZED;
#endif
}

int mjit_cache_load_library(const char* cache_path,
                             const char* kernel_name,
                             char* error_msg_out,
                             int error_msg_capacity) {
    if (!cache_path || !kernel_name) {
        if (error_msg_out && error_msg_capacity > 0) {
            snprintf(error_msg_out, error_msg_capacity,
                     "Parametros cache_path y kernel_name son obligatorios.");
        }
        return -MJIT_ERR_INVALID_BUFFER;
    }

#ifdef __OBJC__
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return -MJIT_ERR_UNINITIALIZED;

    NSString* path = [NSString stringWithUTF8String:cache_path];
    if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
        if (error_msg_out && error_msg_capacity > 0) {
            snprintf(error_msg_out, error_msg_capacity, "Cache no encontrado: %s", cache_path);
        }
        return -MJIT_ERR_COMPILATION_FAILED;
    }

    // Leer fuente MSL del archivo de cache
    NSError* err = nil;
    NSString* src = [NSString stringWithContentsOfFile:path
                                               encoding:NSUTF8StringEncoding
                                                  error:&err];
    if (err || !src) {
        if (error_msg_out && error_msg_capacity > 0) {
            snprintf(error_msg_out, error_msg_capacity, "Error leyendo cache: %s",
                     [[err localizedDescription] UTF8String]);
        }
        return -MJIT_ERR_COMPILATION_FAILED;
    }

    // Recompilar JIT desde fuente cacheada
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [device newLibraryWithSource:src
                                               options:opts
                                                 error:&err];
    if (err || !lib) {
        if (error_msg_out && error_msg_capacity > 0) {
            snprintf(error_msg_out, error_msg_capacity, "%s",
                     [[err localizedDescription] UTF8String]);
        }
        return -MJIT_ERR_COMPILATION_FAILED;
    }

    NSString* kname = [NSString stringWithUTF8String:kernel_name];
    id<MTLFunction> func = [lib newFunctionWithName:kname];
    if (!func) {
        if (error_msg_out && error_msg_capacity > 0) {
            snprintf(error_msg_out, error_msg_capacity,
                     "Kernel '%s' no encontrado en la libreria cacheada.", kernel_name);
        }
        return -MJIT_ERR_COMPILATION_FAILED;
    }

    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func
                                                                             error:&err];
    if (err || !pso) {
        if (error_msg_out && error_msg_capacity > 0) {
            const char* desc = [[err localizedDescription] UTF8String];
            snprintf(error_msg_out, error_msg_capacity, "%s", desc ? desc : "unknown");
        }
        return -MJIT_ERR_COMPILATION_FAILED;
    }

    auto* pipeline = new MetalJITPipeline();
    pipeline->metalState  = pso;
    pipeline->metalLib    = lib;
    pipeline->cpuFunc     = nullptr;
    pipeline->cpuUserData = nullptr;
    pipeline->dataType    = MJIT_TYPE_UINT64;
    pipeline->isGPU       = true;

    return mjit_register_pipeline(pipeline);
#else
    return -MJIT_ERR_UNINITIALIZED;
#endif
}

int mjit_cache_exists(const char* cache_path) {
#ifdef __OBJC__
    NSString* path = [NSString stringWithUTF8String:cache_path];
    return [[NSFileManager defaultManager] fileExistsAtPath:path] ? 1 : 0;
#else
    return 0;
#endif
}

} // extern "C"
