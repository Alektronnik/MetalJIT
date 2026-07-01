// ==========================================================================
// MetalJITArchive.mm — MTLBinaryArchive: compilacion AOT con cache binario
// ==========================================================================

#import "MetalJITArchive.h"
#import "MetalJITInternal.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstdio>

static id<MTLDevice> _archive_device() {
    return MTLCreateSystemDefaultDevice();
}

extern "C" {

int mjit_archive_save(const char* shader_src,
                       const char* kernel_name,
                       const char* archive_path,
                       char* error_msg_out,
                       int error_msg_capacity) {

    id<MTLDevice> device = _archive_device();
    if (!device) return -MJIT_ERR_UNINITIALIZED;

    NSString* src   = [NSString stringWithUTF8String:shader_src];
    NSString* kname = [NSString stringWithUTF8String:kernel_name];
    NSString* path  = [NSString stringWithUTF8String:archive_path];

    NSString* dir = [path stringByDeletingLastPathComponent];
    [[NSFileManager defaultManager] createDirectoryAtPath:dir
                              withIntermediateDirectories:YES attributes:nil error:nil];

    NSError* err = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
    if (err) {
        if (error_msg_out && error_msg_capacity > 0)
            snprintf(error_msg_out, error_msg_capacity, "%s", [[err localizedDescription] UTF8String]);
        return -MJIT_ERR_COMPILATION_FAILED;
    }

    id<MTLFunction> func = [lib newFunctionWithName:kname];
    if (!func) {
        if (error_msg_out && error_msg_capacity > 0)
            snprintf(error_msg_out, error_msg_capacity, "Kernel '%s' no encontrado.", kernel_name);
        return -MJIT_ERR_COMPILATION_FAILED;
    }

    MTLComputePipelineDescriptor* pipeDesc = [[MTLComputePipelineDescriptor alloc] init];
    pipeDesc.computeFunction = func;

    // Archive nuevo: NO se pone URL en el descriptor (daria "Invalid URL" si el
    // archivo no existe). Se crea vacio, se agregan pipelines, y se serializa.
    MTLBinaryArchiveDescriptor* archDesc = [[MTLBinaryArchiveDescriptor alloc] init];

    id<MTLBinaryArchive> archive = [device newBinaryArchiveWithDescriptor:archDesc error:&err];
    if (err || !archive) {
        if (error_msg_out && error_msg_capacity > 0)
            snprintf(error_msg_out, error_msg_capacity, "%s", [[err localizedDescription] UTF8String]);
        return -MJIT_ERR_COMPILATION_FAILED;
    }

    BOOL added = [archive addComputePipelineFunctionsWithDescriptor:pipeDesc error:&err];
    if (!added || err) {
        if (error_msg_out && error_msg_capacity > 0)
            snprintf(error_msg_out, error_msg_capacity, "%s", [[err localizedDescription] UTF8String]);
        return -MJIT_ERR_COMPILATION_FAILED;
    }

    // Persistir el archive a disco
    BOOL saved = [archive serializeToURL:[NSURL fileURLWithPath:path] error:&err];
    if (!saved || err) {
        if (error_msg_out && error_msg_capacity > 0)
            snprintf(error_msg_out, error_msg_capacity, "%s", [[err localizedDescription] UTF8String]);
        return -MJIT_ERR_COMPILATION_FAILED;
    }
    return MJIT_SUCCESS;
}

int mjit_compile_with_archive(const char* shader_src,
                               const char* kernel_name,
                               const char* archive_path,
                               char* error_msg_out,
                               int error_msg_capacity) {

    id<MTLDevice> device = _archive_device();
    if (!device) return -MJIT_ERR_UNINITIALIZED;

    NSString* src   = [NSString stringWithUTF8String:shader_src];
    NSString* kname = [NSString stringWithUTF8String:kernel_name];
    NSString* path  = [NSString stringWithUTF8String:archive_path];

    NSError* err = nil;
    MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
    id<MTLLibrary> lib = [device newLibraryWithSource:src options:opts error:&err];
    if (err) {
        if (error_msg_out && error_msg_capacity > 0)
            snprintf(error_msg_out, error_msg_capacity, "%s", [[err localizedDescription] UTF8String]);
        return -MJIT_ERR_COMPILATION_FAILED;
    }

    id<MTLFunction> func = [lib newFunctionWithName:kname];
    if (!func) {
        if (error_msg_out && error_msg_capacity > 0)
            snprintf(error_msg_out, error_msg_capacity, "Kernel '%s' no encontrado.", kernel_name);
        return -MJIT_ERR_COMPILATION_FAILED;
    }

    // Intentar cargar archive existente (cache binario GPU)
    id<MTLBinaryArchive> archive = nil;
    if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
        MTLBinaryArchiveDescriptor* archDesc = [[MTLBinaryArchiveDescriptor alloc] init];
        archDesc.url = [NSURL fileURLWithPath:path];
        archive = [device newBinaryArchiveWithDescriptor:archDesc error:nil];
    }

    // Construir PSO (con archive si existe = sin recompilacion GPU)
    MTLComputePipelineDescriptor* psoDesc = [[MTLComputePipelineDescriptor alloc] init];
    psoDesc.computeFunction = func;
    if (archive) psoDesc.binaryArchives = @[archive];

    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithDescriptor:psoDesc
                                                                             options:MTLPipelineOptionNone
                                                                          reflection:nil
                                                                               error:&err];
    if (err) {
        if (error_msg_out && error_msg_capacity > 0)
            snprintf(error_msg_out, error_msg_capacity, "%s", [[err localizedDescription] UTF8String]);
        return -MJIT_ERR_COMPILATION_FAILED;
    }

    // Si no habia archive, guardar para futuras ejecuciones
    if (!archive) {
        int save_rc = mjit_archive_save(shader_src, kernel_name, archive_path,
                                         error_msg_out, error_msg_capacity);
        // Si falla el guardado, reportarlo pero no bloquear el pipeline:
        // la compilacion JIT ya tuvo exito, solo la cache futura queda sin efecto.
        if (save_rc != MJIT_SUCCESS && error_msg_out && error_msg_capacity > 0) {
            // error_msg_out ya fue escrito por mjit_archive_save
        }
    }

    auto* pipeline = new MetalJITPipeline();
    pipeline->metalState  = pso;
    pipeline->metalLib    = lib;
    pipeline->cpuFunc     = nullptr;
    pipeline->cpuUserData = nullptr;
    pipeline->dataType    = MJIT_TYPE_UINT64;
    pipeline->isGPU       = true;
    return mjit_register_pipeline(pipeline);
}

int mjit_archive_exists(const char* archive_path) {
    return [[NSFileManager defaultManager] fileExistsAtPath:[NSString stringWithUTF8String:archive_path]] ? 1 : 0;
}

} // extern "C"
