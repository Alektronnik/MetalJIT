#import "MetalJITBridge.h"
#import "MetalJITCore.h"
#import "MetalJITKernels.h"

NSString * const MetalJITErrorDomain = @"com.metaljit.MetalJIT";

// =====================================================================
// MetalJITPipelineHandle
// =====================================================================
@implementation MetalJITPipelineHandle
- (instancetype)initWithHandle:(int)h {
    self = [super init];
    if (self) { _handle = h; }
    return self;
}
@end

// =====================================================================
// MetalJITBridge
// =====================================================================
@implementation MetalJITBridge

// --- Compilacion ---

+ (nullable MetalJITPipelineHandle *)compilePipelineWithShader:(nullable NSString *)shaderSource
                                                     kernelName:(nullable NSString *)kernelName
                                                          error:(NSError **)error {

    const char *c_shader = shaderSource ? [shaderSource UTF8String] : nullptr;
    const char *c_kernel = kernelName   ? [kernelName UTF8String]   : nullptr;

    char errorBuf[512] = {0};
    int handle = mjit_compile_pipeline(c_shader, c_kernel, errorBuf, sizeof(errorBuf));

    if (handle < 0) {
        if (error) {
            int code = -handle;  // convertir negativo a positivo para NSError
            NSString *desc = errorBuf[0]
                ? [NSString stringWithUTF8String:errorBuf]
                : @"MetalJIT: fallo al compilar shader Metal JIT.";
            *error = [NSError errorWithDomain:MetalJITErrorDomain
                                         code:code
                                     userInfo:@{ NSLocalizedDescriptionKey: desc }];
        }
        return nil;
    }

    return [[MetalJITPipelineHandle alloc] initWithHandle:handle];
}

// --- Destruccion ---

+ (BOOL)destroyPipeline:(MetalJITPipelineHandle *)pipelineHandle
                  error:(NSError **)error {

    int result = mjit_destroy_pipeline(pipelineHandle.handle);
    if (result != MJIT_SUCCESS) {
        if (error) {
            *error = [NSError errorWithDomain:MetalJITErrorDomain
                                         code:result
                                     userInfo:@{ NSLocalizedDescriptionKey: @"MetalJIT: handle de pipeline invalido." }];
        }
        return NO;
    }
    return YES;
}

// --- Registro de CPU fallback para built-in kernels ---

+ (BOOL)setBuiltInCPUFallback:(MetalJITPipelineHandle *)pipelineHandle
                   kernelType:(int)kernelType
                        error:(NSError **)error {

    MJITCPUComputeFunc func = nullptr;
    switch (kernelType) {
        case 0: func = mjit_cpu_crypto;    break;
        case 1: func = mjit_cpu_tensor;    break;
        case 2: func = mjit_cpu_logic;     break;
        case 3: func = mjit_cpu_physics;   break;
        case 4: func = mjit_cpu_topology;  break;
        default:
            if (error) {
                *error = [NSError errorWithDomain:MetalJITErrorDomain
                                             code:MetalJITBridgeErrorTypeMismatch
                                         userInfo:@{ NSLocalizedDescriptionKey:
                                             @"MetalJIT: tipo de built-in kernel invalido." }];
            }
            return NO;
    }

    int result = mjit_set_cpu_fallback(pipelineHandle.handle, func, nullptr);
    if (result != MJIT_SUCCESS) {
        if (error) {
            *error = [NSError errorWithDomain:MetalJITErrorDomain code:result userInfo:@{
                NSLocalizedDescriptionKey: @"MetalJIT: fallo al registrar CPU fallback." }];
        }
        return NO;
    }
    return YES;
}

+ (nullable MetalJITPipelineHandle *)wrapExistingHandle:(int)handle {
    if (handle <= 0) return nil;
    return [[MetalJITPipelineHandle alloc] initWithHandle:handle];
}

// --- Despacho unitario ---

+ (BOOL)dispatchWithPipeline:(MetalJITPipelineHandle *)pipelineHandle
                       input:(const void *)inputBuffer
                      output:(void *)outputBuffer
               elementCount:(int)elementCount
                   dataType:(int)dataType
                       error:(NSError **)error {

    int result = mjit_dispatch(pipelineHandle.handle,
                                inputBuffer, outputBuffer,
                                elementCount,
                                (MJITDataType)dataType);

    if (result != MJIT_SUCCESS) {
        if (error) {
            *error = [NSError errorWithDomain:MetalJITErrorDomain
                                         code:result
                                     userInfo:@{ NSLocalizedDescriptionKey:
                                         [NSString stringWithFormat:@"MetalJIT: dispatch fallido (codigo %d).", result] }];
        }
        return NO;
    }
    return YES;
}

// --- Despacho batch ---

+ (BOOL)dispatchBatchWithPipeline:(MetalJITPipelineHandle *)pipelineHandle
                         payloads:(const void *)payloads
                          results:(void *)results
                      numBatches:(int)numBatches
              elementsPerPayload:(int)elementsPerPayload
                         dataType:(int)dataType
                            error:(NSError **)error {

    int result = mjit_dispatch_batch(pipelineHandle.handle,
                                      payloads, results,
                                      numBatches, elementsPerPayload,
                                      (MJITDataType)dataType);

    if (result != MJIT_SUCCESS) {
        if (error) {
            *error = [NSError errorWithDomain:MetalJITErrorDomain
                                         code:result
                                     userInfo:@{ NSLocalizedDescriptionKey:
                                         [NSString stringWithFormat:@"MetalJIT: batch dispatch fallido (codigo %d).", result] }];
        }
        return NO;
    }
    return YES;
}

// --- Despacho asincrono ---

// Trampolines: convierten callback C → bloque ObjC
static void _async_trampoline(int errorCode, void* ctx) {
    void (^block)(NSError*) = (__bridge_transfer void(^)(NSError*))ctx;
    NSError* err = nil;
    if (errorCode != MJIT_SUCCESS) {
        err = [NSError errorWithDomain:MetalJITErrorDomain
                                  code:errorCode
                              userInfo:@{ NSLocalizedDescriptionKey:
                                  [NSString stringWithFormat:@"MetalJIT async: error %d", errorCode] }];
    }
    dispatch_async(dispatch_get_main_queue(), ^{ block(err); });
}

+ (void)dispatchAsyncWithPipeline:(MetalJITPipelineHandle *)pipelineHandle
                            input:(const void *)inputBuffer
                           output:(void *)outputBuffer
                    elementCount:(int)elementCount
                        dataType:(int)dataType
               completionHandler:(void (^)(NSError * _Nullable error))handler {

    void* ctx = (__bridge_retained void*)[handler copy];
    mjit_dispatch_async(pipelineHandle.handle,
                         inputBuffer, outputBuffer,
                         elementCount, (MJITDataType)dataType,
                         _async_trampoline, ctx);
}

+ (void)dispatchBatchAsyncWithPipeline:(MetalJITPipelineHandle *)pipelineHandle
                              payloads:(const void *)payloads
                               results:(void *)results
                           numBatches:(int)numBatches
                   elementsPerPayload:(int)elementsPerPayload
                              dataType:(int)dataType
                    completionHandler:(void (^)(NSError * _Nullable error))handler {

    void* ctx = (__bridge_retained void*)[handler copy];
    mjit_dispatch_batch_async(pipelineHandle.handle,
                               payloads, results,
                               numBatches, elementsPerPayload,
                               (MJITDataType)dataType,
                               _async_trampoline, ctx);
}

@end
