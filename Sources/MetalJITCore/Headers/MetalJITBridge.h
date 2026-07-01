#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

extern NSString * const MetalJITErrorDomain;

typedef NS_ENUM(NSInteger, MetalJITBridgeError) {
    MetalJITBridgeErrorUninitialized      = 101,
    MetalJITBridgeErrorCompilationFailed  = 102,
    MetalJITBridgeErrorInvalidBuffer      = 103,
    MetalJITBridgeErrorInvalidHandle      = 104,
    MetalJITBridgeErrorOverflow           = 105,
    MetalJITBridgeErrorUnderflow          = 106,
    MetalJITBridgeErrorTypeMismatch       = 107
};

/// Manejador opaco de un pipeline compilado.
/// Se destruye automaticamente al llamar a -destroyPipeline.
@interface MetalJITPipelineHandle : NSObject
@property (nonatomic, readonly) int handle;
@end

/// Puente ObjC hacia el motor Metal JIT.
@interface MetalJITBridge : NSObject

// --- Compilacion ---

/// Compila un shader Metal JIT. Devuelve handle (>0) o nil si falla.
+ (nullable MetalJITPipelineHandle *)compilePipelineWithShader:(nullable NSString *)shaderSource
                                                     kernelName:(nullable NSString *)kernelName
                                                          error:(NSError **)error;

/// Destruye un pipeline y libera recursos GPU.
+ (BOOL)destroyPipeline:(MetalJITPipelineHandle *)pipelineHandle
                  error:(NSError **)error;

/// Registra un fallback CPU para un built-in kernel.
/// kernelType: 0=crypto, 1=tensor, 2=logic, 3=physics, 4=topology
+ (BOOL)setBuiltInCPUFallback:(MetalJITPipelineHandle *)pipelineHandle
                   kernelType:(int)kernelType
                        error:(NSError **)error;

/// Crea un wrapper para un handle de pipeline ya existente (usado por Cache).
+ (nullable MetalJITPipelineHandle *)wrapExistingHandle:(int)handle;

// --- Despacho ---

/// Despacho unitario Zero-Copy sobre buffers tipados.
/// dataType: 0=UInt64, 1=Float32, 2=Float64, 3=Int32, 4=Int64
+ (BOOL)dispatchWithPipeline:(MetalJITPipelineHandle *)pipelineHandle
                       input:(const void *)inputBuffer
                      output:(void *)outputBuffer
               elementCount:(int)elementCount
                   dataType:(int)dataType
                       error:(NSError **)error;

/// Despacho por lotes (batch): multiples payloads en una llamada GPU.
/// dataType: 0=UInt64, 1=Float32, 2=Float64, 3=Int32, 4=Int64
+ (BOOL)dispatchBatchWithPipeline:(MetalJITPipelineHandle *)pipelineHandle
                         payloads:(const void *)payloads
                          results:(void *)results
                      numBatches:(int)numBatches
              elementsPerPayload:(int)elementsPerPayload
                         dataType:(int)dataType
                            error:(NSError **)error;

// --- Despacho asincrono ---

+ (void)dispatchAsyncWithPipeline:(MetalJITPipelineHandle *)pipelineHandle
                            input:(const void *)inputBuffer
                           output:(void *)outputBuffer
                    elementCount:(int)elementCount
                        dataType:(int)dataType
               completionHandler:(void (^)(NSError * _Nullable error))handler;

+ (void)dispatchBatchAsyncWithPipeline:(MetalJITPipelineHandle *)pipelineHandle
                              payloads:(const void *)payloads
                               results:(void *)results
                           numBatches:(int)numBatches
                   elementsPerPayload:(int)elementsPerPayload
                              dataType:(int)dataType
                    completionHandler:(void (^)(NSError * _Nullable error))handler;

@end

NS_ASSUME_NONNULL_END
