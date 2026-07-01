import Foundation
import MetalJITCore

// MARK: - Errores de despacho

/// Errores que puede producir el despacho de pipelines.
public enum ComputeDispatcherError: Error, Equatable {
    /// El pipeline no existe o fue destruido.
    case invalidHandle
    /// El buffer de entrada o salida es nulo o tiene tamano incorrecto.
    case invalidBuffer
    /// El contador de elementos excede Int32.max.
    case overflow
    /// Datos insuficientes para el pipeline.
    case underflow
    /// El tipo de dato no coincide con el esperado por el pipeline.
    case typeMismatch
    /// Error desconocido con codigo numerico.
    case unknown(Int)
}

// MARK: - Tipos de dato soportados

/// Mapea los tipos Swift a los valores crudos del enum C MJITDataType:
///   0 = UInt64, 1 = Float32, 2 = Float64, 3 = Int32, 4 = Int64
private enum DataTypeTag: Int32 {
    case uint64  = 0
    case float32 = 1
    case float64 = 2
    case int32   = 3
    case int64   = 4
    case float16 = 5
}

// MARK: - Despachador de computo Zero-Copy

/// Despacha kernels de computo compilados (via `JITCompiler.compile()`)
/// sobre buffers de memoria, usando Zero-Copy sobre la arquitectura UMA
/// de Apple Silicon cuando la GPU esta disponible.
///
/// Auto-enrutamiento:
/// - N <  10,000 elementos -> CPU (Grand Central Dispatch + fallback)
/// - N >= 10,000 elementos -> GPU (Metal)
public final class ComputeDispatcher: Sendable {

    public init() {}

    // MARK: Despacho UInt64

    /// Despacha un kernel sobre buffers UInt64.
    public func dispatch(
        pipeline: Pipeline,
        input: UnsafeMutableBufferPointer<UInt64>,
        output: UnsafeMutableBufferPointer<UInt64>
    ) throws {
        guard input.count == output.count, input.count > 0 else {
            throw ComputeDispatcherError.invalidBuffer
        }

        try execute(pipeline: pipeline,
                    inputPtr: input.baseAddress!,
                    outputPtr: output.baseAddress!,
                    count: input.count,
                    type: .uint64)
    }

    // MARK: Despacho Float32

    /// Despacha un kernel sobre buffers Float32.
    public func dispatchFloat32(
        pipeline: Pipeline,
        input: UnsafeMutableBufferPointer<Float32>,
        output: UnsafeMutableBufferPointer<Float32>
    ) throws {
        guard input.count == output.count, input.count > 0 else {
            throw ComputeDispatcherError.invalidBuffer
        }

        try execute(pipeline: pipeline,
                    inputPtr: input.baseAddress!,
                    outputPtr: output.baseAddress!,
                    count: input.count,
                    type: .float32)
    }

    // MARK: Despacho Float64

    /// Despacha un kernel sobre buffers Float64 (Double).
    public func dispatchFloat64(
        pipeline: Pipeline,
        input: UnsafeMutableBufferPointer<Float64>,
        output: UnsafeMutableBufferPointer<Float64>
    ) throws {
        guard input.count == output.count, input.count > 0 else {
            throw ComputeDispatcherError.invalidBuffer
        }

        try execute(pipeline: pipeline,
                    inputPtr: input.baseAddress!,
                    outputPtr: output.baseAddress!,
                    count: input.count,
                    type: .float64)
    }

    // MARK: Despacho Int32

    /// Despacha un kernel sobre buffers Int32.
    public func dispatchInt32(
        pipeline: Pipeline,
        input: UnsafeMutableBufferPointer<Int32>,
        output: UnsafeMutableBufferPointer<Int32>
    ) throws {
        guard input.count == output.count, input.count > 0 else {
            throw ComputeDispatcherError.invalidBuffer
        }

        try execute(pipeline: pipeline,
                    inputPtr: input.baseAddress!,
                    outputPtr: output.baseAddress!,
                    count: input.count,
                    type: .int32)
    }

    // MARK: Despacho Float16

    /// Despacha un kernel sobre buffers Float16 (half precision).
    /// Útil para ML/inferencia donde la precision reducida es aceptable.
    public func dispatchFloat16(
        pipeline: Pipeline,
        input: UnsafeMutableBufferPointer<Float16>,
        output: UnsafeMutableBufferPointer<Float16>
    ) throws {
        guard input.count == output.count, input.count > 0 else {
            throw ComputeDispatcherError.invalidBuffer
        }

        try execute(pipeline: pipeline,
                    inputPtr: input.baseAddress!,
                    outputPtr: output.baseAddress!,
                    count: input.count,
                    type: .float16)
    }

    // MARK: Batch dispatch UInt64

    /// Despacho por lotes: procesa `numBatches` payloads en paralelo.
    ///
    /// - Parameters:
    ///   - pipeline: Pipeline compilado.
    ///   - payloads: Buffer plano con (numBatches * elementsPerBatch) elementos.
    ///   - results: Buffer de salida con numBatches elementos.
    ///   - numBatches: Cantidad de lotes.
    ///   - elementsPerBatch: Elementos por lote (ej. 6 para raytracing).
    public func dispatchBatch(
        pipeline: Pipeline,
        payloads: UnsafeMutableBufferPointer<UInt64>,
        results: UnsafeMutableBufferPointer<UInt64>,
        numBatches: Int,
        elementsPerBatch: Int
    ) throws {
        guard payloads.count == numBatches * elementsPerBatch,
              results.count == numBatches,
              numBatches > 0,
              elementsPerBatch > 0 else {
            throw ComputeDispatcherError.invalidBuffer
        }

        try executeBatch(pipeline: pipeline,
                         payloadsPtr: payloads.baseAddress!,
                         resultsPtr: results.baseAddress!,
                         numBatches: numBatches,
                         elementsPerBatch: elementsPerBatch,
                         type: .uint64)
    }

    // MARK: Batch dispatch Float32

    /// Batch dispatch para Float32.
    public func dispatchBatchFloat32(
        pipeline: Pipeline,
        payloads: UnsafeMutableBufferPointer<Float32>,
        results: UnsafeMutableBufferPointer<Float32>,
        numBatches: Int,
        elementsPerBatch: Int
    ) throws {
        guard payloads.count == numBatches * elementsPerBatch,
              results.count == numBatches,
              numBatches > 0,
              elementsPerBatch > 0 else {
            throw ComputeDispatcherError.invalidBuffer
        }

        try executeBatch(pipeline: pipeline,
                         payloadsPtr: payloads.baseAddress!,
                         resultsPtr: results.baseAddress!,
                         numBatches: numBatches,
                         elementsPerBatch: elementsPerBatch,
                         type: .float32)
    }

    // MARK: Batch dispatch Float64

    /// Batch dispatch para Float64.
    public func dispatchBatchFloat64(
        pipeline: Pipeline,
        payloads: UnsafeMutableBufferPointer<Float64>,
        results: UnsafeMutableBufferPointer<Float64>,
        numBatches: Int,
        elementsPerBatch: Int
    ) throws {
        guard payloads.count == numBatches * elementsPerBatch,
              results.count == numBatches,
              numBatches > 0,
              elementsPerBatch > 0 else {
            throw ComputeDispatcherError.invalidBuffer
        }

        try executeBatch(pipeline: pipeline,
                         payloadsPtr: payloads.baseAddress!,
                         resultsPtr: results.baseAddress!,
                         numBatches: numBatches,
                         elementsPerBatch: elementsPerBatch,
                         type: .float64)
    }

    // MARK: - Despacho asincrono (async/await)

    /// Despacho asincrono UInt64. Retorna inmediatamente; la GPU procesa en background.
    @available(macOS 10.15, *)
    public func dispatchAsync(
        pipeline: Pipeline,
        input: UnsafeMutableBufferPointer<UInt64>,
        output: UnsafeMutableBufferPointer<UInt64>
    ) async throws {
        guard input.count == output.count, input.count > 0 else {
            throw ComputeDispatcherError.invalidBuffer
        }
        try await executeAsync(pipeline: pipeline,
                                inputPtr: input.baseAddress!,
                                outputPtr: output.baseAddress!,
                                count: input.count,
                                type: .uint64)
    }

    /// Despacho asincrono Float64.
    @available(macOS 10.15, *)
    public func dispatchFloat64Async(
        pipeline: Pipeline,
        input: UnsafeMutableBufferPointer<Float64>,
        output: UnsafeMutableBufferPointer<Float64>
    ) async throws {
        guard input.count == output.count, input.count > 0 else {
            throw ComputeDispatcherError.invalidBuffer
        }
        try await executeAsync(pipeline: pipeline,
                                inputPtr: input.baseAddress!,
                                outputPtr: output.baseAddress!,
                                count: input.count,
                                type: .float64)
    }

    /// Batch async UInt64.
    @available(macOS 10.15, *)
    public func dispatchBatchAsync(
        pipeline: Pipeline,
        payloads: UnsafeMutableBufferPointer<UInt64>,
        results: UnsafeMutableBufferPointer<UInt64>,
        numBatches: Int,
        elementsPerBatch: Int
    ) async throws {
        guard payloads.count == numBatches * elementsPerBatch,
              results.count == numBatches,
              numBatches > 0, elementsPerBatch > 0 else {
            throw ComputeDispatcherError.invalidBuffer
        }
        try await executeBatchAsync(pipeline: pipeline,
                                     payloadsPtr: payloads.baseAddress!,
                                     resultsPtr: results.baseAddress!,
                                     numBatches: numBatches,
                                     elementsPerBatch: elementsPerBatch,
                                     type: .uint64)
    }

    // MARK: - Metodos privados async

    private func execute(
        pipeline: Pipeline,
        inputPtr: UnsafeRawPointer,
        outputPtr: UnsafeMutableRawPointer,
        count: Int,
        type: DataTypeTag
    ) throws {
        do {
            guard let ec = Int32(exactly: count) else {
                throw ComputeDispatcherError.overflow
            }
            try MetalJITBridge.dispatch(
                withPipeline: pipeline.handle,
                input: inputPtr,
                output: outputPtr,
                elementCount: ec,
                dataType: type.rawValue
            )
        } catch let error as NSError {
            throw mapError(error)
        }
    }

    private func executeBatch(
        pipeline: Pipeline,
        payloadsPtr: UnsafeRawPointer,
        resultsPtr: UnsafeMutableRawPointer,
        numBatches: Int,
        elementsPerBatch: Int,
        type: DataTypeTag
    ) throws {
        do {
            guard let nb = Int32(exactly: numBatches),
                  let ep = Int32(exactly: elementsPerBatch) else {
                throw ComputeDispatcherError.overflow
            }
            try MetalJITBridge.dispatchBatch(
                withPipeline: pipeline.handle,
                payloads: payloadsPtr,
                results: resultsPtr,
                numBatches: nb,
                elementsPerPayload: ep,
                dataType: type.rawValue
            )
        } catch let error as NSError {
            throw mapError(error)
        }
    }

    private func mapError(_ error: NSError) -> ComputeDispatcherError {
        switch error.code {
        case MetalJITBridgeError.invalidHandle.rawValue:
            return .invalidHandle
        case MetalJITBridgeError.invalidBuffer.rawValue:
            return .invalidBuffer
        case MetalJITBridgeError.overflow.rawValue:
            return .overflow
        case MetalJITBridgeError.underflow.rawValue:
            return .underflow
        case MetalJITBridgeError.typeMismatch.rawValue:
            return .typeMismatch
        default:
            return .unknown(error.code)
        }
    }

    @available(macOS 10.15, *)
    private func executeAsync(
        pipeline: Pipeline,
        inputPtr: UnsafeRawPointer,
        outputPtr: UnsafeMutableRawPointer,
        count: Int,
        type: DataTypeTag
    ) async throws {
        guard let ec = Int32(exactly: count) else {
            throw ComputeDispatcherError.overflow
        }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            MetalJITBridge.dispatchAsync(
                withPipeline: pipeline.handle,
                input: inputPtr,
                output: outputPtr,
                elementCount: ec,
                dataType: type.rawValue
            ) { error in
                if let nsError = error as NSError? {
                    continuation.resume(throwing: self.mapError(nsError))
                } else {
                    continuation.resume()
                }
            }
        }
    }

    @available(macOS 10.15, *)
    private func executeBatchAsync(
        pipeline: Pipeline,
        payloadsPtr: UnsafeRawPointer,
        resultsPtr: UnsafeMutableRawPointer,
        numBatches: Int,
        elementsPerBatch: Int,
        type: DataTypeTag
    ) async throws {
        guard let nb = Int32(exactly: numBatches),
              let ep = Int32(exactly: elementsPerBatch) else {
            throw ComputeDispatcherError.overflow
        }
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            MetalJITBridge.dispatchBatchAsync(
                withPipeline: pipeline.handle,
                payloads: payloadsPtr,
                results: resultsPtr,
                numBatches: nb,
                elementsPerPayload: ep,
                dataType: type.rawValue
            ) { error in
                if let nsError = error as NSError? {
                    continuation.resume(throwing: self.mapError(nsError))
                } else {
                    continuation.resume()
                }
            }
        }
    }
}
