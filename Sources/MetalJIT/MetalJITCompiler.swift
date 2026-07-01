import Foundation
import MetalJITCore

// MARK: - Errores de compilacion JIT

/// Errores que puede producir la compilacion JIT de shaders Metal.
public enum JITCompilerError: Error, Equatable {
    /// El shader MSL contiene errores de sintaxis o no pudo compilarse.
    case compilationFailed(String)
    /// El pipeline fue destruido o nunca fue valido.
    case invalidHandle
    /// Error desconocido con codigo numerico.
    case unknown(Int)
}

// MARK: - Pipeline compilado

/// Representa un pipeline Metal compilado listo para despachar.
///
/// Un `Pipeline` se obtiene llamando a ``JITCompiler/compile(shaderSource:kernelName:)``
/// o ``JITCompiler/compile(builtIn:)``. Contiene el estado de GPU necesario para
/// ejecutar el shader.
///
/// Se destruye automaticamente al llamar a ``destroy()`` o al ser desasignado.
/// Es `Sendable` y puede compartirse entre hilos.
public final class Pipeline: @unchecked Sendable {
    let handle: MetalJITPipelineHandle

    init(handle: MetalJITPipelineHandle) {
        self.handle = handle
    }

    deinit {
        try? MetalJITBridge.destroyPipeline(handle)
    }

    /// Destruye explicitamente el pipeline liberando recursos GPU.
    public func destroy() throws {
        do {
            try MetalJITBridge.destroyPipeline(handle)
        } catch let error as NSError {
            if error.code == MetalJITBridgeError.invalidHandle.rawValue {
                throw JITCompilerError.invalidHandle
            }
            throw JITCompilerError.unknown(error.code)
        }
    }
}

// MARK: - Compilador JIT de shaders Metal

/// Compila codigo fuente MSL (Metal Shading Language) en tiempo de ejecucion
/// y devuelve un `Pipeline` listo para ser despachado por `ComputeDispatcher`.
public final class JITCompiler: Sendable {

    public init() {}

    /// Compila un shader Metal JIT.
    ///
    /// - Parameters:
    ///   - shaderSource: Codigo fuente MSL. Si es nil, pipeline CPU-only.
    ///   - kernelName: Nombre de la funcion kernel dentro del shader.
    /// - Returns: `Pipeline` compilado.
    public func compile(
        shaderSource: String? = nil,
        kernelName: String? = nil
    ) throws -> Pipeline {
        do {
            let handle = try MetalJITBridge.compilePipeline(
                withShader: shaderSource,
                kernelName: kernelName
            )
            return Pipeline(handle: handle)
        } catch let error as NSError {
            let desc = error.localizedDescription
            if error.code == MetalJITBridgeError.compilationFailed.rawValue {
                throw JITCompilerError.compilationFailed(desc)
            }
            throw JITCompilerError.unknown(error.code)
        }
    }

    /// Compila un kernel predefinido (built-in).
    ///
    /// Incluye automaticamente el shader MSL y registra el fallback CPU.
    /// Solo requiere el tipo de kernel; los demas parametros se derivan.
    ///
    /// - Parameter kernel: Tipo de kernel predefinido (.crypto, .tensor, etc.)
    /// - Returns: `Pipeline` compilado con fallback CPU registrado.
    public func compile(builtIn kernel: BuiltInKernel) throws -> Pipeline {
        let pipeline = try compile(
            shaderSource: kernel.mslSource,
            kernelName: kernel.kernelName
        )

        // Registrar fallback CPU para este kernel
        let kernelType: Int32 = {
            switch kernel {
            case .crypto:    return 0
            case .tensor:    return 1
            case .logic:     return 2
            case .physics:   return 3
            case .topology:  return 4
            }
        }()

        do {
            try MetalJITBridge.setBuiltInCPUFallback(pipeline.handle,
                                                      kernelType: kernelType)
        } catch {
            // No es fatal: el kernel GPU sigue funcionando
            print("[MetalJIT] Aviso: no se pudo registrar CPU fallback para \(kernel.kernelName)")
        }

        return pipeline
    }
}
