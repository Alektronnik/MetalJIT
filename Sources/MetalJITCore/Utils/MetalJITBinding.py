"""
==========================================================================
MetalJITBinding — Binding Python nativo para MetalJIT
==========================================================================
Carga la libreria dinamica del motor MetalJIT y expone una API Pythonica
para compilar shaders Metal JIT y despacharlos con Zero-Copy.

Requiere:
  - macOS 13+ (Apple Silicon)
  - Libreria dinamica compilada: libMetalJIT.dylib

Uso basico:
  from MetalJITBinding import MetalJIT

  api = MetalJIT()
  handle = api.compile(shader_source, "miKernel")
  api.dispatch(handle, input_array, output_array)
  api.destroy(handle)

Zero-Copy con NumPy:
  import numpy as np
  ptr_in  = array_in.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
  ptr_out = array_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
  api.dispatch_raw(handle, ptr_in, ptr_out, count, api.DTYPE_UINT64)
==========================================================================
"""

import ctypes
import os
import sys
from typing import Optional, Tuple, List, Union

# ======================================================================
# CONSTANTES DE TIPO DE DATO (alineadas con MJITDataType en C)
# ======================================================================
DTYPE_UINT64  = 0
DTYPE_FLOAT32 = 1
DTYPE_FLOAT64 = 2
DTYPE_INT32   = 3
DTYPE_INT64   = 4

DTYPE_NAMES = {
    0: "uint64",
    1: "float32",
    2: "float64",
    3: "int32",
    4: "int64",
}

DTYPE_SIZES = {
    0: 8,   # uint64
    1: 4,   # float32
    2: 8,   # float64
    3: 4,   # int32
    4: 8,   # int64
}

# ======================================================================
# CODIGOS DE ERROR
# ======================================================================
ERR_SUCCESS            = 0
ERR_UNINITIALIZED      = 101
ERR_COMPILATION_FAILED = 102
ERR_INVALID_BUFFER     = 103
ERR_INVALID_HANDLE     = 104
ERR_OVERFLOW           = 105
ERR_UNDERFLOW          = 106
ERR_TYPE_MISMATCH      = 107

ERROR_MESSAGES = {
    101: "Motor no inicializado. Compila un pipeline antes de despachar.",
    102: "Fallo al compilar el shader Metal JIT.",
    103: "Buffer invalido: punteros nulos o tamanio incorrecto.",
    104: "Handle de pipeline invalido. El pipeline fue destruido o nunca se compilo.",
    105: "Desbordamiento: mas elementos de los soportados.",
    106: "Insuficiencia: faltan elementos para completar la operacion.",
    107: "Tipo de dato incompatible con el pipeline.",
}


class MetalJITError(Exception):
    """Excepcion unificada para errores del motor MetalJIT."""

    def __init__(self, code: int, message: str = ""):
        self.code = code
        self.message = message or ERROR_MESSAGES.get(code, f"Error desconocido ({code})")
        super().__init__(f"[MetalJIT] Codigo {code}: {self.message}")


# ======================================================================
# CLASE PRINCIPAL
# ======================================================================

class MetalJIT:
    """
    Cliente Python para el motor MetalJIT.

    Carga la libreria dinamica compilada y expone los metodos:
      - compile()          : compilar shader Metal JIT -> handle
      - destroy()          : liberar pipeline
      - dispatch()         : despacho unitario tipado
      - dispatch_batch()   : despacho por lotes
      - dispatch_raw()     : despacho con punteros crudos (zero-copy NumPy)
      - set_cpu_fallback() : registrar funcion de fallback CPU
    """

    # Constantes de tipo exportadas
    DTYPE_UINT64  = DTYPE_UINT64
    DTYPE_FLOAT32 = DTYPE_FLOAT32
    DTYPE_FLOAT64 = DTYPE_FLOAT64
    DTYPE_INT32   = DTYPE_INT32
    DTYPE_INT64   = DTYPE_INT64

    def __init__(self, lib_path: Optional[str] = None):
        """
        Inicializa el cliente MetalJIT.

        Args:
            lib_path: Ruta a libMetalJIT.dylib. Si es None, busca en:
                      1. Variable de entorno METALJIT_LIB
                      2. Directorio del script
                      3. .build/debug/ (SPM)
                      4. Output/MetalJIT.xcframework/ (build_framework.sh)
        """
        self._lib = self._load_library(lib_path)
        self._define_signatures()
        self._pipelines: List[int] = []  # handles activos

    # ------------------------------------------------------------------
    # Carga de libreria
    # ------------------------------------------------------------------

    def _load_library(self, lib_path: Optional[str]) -> ctypes.CDLL:
        search_paths = []

        if lib_path:
            search_paths.append(lib_path)

        # Variable de entorno
        env_path = os.environ.get("METALJIT_LIB")
        if env_path:
            search_paths.append(env_path)

        # Directorio del script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        search_paths.append(os.path.join(script_dir, "libMetalJIT.dylib"))

        # SPM .build/debug (swift build --product MetalJITCore)
        repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
        search_paths.append(os.path.join(repo_root, ".build", "debug", "libMetalJITCore.dylib"))

        # Framework output
        search_paths.append(os.path.join(repo_root, "Output",
                                          "MetalJIT.xcframework",
                                          "macos-arm64_x86_64",
                                          "MetalJIT.framework", "MetalJIT"))

        for path in search_paths:
            if os.path.exists(path):
                try:
                    return ctypes.CDLL(path)
                except OSError as e:
                    continue

        raise RuntimeError(
            "No se encontro libMetalJIT.dylib. Buscado en:\n  " +
            "\n  ".join(search_paths) +
            "\n\nCompila el framework primero con: swift build -c release"
        )

    def _define_signatures(self):
        """Define las firmas ctypes de la API C."""
        lib = self._lib

        # MJITPipelineHandle mjit_compile_pipeline(const char*, const char*, char*, int)
        lib.mjit_compile_pipeline.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_char_p, ctypes.c_int
        ]
        lib.mjit_compile_pipeline.restype = ctypes.c_int

        # int mjit_destroy_pipeline(MJITPipelineHandle)
        lib.mjit_destroy_pipeline.argtypes = [ctypes.c_int]
        lib.mjit_destroy_pipeline.restype = ctypes.c_int

        # int mjit_dispatch(int, const void*, void*, int, int)
        lib.mjit_dispatch.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int
        ]
        lib.mjit_dispatch.restype = ctypes.c_int

        # int mjit_dispatch_batch(int, const void*, void*, int, int, int)
        lib.mjit_dispatch_batch.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        lib.mjit_dispatch_batch.restype = ctypes.c_int

        # int mjit_data_type_size(int)
        lib.mjit_data_type_size.argtypes = [ctypes.c_int]
        lib.mjit_data_type_size.restype = ctypes.c_int

        # const char* mjit_data_type_name(int)
        lib.mjit_data_type_name.argtypes = [ctypes.c_int]
        lib.mjit_data_type_name.restype = ctypes.c_char_p

    # ------------------------------------------------------------------
    # COMPILACION JIT
    # ------------------------------------------------------------------

    def compile(self, shader_source: str, kernel_name: str) -> int:
        """
        Compila un shader Metal JIT y devuelve un handle de pipeline.

        Args:
            shader_source: Codigo fuente MSL (Metal Shading Language).
            kernel_name:   Nombre de la funcion kernel en el shader.

        Returns:
            Handle de pipeline (entero > 0).

        Raises:
            MetalJITError: Si la compilacion falla.
        """
        c_shader = shader_source.encode("utf-8")
        c_kernel = kernel_name.encode("utf-8")
        error_buf = ctypes.create_string_buffer(512)

        handle = self._lib.mjit_compile_pipeline(c_shader, c_kernel,
                                                   error_buf, 512)

        if handle < 0:
            msg = error_buf.value.decode("utf-8", errors="replace") if error_buf.value else ""
            raise MetalJITError(handle, msg)

        self._pipelines.append(handle)
        return handle

    def compile_cpu(self) -> int:
        """
        Crea un pipeline CPU-only (sin shader Metal).

        Util para desarrollo, testing, o cuando no hay GPU disponible.
        El comportamiento por defecto copia el primer elemento de cada lote.

        Returns:
            Handle de pipeline (entero > 0).
        """
        handle = self._lib.mjit_compile_pipeline(None, None, None, 0)
        if handle < 0:
            raise MetalJITError(handle, "Fallo al crear pipeline CPU.")
        self._pipelines.append(handle)
        return handle

    def destroy(self, handle: int):
        """
        Destruye un pipeline y libera sus recursos GPU.

        Args:
            handle: Handle devuelto por compile() o compile_cpu().

        Raises:
            MetalJITError: Si el handle es invalido.
        """
        result = self._lib.mjit_destroy_pipeline(handle)
        if result != ERR_SUCCESS:
            raise MetalJITError(result)
        if handle in self._pipelines:
            self._pipelines.remove(handle)

    # ------------------------------------------------------------------
    # DESPACHO TIPADO (con listas/arrays de Python)
    # ------------------------------------------------------------------

    def dispatch(self, handle: int, input_data: list, output_data: list,
                 dtype: int = DTYPE_UINT64):
        """
        Despacho unitario sobre listas de Python.

        Convierte las listas a arrays ctypes, despacha, y escribe
        los resultados de vuelta en output_data.

        Args:
            handle:      Handle de pipeline.
            input_data:  Lista de entrada.
            output_data: Lista de salida (modificada in-place).
            dtype:       Tipo de dato (DTYPE_UINT64, DTYPE_FLOAT32, etc.).
        """
        count = len(input_data)

        if dtype == DTYPE_UINT64:
            ctype = ctypes.c_uint64
        elif dtype == DTYPE_FLOAT32:
            ctype = ctypes.c_float
        elif dtype == DTYPE_FLOAT64:
            ctype = ctypes.c_double
        elif dtype == DTYPE_INT32:
            ctype = ctypes.c_int32
        elif dtype == DTYPE_INT64:
            ctype = ctypes.c_int64
        else:
            raise ValueError(f"Tipo de dato no soportado: {dtype}")

        arr_in = (ctype * count)(*input_data)
        arr_out = (ctype * count)()

        result = self._lib.mjit_dispatch(handle, arr_in, arr_out, count, dtype)
        if result != ERR_SUCCESS:
            raise MetalJITError(result)

        # Copiar resultados de vuelta a la lista de salida
        for i in range(count):
            output_data[i] = arr_out[i]

    def dispatch_batch(self, handle: int,
                       payloads: list, results: list,
                       num_batches: int, elements_per_batch: int,
                       dtype: int = DTYPE_UINT64):
        """
        Despacho por lotes sobre listas de Python.

        Args:
            handle:             Handle de pipeline.
            payloads:           Lista plana con num_batches * elements_per_batch elementos.
            results:            Lista de salida con num_batches elementos (modificada in-place).
            num_batches:        Cantidad de lotes.
            elements_per_batch: Elementos por lote.
            dtype:              Tipo de dato.
        """
        if dtype == DTYPE_UINT64:
            ctype = ctypes.c_uint64
        elif dtype == DTYPE_FLOAT32:
            ctype = ctypes.c_float
        elif dtype == DTYPE_FLOAT64:
            ctype = ctypes.c_double
        elif dtype == DTYPE_INT32:
            ctype = ctypes.c_int32
        elif dtype == DTYPE_INT64:
            ctype = ctypes.c_int64
        else:
            raise ValueError(f"Tipo de dato no soportado: {dtype}")

        total_elements = num_batches * elements_per_batch
        arr_payloads = (ctype * total_elements)(*payloads)
        arr_results = (ctype * num_batches)()

        result = self._lib.mjit_dispatch_batch(
            handle, arr_payloads, arr_results,
            num_batches, elements_per_batch, dtype
        )
        if result != ERR_SUCCESS:
            raise MetalJITError(result)

        for i in range(num_batches):
            results[i] = arr_results[i]

    # ------------------------------------------------------------------
    # DESPACHO RAW (Zero-Copy con NumPy)
    # ------------------------------------------------------------------

    def dispatch_raw(self, handle: int,
                     ptr_in, ptr_out,
                     count: int, dtype: int = DTYPE_UINT64):
        """
        Despacho Zero-Copy usando punteros crudos.

        Diseniado para usar con NumPy sin copias:

            import numpy as np
            arr_in  = np.array([1, 2, 3], dtype=np.uint64)
            arr_out = np.zeros(3, dtype=np.uint64)
            ptr_in  = arr_in.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
            ptr_out = arr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
            api.dispatch_raw(handle, ptr_in, ptr_out, 3)

        Args:
            handle: Handle de pipeline.
            ptr_in:  Puntero ctypes al buffer de entrada.
            ptr_out: Puntero ctypes al buffer de salida.
            count:   Cantidad de elementos.
            dtype:   Tipo de dato.
        """
        result = self._lib.mjit_dispatch(handle, ptr_in, ptr_out, count, dtype)
        if result != ERR_SUCCESS:
            raise MetalJITError(result)

    def dispatch_batch_raw(self, handle: int,
                           ptr_payloads, ptr_results,
                           num_batches: int, elements_per_batch: int,
                           dtype: int = DTYPE_UINT64):
        """
        Batch dispatch Zero-Copy con punteros crudos (NumPy).

        Args:
            handle:             Handle de pipeline.
            ptr_payloads:       Puntero al buffer plano de payloads.
            ptr_results:        Puntero al buffer de resultados.
            num_batches:        Cantidad de lotes.
            elements_per_batch: Elementos por lote.
            dtype:              Tipo de dato.
        """
        result = self._lib.mjit_dispatch_batch(
            handle, ptr_payloads, ptr_results,
            num_batches, elements_per_batch, dtype
        )
        if result != ERR_SUCCESS:
            raise MetalJITError(result)

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------

    def data_type_size(self, dtype: int) -> int:
        """Tamano en bytes del tipo de dato."""
        return self._lib.mjit_data_type_size(dtype)

    def data_type_name(self, dtype: int) -> str:
        """Nombre legible del tipo de dato."""
        result = self._lib.mjit_data_type_name(dtype)
        return result.decode("utf-8") if result else "unknown"

    def list_pipelines(self) -> List[int]:
        """Devuelve los handles de pipelines activos."""
        return list(self._pipelines)

    def destroy_all(self):
        """Destruye todos los pipelines activos."""
        for handle in list(self._pipelines):
            try:
                self.destroy(handle)
            except MetalJITError:
                pass

    # ------------------------------------------------------------------
    # CIERRE CONTROLADO
    # ------------------------------------------------------------------

    def close(self):
        """Libera todos los recursos. Equivalente a destroy_all()."""
        self.destroy_all()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ======================================================================
# PRUEBA RAPIDA
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print(" MetalJIT Binding Python — Prueba de concepto")
    print("=" * 60)

    try:
        api = MetalJIT()
        print("[OK] Libreria cargada correctamente.")

        # Test compilacion CPU
        handle = api.compile_cpu()
        print(f"[OK] Pipeline CPU creado: handle={handle}")

        # Test dispatch uint64
        inp = [1, 2, 3, 4, 5]
        out = [0, 0, 0, 0, 0]
        api.dispatch(handle, inp, out)
        print(f"[OK] Dispatch UInt64: {out}")
        assert out == [1, 2, 3, 4, 5], f"Esperaba [1,2,3,4,5], obtuve {out}"

        # Test batch
        payloads = [1, 2, 3, 10, 20, 30, 100, 200, 300]
        results  = [0, 0, 0]
        api.dispatch_batch(handle, payloads, results, num_batches=3, elements_per_batch=3)
        print(f"[OK] Batch dispatch: {results}")
        assert results == [1, 10, 100], f"Esperaba [1,10,100], obtuve {results}"

        # Test float64
        f_inp  = [3.14, 2.71, 1.41]
        f_out  = [0.0, 0.0, 0.0]
        api.dispatch(handle, f_inp, f_out, dtype=api.DTYPE_FLOAT64)
        print(f"[OK] Dispatch Float64: {f_out}")
        assert f_out == [3.14, 2.71, 1.41]

        # Test compilacion shader
        shader = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void doble(device const uint64_t* in  [[buffer(0)]],
                          device uint64_t* out        [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
            out[id] = in[id] * 2;
        }
        """
        h2 = api.compile(shader, "doble")
        print(f"[OK] Shader Metal JIT compilado: handle={h2}")

        # Limpiar
        api.destroy_all()
        print("[OK] Recursos liberados.")
        print("\nPrueba completada exitosamente.")

    except MetalJITError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[AVISO] {e}")
        print("Esto es normal si no has compilado el framework.")
        print("Ejecuta: swift build -c release")
    except Exception as e:
        print(f"[ERROR] Excepcion inesperada: {e}")
        sys.exit(1)
