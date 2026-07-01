"""
==========================================================================
MetalJITExample.py — Ejemplo completo de la API Python de MetalJIT
==========================================================================
Demuestra: compilacion JIT, despacho tipado, batch, multi-pipeline,
           zero-copy con NumPy, y manejo de errores.

Ejecutar:
    METALJIT_LIB=../../.build/debug/libMetalJITCore.dylib python3 MetalJITExample.py

La libreria se genera automaticamente con:
    swift build --product MetalJITCore
==========================================================================
"""

import sys
import os

# Aseguramos que Utils este en el path
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, "..", "..", "Sources", "MetalJITCore", "Utils")
sys.path.insert(0, utils_dir)

from MetalJITBinding import MetalJIT, MetalJITError


def ejemplo_1_cpu_only(api: MetalJIT):
    """Pipeline CPU-only: despacho basico sin GPU."""
    print("[1] Pipeline CPU-only")

    handle = api.compile_cpu()
    entrada = [10, 20, 30, 40, 50]
    salida  = [0, 0, 0, 0, 0]

    api.dispatch(handle, entrada, salida)
    print(f"    Entrada: {entrada}")
    print(f"    Salida : {salida}")
    assert salida == entrada, "CPU fallback debe copiar"
    print("    ✓ OK\n")
    return handle


def ejemplo_2_compilar_shader(api: MetalJIT):
    """Compilar un shader Metal JIT desde Python."""
    print("[2] Compilacion Metal JIT")

    shader = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void doble(
        device const uint64_t* in  [[buffer(0)]],
        device uint64_t* out       [[buffer(1)]],
        uint id [[thread_position_in_grid]])
    {
        out[id] = in[id] * 2;
    }
    """

    handle = api.compile(shader, "doble")
    print(f"    Shader 'doble' compilado: handle={handle}")
    print("    ✓ OK\n")
    return handle


def ejemplo_3_float64(api: MetalJIT):
    """Despacho con Float64 (double precision)."""
    print("[3] Despacho Float64")

    handle = api.compile_cpu()
    entrada = [3.1415926535, 2.7182818284, 1.4142135623]
    salida  = [0.0, 0.0, 0.0]

    api.dispatch(handle, entrada, salida, dtype=api.DTYPE_FLOAT64)
    print(f"    Entrada: {entrada}")
    print(f"    Salida : {salida}")
    assert salida == entrada, "Float64 fallback debe copiar"
    print("    ✓ OK\n")
    return handle


def ejemplo_4_batch(api: MetalJIT):
    """Despacho por lotes: 3 lotes de 4 elementos cada uno."""
    print("[4] Batch dispatch (3 lotes de 4 elementos)")

    handle = api.compile_cpu()
    payloads = [
        1,  2,  3,  4,
        10, 20, 30, 40,
        100, 200, 300, 400
    ]
    resultados = [0, 0, 0]

    api.dispatch_batch(handle, payloads, resultados,
                       num_batches=3, elements_per_batch=4)
    print(f"    Payloads  : {payloads}")
    print(f"    Resultados: {resultados}")
    assert resultados == [1, 10, 100], "Batch copia primer elemento"
    print("    ✓ OK\n")
    return handle


def ejemplo_5_multi_pipeline(api: MetalJIT):
    """Multiples pipelines simultaneos."""
    print("[5] Multi-pipeline simultaneo")

    doble = api.compile("""
    #include <metal_stdlib>
    using namespace metal;
    kernel void doble(
        device const uint64_t* in  [[buffer(0)]],
        device uint64_t* out       [[buffer(1)]],
        uint id [[thread_position_in_grid]])
    { out[id] = in[id] * 2; }
    """, "doble")

    triple = api.compile("""
    #include <metal_stdlib>
    using namespace metal;
    kernel void triple(
        device const uint64_t* in  [[buffer(0)]],
        device uint64_t* out       [[buffer(1)]],
        uint id [[thread_position_in_grid]])
    { out[id] = in[id] * 3; }
    """, "triple")

    print(f"    Pipeline doble:  handle={doble}")
    print(f"    Pipeline triple: handle={triple}")
    assert doble != triple, "Handles deben ser distintos"

    api.destroy(doble)
    print("    Pipeline doble destruido. Triple sigue activo.")
    print("    ✓ OK\n")
    return triple


def ejemplo_6_compilacion_fallida(api: MetalJIT):
    """Captura de errores de compilacion MSL."""
    print("[6] Compilacion fallida con mensaje de error")

    shader_roto = "kernel void roto() { ESTO_NO_COMPILA }"
    try:
        api.compile(shader_roto, "roto")
        assert False, "No deberia llegar aqui"
    except MetalJITError as e:
        print(f"    Error capturado: {e}")
        print("    ✓ OK (esperado)\n")


def ejemplo_7_zero_copy_numpy(api: MetalJIT):
    """Zero-Copy con NumPy: punteros crudos sin copia intermedia."""
    print("[7] Zero-Copy con NumPy")

    try:
        import numpy as np
        import ctypes
    except ImportError:
        print("    ⚠ NumPy no disponible, saltando.\n")
        return None

    handle = api.compile_cpu()
    count = 1_000_000

    arr_in  = np.arange(count, dtype=np.uint64)
    arr_out = np.zeros(count, dtype=np.uint64)

    ptr_in  = arr_in.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
    ptr_out = arr_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))

    api.dispatch_raw(handle, ptr_in, ptr_out, count, api.DTYPE_UINT64)

    print(f"    Elementos procesados: {count:,}")
    print(f"    Primeros 5: {arr_out[:5].tolist()}")
    assert arr_out[0] == 0 and arr_out[1] == 1, "Zero-copy incorrecto"
    print("    ✓ OK\n")
    return handle


# ======================================================================
# ORQUESTADOR
# ======================================================================
def main():
    print("=" * 60)
    print(" MetalJIT — Ejemplo Python")
    print("=" * 60)
    print()

    try:
        api = MetalJIT()
    except RuntimeError as e:
        print(f"[AVISO] {e}")
        print()
        print("Para compilar MetalJIT:")
        print("  cd MetalJIT && swift build")
        print()
        print("Luego exporta la ruta:")
        print("  export METALJIT_LIB=.build/debug/libMetalJITCore.dylib")
        sys.exit(1)

    handles = []

    try:
        handles.append(ejemplo_1_cpu_only(api))
        handles.append(ejemplo_2_compilar_shader(api))
        handles.append(ejemplo_3_float64(api))
        handles.append(ejemplo_4_batch(api))
        handles.append(ejemplo_5_multi_pipeline(api))
        ejemplo_6_compilacion_fallida(api)
        handles.append(ejemplo_7_zero_copy_numpy(api))

    except MetalJITError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Excepcion inesperada: {e}")
        sys.exit(1)
    finally:
        # Limpiar pipelines activos
        for h in handles:
            try:
                api.destroy(h) if h else None
            except MetalJITError:
                pass

    print("=" * 60)
    print(" MetalJIT Python — Todos los ejemplos completados")
    print("=" * 60)


if __name__ == "__main__":
    main()
