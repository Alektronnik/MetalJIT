#!/usr/bin/env python3
"""
==========================================================================
MetalJITBindingTest.py — Test de integracion del binding Python
==========================================================================
Ejecutar:
    METALJIT_LIB=../.build/debug/libMetalJITCore.dylib python3 MetalJITBindingTest.py
==========================================================================
"""

import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, "..", "Sources", "MetalJITCore", "Utils")
sys.path.insert(0, utils_dir)

from MetalJITBinding import MetalJIT, MetalJITError

passed = 0
failed = 0

def check(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓ {msg}")
    else:
        failed += 1
        print(f"  ✗ {msg}")

def check_raises(exc_type, func, msg):
    global passed, failed
    try:
        func()
        failed += 1
        print(f"  ✗ {msg} — esperaba excepcion")
    except exc_type:
        passed += 1
        print(f"  ✓ {msg}")
    except Exception as e:
        failed += 1
        print(f"  ✗ {msg} — excepcion incorrecta: {e}")

# ---------------------------------------------------------------------------
print("=" * 60)
print(" MetalJIT Binding Python — Test de integracion")
print("=" * 60)

try:
    api = MetalJIT()
except RuntimeError:
    print("[AVISO] No se encontro libMetalJITCore.dylib.")
    print("Compila con: swift build --product MetalJITCore")
    print("Exporta: METALJIT_LIB=../../.build/debug/libMetalJITCore.dylib")
    sys.exit(1)

# --- Compilacion ---
print("\n[Compilacion]")

h_cpu = api.compile_cpu()
check(h_cpu > 0, "compile_cpu() devuelve handle > 0")

shader = """
#include <metal_stdlib>
using namespace metal;
kernel void test_k(device const uint64_t* in [[buffer(0)]],
                   device uint64_t* out [[buffer(1)]],
                   uint id [[thread_position_in_grid]]) { out[id] = in[id] * 2; }
"""
h_gpu = api.compile(shader, "test_k")
check(h_gpu > 0, "compile(shader, kernel) devuelve handle > 0")

check_raises(MetalJITError, lambda: api.compile("basura", "x"), "Shader invalido lanza MetalJITError")

# --- Dispatch ---
print("\n[Dispatch]")

data_in  = [10, 20, 30, 40, 50]
data_out = [0, 0, 0, 0, 0]
api.dispatch(h_cpu, data_in, data_out)
check(data_out == data_in, "dispatch UInt64 copia entrada → salida")

f_in  = [3.14, 2.71, 1.41]
f_out = [0.0, 0.0, 0.0]
api.dispatch(h_cpu, f_in, f_out, dtype=api.DTYPE_FLOAT64)
check(f_out == f_in, "dispatch Float64")

# --- Batch ---
print("\n[Batch]")

payloads = [1, 2, 3, 10, 20, 30, 100, 200, 300]
results  = [0, 0, 0]
api.dispatch_batch(h_cpu, payloads, results, 3, 3)
check(results == [1, 10, 100], "batch dispatch (3 lotes x 3)")

# --- Zero-Copy ---
print("\n[Zero-Copy]")

import ctypes
arr_in  = (ctypes.c_uint64 * 5)(10, 20, 30, 40, 50)
arr_out = (ctypes.c_uint64 * 5)()
ptr_in  = ctypes.cast(arr_in, ctypes.POINTER(ctypes.c_uint64))
ptr_out = ctypes.cast(arr_out, ctypes.POINTER(ctypes.c_uint64))
api.dispatch_raw(h_cpu, ptr_in, ptr_out, 5)
check(list(arr_out) == [10, 20, 30, 40, 50], "dispatch_raw con punteros crudos")

# --- Multi-pipeline ---
print("\n[Multi-pipeline]")

h1 = api.compile_cpu()
h2 = api.compile_cpu()
h3 = api.compile(shader, "test_k")
check(h1 != h2 and h2 != h3, "Handles unicos")
api.destroy(h1)
check(h1 not in api.list_pipelines(), "destroy() elimina handle de la lista")

# --- Utilidades ---
print("\n[Utilidades]")

check(api.data_type_size(api.DTYPE_UINT64) == 8, "UInt64 = 8 bytes")
check(api.data_type_size(api.DTYPE_FLOAT32) == 4, "Float32 = 4 bytes")
check(api.data_type_name(api.DTYPE_FLOAT64) == "Float64", "nombre Float64")

# --- Limpieza ---
api.destroy_all()

# --- Resumen ---
print(f"\n{'='*60}")
total = passed + failed
print(f" Resultado: {passed}/{total} pasaron")
if failed > 0:
    print(f" {failed} tests FALLIDOS")
    sys.exit(1)
else:
    print(" Todos los tests pasaron correctamente")
print(f"{'='*60}")
