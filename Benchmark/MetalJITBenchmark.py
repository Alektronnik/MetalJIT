"""
==========================================================================
MetalJITBenchmark.py — Benchmark standalone Python
==========================================================================
Mide rendimiento real de MetalJIT desde Python sin overhead de XCTest.

Ejecutar:
    METALJIT_LIB=../.build/debug/libMetalJITCore.dylib python MetalJITBenchmark.py
==========================================================================
"""

import sys
import os
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, "..", "Sources", "MetalJITCore", "Utils")
sys.path.insert(0, utils_dir)

from MetalJITBinding import MetalJIT, MetalJITError

# ---------------------------------------------------------------------------
# Utilidad de medicion
# ---------------------------------------------------------------------------
def measure(label, func, iterations=1):
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    elapsed = time.perf_counter() - start
    per_it = elapsed / iterations
    iter_str = f" (x{iterations})" if iterations > 1 else ""
    print(f"  {label:<38} {per_it:8.4f} s{iter_str}")
    return per_it


def divider():
    print("-" * 66)


# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------
COUNT   = 1_000_000
BATCHES = 100_000
PER_B   = 6

try:
    api = MetalJIT()
except RuntimeError as e:
    print(f"[AVISO] {e}")
    print("Compila MetalJIT: cd .. && swift build")
    sys.exit(1)

print(f"""
=================================================================
 MetalJIT Benchmark Python — {COUNT:,} elementos / {BATCHES:,} lotes
=================================================================
""")

# --- CORE ---
print("\n[CORE]")
cpu = api.compile_cpu()
uin = list(range(COUNT))
uout = [0] * COUNT
measure("dispatch UInt64", lambda: api.dispatch(cpu, uin, uout))

fin = [3.14159] * COUNT
fout = [0.0] * COUNT
measure("dispatch Float64", lambda: api.dispatch(cpu, fin, fout, dtype=api.DTYPE_FLOAT64))
divider()

# --- BATCH ---
print("\n[BATCH]")
bp = api.compile_cpu()
pl = [1] * (BATCHES * PER_B)
rl = [0] * BATCHES
measure(f"batch UInt64 ({BATCHES//1000}k x {PER_B})",
        lambda: api.dispatch_batch(bp, pl, rl, BATCHES, PER_B))

fpl = [1.0] * (BATCHES * 4)
frl = [0.0] * BATCHES
measure(f"batch Float64 ({BATCHES//1000}k x 4)",
        lambda: api.dispatch_batch(bp, fpl, frl, BATCHES, 4, dtype=api.DTYPE_FLOAT64))
divider()

# --- COMPILER ---
print("\n[COMPILER]")
shader = """
#include <metal_stdlib>
using namespace metal;
kernel void bk(device const uint64_t* in [[buffer(0)]],
               device uint64_t* out [[buffer(1)]],
               uint id [[thread_position_in_grid]]) { out[id] = in[id] * 2; }
"""
measure("compilar JIT", lambda: api.destroy(api.compile(shader, "bk")), iterations=10)

cpu_only = lambda: api.destroy(api.compile_cpu())
measure("compilar CPU-only", cpu_only, iterations=10)
divider()

# --- DISPATCH ---
print("\n[DISPATCH]")
bh = api.compile_cpu()
bi = [1] * COUNT
bo = [0] * COUNT
measure("dispatch listas Python", lambda: api.dispatch(bh, bi, bo))
api.destroy(bh)
divider()

# --- BINDING ---
print("\n[BINDING]")
import ctypes
bdh = api.compile_cpu()
arr_in  = (ctypes.c_uint64 * COUNT)(*([1] * COUNT))
arr_out = (ctypes.c_uint64 * COUNT)()
ptr_in  = ctypes.cast(arr_in, ctypes.POINTER(ctypes.c_uint64))
ptr_out = ctypes.cast(arr_out, ctypes.POINTER(ctypes.c_uint64))
measure("dispatch raw (zero-copy)", lambda: api.dispatch_raw(bdh, ptr_in, ptr_out, COUNT))
api.destroy(bdh)
divider()

# --- NATIVE ---
print("\n[NATIVE]")
nf = [1.0] * COUNT
measure("Float * 2.0 bucle", lambda: [x * 2.0 for x in nf])

nd = [1.0] * COUNT
measure("Double * 2.0 bucle", lambda: [x * 2.0 for x in nd])

api.destroy_all()

print("\n=================================================================")
print(" MetalJIT Benchmark Python — Completado")
print("=================================================================")
