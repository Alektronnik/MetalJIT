// ==========================================================================
// MetalJITKernels.h — Implementaciones CPU de kernels predefinidos
// ==========================================================================
// Funciones de fallback CPU para los kernels predefinidos.
// Firman como MJITCPUComputeFunc: (const void* in, void* out, int n, void* ctx)
//
// En modo batch: n = elementos por lote, out = 1 resultado por lote.
// En modo unitario: n = elementos, out = n elementos (copia o reduce).
// ==========================================================================

#ifndef MetalJITKernels_h
#define MetalJITKernels_h

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Kernel 1: Criptografia — MurmurHash3 avalancha
// Entrada: n uint64_t, Salida: 1 uint64_t (hash final)
void mjit_cpu_crypto(const void* input, void* output, int n, void* ctx);

// Kernel 2: Tensor IA — Suma ponderada + LeakyReLU
// Entrada: n uint64_t (doubles empaquetados), Salida: 1 uint64_t
void mjit_cpu_tensor(const void* input, void* output, int n, void* ctx);

// Kernel 3: Logica SAT — AND bitwise sobre n caras
// Entrada: n uint64_t, Salida: 1 uint64_t (interseccion)
void mjit_cpu_logic(const void* input, void* output, int n, void* ctx);

// Kernel 4: Fisica 3D — Interseccion rayo-esfera (n DEBE ser 6)
// Entrada: 6 uint64_t (doubles: Ox,Oy,Oz,Dx,Dy,Dz), Salida: 1 uint64_t
void mjit_cpu_physics(const void* input, void* output, int n, void* ctx);

// Kernel 5: Topologia — Proyector armonico + Kappa (n DEBE ser 6)
// Entrada: 6 uint64_t (doubles), Salida: 1 uint64_t (kappa)
void mjit_cpu_topology(const void* input, void* output, int n, void* ctx);

// Utilidad: sqrt puro sin libc (metodo babilonico)
double mjit_cpu_sqrt(double n);

#ifdef __cplusplus
}
#endif

#endif /* MetalJITKernels_h */
