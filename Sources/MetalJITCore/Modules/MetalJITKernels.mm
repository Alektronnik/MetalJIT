// ==========================================================================
// MetalJITKernels.mm — Implementaciones CPU de kernels predefinidos
// ==========================================================================
// Implementaciones CPU de los kernels predefinidos.
// Cada funcion recibe n elementos de entrada y produce 1 resultado de salida,
// adecuado para batch dispatch donde cada hilo procesa un lote.
// ==========================================================================

#import "MetalJITKernels.h"
#include <cstring>
#include <cmath>

// ---------------------------------------------------------------------------
// Utilidad: sqrt babilonico (sin dependencia de libc)
// ---------------------------------------------------------------------------
double mjit_cpu_sqrt(double n) {
    if (n <= 0.0) return 0.0;
    double x = (n < 1.0) ? 1.0 : n;
    double y = (x + n / x) / 2.0;
    while (x - y > 1e-12) {
        x = y;
        y = (x + n / x) / 2.0;
    }
    return x;
}

// ---------------------------------------------------------------------------
// Kernel 1: Criptografia (MurmurHash3-inspired avalanche)
// ---------------------------------------------------------------------------
void mjit_cpu_crypto(const void* input, void* output, int n, void*) {
    const uint64_t* data = (const uint64_t*)input;
    uint64_t accumulator = 0x811c9dc5;

    for (int i = 0; i < n; ++i) {
        uint64_t face = data[i];
        face = (face << (i * 7 + 3)) | (face >> (64 - (i * 7 + 3)));
        face *= 0xff51afd7ed558ccdULL;
        accumulator ^= face;
        accumulator = (accumulator << 27) | (accumulator >> 37);
        accumulator = accumulator * 5 + 0x52dce729;
    }
    accumulator ^= (uint64_t)n;
    accumulator ^= accumulator >> 33;
    accumulator *= 0xff51afd7ed558ccdULL;
    accumulator ^= accumulator >> 33;
    accumulator *= 0xc4ceb9fe1a85ec53ULL;
    accumulator ^= accumulator >> 33;

    *(uint64_t*)output = accumulator;
}

// ---------------------------------------------------------------------------
// Kernel 2: Tensor IA (Suma ponderada + Leaky ReLU, alpha=0.01)
// ---------------------------------------------------------------------------
void mjit_cpu_tensor(const void* input, void* output, int n, void*) {
    const uint64_t* data = (const uint64_t*)input;
    double sum = 0.0;

    for (int i = 0; i < n; ++i) {
        double value;
        std::memcpy(&value, &data[i], sizeof(double));
        sum += value;
    }

    const double alpha = 0.01;
    double activated = (sum > 0.0) ? sum : (sum * alpha);

    uint64_t packed;
    std::memcpy(&packed, &activated, sizeof(uint64_t));
    *(uint64_t*)output = packed;
}

// ---------------------------------------------------------------------------
// Kernel 3: Logica SAT (AND bitwise sobre todas las caras)
// ---------------------------------------------------------------------------
void mjit_cpu_logic(const void* input, void* output, int n, void*) {
    const uint64_t* data = (const uint64_t*)input;
    uint64_t accumulator = 0xFFFFFFFFFFFFFFFF;

    for (int i = 0; i < n; ++i) {
        accumulator &= data[i];
    }

    *(uint64_t*)output = accumulator;
}

// ---------------------------------------------------------------------------
// Kernel 4: Fisica 3D (Interseccion rayo-esfera)
//           Requiere exactamente 6 doubles: Ox,Oy,Oz,Dx,Dy,Dz
// ---------------------------------------------------------------------------
void mjit_cpu_physics(const void* input, void* output, int n, void*) {
    if (n != 6) {
        *(uint64_t*)output = 0;
        return;
    }

    const uint64_t* data = (const uint64_t*)input;
    double Ox, Oy, Oz, Dx, Dy, Dz;
    std::memcpy(&Ox, &data[0], sizeof(double));
    std::memcpy(&Oy, &data[1], sizeof(double));
    std::memcpy(&Oz, &data[2], sizeof(double));
    std::memcpy(&Dx, &data[3], sizeof(double));
    std::memcpy(&Dy, &data[4], sizeof(double));
    std::memcpy(&Dz, &data[5], sizeof(double));

    // Normalizar direccion
    double mag = mjit_cpu_sqrt(Dx*Dx + Dy*Dy + Dz*Dz);
    if (mag > 0.0) { Dx /= mag; Dy /= mag; Dz /= mag; }

    // Ecuacion cuadratica: at^2 + bt + c = 0
    // Esfera en (0,0,0) con radio = 10.0
    double b = 2.0 * (Ox*Dx + Oy*Dy + Oz*Dz);
    double c = (Ox*Ox + Oy*Oy + Oz*Oz) - 100.0;

    double discriminant = b*b - 4.0*c;
    double hit_distance = -1.0;

    if (discriminant >= 0.0) {
        double root = mjit_cpu_sqrt(discriminant);
        double t1 = (-b - root) / 2.0;
        double t2 = (-b + root) / 2.0;
        if (t1 > 0.0) hit_distance = t1;
        else if (t2 > 0.0) hit_distance = t2;
    }

    uint64_t packed;
    std::memcpy(&packed, &hit_distance, sizeof(uint64_t));
    *(uint64_t*)output = packed;
}

// ---------------------------------------------------------------------------
// Kernel 5: Topologia (Proyector armonico sintetico + Kappa topologico)
//           Requiere exactamente 6 doubles
// ---------------------------------------------------------------------------
void mjit_cpu_topology(const void* input, void* output, int n, void*) {
    if (n != 6) {
        *(uint64_t*)output = 0;
        return;
    }

    const uint64_t* data = (const uint64_t*)input;
    double raw[6];
    for (int i = 0; i < 6; ++i) {
        std::memcpy(&raw[i], &data[i], sizeof(double));
    }

    // Proyector armonico P(6x6)
    const double P[6][6] = {
        { 0.25, -0.25,  0.00,  0.25, -0.25,  0.00},
        {-0.25,  0.50, -0.25, -0.25,  0.50, -0.25},
        { 0.00, -0.25,  0.25,  0.00, -0.25,  0.25},
        { 0.25, -0.25,  0.00,  0.25, -0.25,  0.00},
        {-0.25,  0.50, -0.25, -0.25,  0.50, -0.25},
        { 0.00, -0.25,  0.25,  0.00, -0.25,  0.25}
    };

    double harmonic[6] = {0};
    double norm_sq = 0.0;

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            harmonic[i] += P[i][j] * raw[j];
        }
        norm_sq += harmonic[i] * harmonic[i];
    }

    double norm = mjit_cpu_sqrt(norm_sq);
    double kappa = 0.5 * std::tanh(norm * 3.0) + 0.3;

    uint64_t packed;
    std::memcpy(&packed, &kappa, sizeof(uint64_t));
    *(uint64_t*)output = packed;
}
