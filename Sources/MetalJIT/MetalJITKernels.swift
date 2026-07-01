import Foundation

// ==========================================================================
// MetalJITKernels.swift
// ==========================================================================
// Catalogo de kernels predefinidos.
// Cada kernel incluye:
//   - Codigo fuente MSL (Metal Shading Language)
//   - Nombre de la funcion kernel
//   - Numero de caras/elementos esperados por lote
//   - Funcion de fallback CPU registrada automaticamente
// ==========================================================================

// MARK: - Enum de kernels predefinidos

public enum BuiltInKernel: String, CaseIterable {
    /// Hash avalancha MurmurHash3. Entrada: N uint64. Salida: 1 uint64.
    case crypto

    /// Suma ponderada + LeakyReLU (alpha=0.01). Entrada: N doubles. Salida: 1 double.
    case tensor

    /// AND bitwise sobre todas las caras (SAT). Entrada: N uint64. Salida: 1 uint64.
    case logic

    /// Interseccion rayo-esfera 3D. Entrada: 6 doubles (Ox,Oy,Oz,Dx,Dy,Dz). Salida: 1 double.
    case physics

    /// Proyector armonico + Kappa topologico. Entrada: 6 doubles. Salida: 1 double.
    case topology

    // MARK: - Metadatos

    /// Nombre de la funcion kernel en el shader MSL.
    public var kernelName: String {
        switch self {
        case .crypto:    return "mjit_crypto"
        case .tensor:    return "mjit_tensor"
        case .logic:     return "mjit_logic"
        case .physics:   return "mjit_physics"
        case .topology:  return "mjit_topology"
        }
    }

    /// Numero de elementos por lote.
    /// Para unit dispatch, usar 1. Para batch, usar este valor.
    public var defaultFaceCount: Int {
        switch self {
        case .crypto:    return 6
        case .tensor:    return 9
        case .logic:     return 3
        case .physics:   return 6
        case .topology:  return 6
        }
    }

    /// Tipo de dato nativo del kernel.
    public var nativeDataType: DataType {
        switch self {
        case .crypto, .logic:
            return .uint64
        case .tensor, .physics, .topology:
            return .float64
        }
    }

    /// Descripcion legible.
    public var description: String {
        switch self {
        case .crypto:    return "Hash avalancha MurmurHash3 — N caras → 1 hash"
        case .tensor:    return "Suma ponderada + LeakyReLU — N pesos → 1 activacion"
        case .logic:     return "AND bitwise (SAT) — N clausulas → 1 interseccion"
        case .physics:   return "Interseccion rayo-esfera 3D — 6 doubles → distancia"
        case .topology:  return "Proyector armonico + Kappa — 6 campos → indice topologico"
        }
    }

    // MARK: - Codigo fuente MSL

    /// Shader Metal completo para este kernel.
    public var mslSource: String {
        let header = """
        #include <metal_stdlib>
        using namespace metal;

        inline float uint64_to_float(uint64_t v) {
            uint sign = (v >> 63) & 1;
            int exp = ((v >> 52) & 0x7FF) - 1023;
            uint mantissa = (v >> 29) & 0x7FFFFF;
            if (exp == -1023 && mantissa == 0) return 0.0f;
            uint32_t f_bits = (sign << 31) | (clamp(exp + 127, 0, 255) << 23) | mantissa;
            return as_type<float>(f_bits);
        }

        inline uint64_t float_to_uint64(float f) {
            uint32_t f_bits = as_type<uint32_t>(f);
            uint64_t sign = (f_bits >> 31) & 1;
            int exp = ((f_bits >> 23) & 0xFF) - 127;
            uint64_t mantissa = (f_bits & 0x7FFFFF);
            if (exp == -127 && mantissa == 0) return 0;
            uint64_t d_bits = (sign << 63) | ((uint64_t)(exp + 1023) << 52) | (mantissa << 29);
            return d_bits;
        }

        """

        switch self {
        case .crypto:
            return header + """
            kernel void mjit_crypto(
                device const uint64_t* data    [[buffer(0)]],
                device uint64_t* results       [[buffer(1)]],
                constant int& faces            [[buffer(2)]],
                uint id [[thread_position_in_grid]])
            {
                uint offset = id * faces;
                uint64_t acc = 0x811c9dc5;
                for (int i = 0; i < faces; ++i) {
                    uint64_t face = data[offset + i];
                    face = (face << (i * 7 + 3)) | (face >> (64 - (i * 7 + 3)));
                    face *= 0xff51afd7ed558ccdULL;
                    acc ^= face;
                    acc = (acc << 27) | (acc >> 37);
                    acc = acc * 5 + 0x52dce729;
                }
                acc ^= (uint64_t)faces;
                acc ^= acc >> 33;
                acc *= 0xff51afd7ed558ccdULL;
                acc ^= acc >> 33;
                acc *= 0xc4ceb9fe1a85ec53ULL;
                acc ^= acc >> 33;
                results[id] = acc;
            }
            """

        case .tensor:
            return header + """
            kernel void mjit_tensor(
                device const uint64_t* data    [[buffer(0)]],
                device uint64_t* results       [[buffer(1)]],
                constant int& faces            [[buffer(2)]],
                uint id [[thread_position_in_grid]])
            {
                uint offset = id * faces;
                float sum = 0.0f;
                for (int i = 0; i < faces; ++i) {
                    sum += uint64_to_float(data[offset + i]);
                }
                const float alpha = 0.01f;
                float activated = (sum > 0.0f) ? sum : (sum * alpha);
                results[id] = float_to_uint64(activated);
            }
            """

        case .logic:
            return header + """
            kernel void mjit_logic(
                device const uint64_t* data    [[buffer(0)]],
                device uint64_t* results       [[buffer(1)]],
                constant int& faces            [[buffer(2)]],
                uint id [[thread_position_in_grid]])
            {
                uint offset = id * faces;
                uint64_t acc = 0xFFFFFFFFFFFFFFFF;
                for (int i = 0; i < faces; ++i) {
                    acc &= data[offset + i];
                }
                results[id] = acc;
            }
            """

        case .physics:
            return header + """
            kernel void mjit_physics(
                device const uint64_t* data    [[buffer(0)]],
                device uint64_t* results       [[buffer(1)]],
                constant int& faces            [[buffer(2)]],
                uint id [[thread_position_in_grid]])
            {
                if (faces != 6) { results[id] = 0; return; }
                uint offset = id * 6;
                float Ox = uint64_to_float(data[offset+0]);
                float Oy = uint64_to_float(data[offset+1]);
                float Oz = uint64_to_float(data[offset+2]);
                float Dx = uint64_to_float(data[offset+3]);
                float Dy = uint64_to_float(data[offset+4]);
                float Dz = uint64_to_float(data[offset+5]);
                float mag = sqrt(Dx*Dx + Dy*Dy + Dz*Dz);
                if (mag > 0.0f) { Dx/=mag; Dy/=mag; Dz/=mag; }
                float b = 2.0f * (Ox*Dx + Oy*Dy + Oz*Dz);
                float c = (Ox*Ox + Oy*Oy + Oz*Oz) - 100.0f;
                float disc = b*b - 4.0f*c;
                float hit = -1.0f;
                if (disc >= 0.0f) {
                    float root = sqrt(disc);
                    float t1 = (-b - root) / 2.0f;
                    float t2 = (-b + root) / 2.0f;
                    if (t1 > 0.0f) hit = t1;
                    else if (t2 > 0.0f) hit = t2;
                }
                results[id] = float_to_uint64(hit);
            }
            """

        case .topology:
            return header + """
            kernel void mjit_topology(
                device const uint64_t* data    [[buffer(0)]],
                device uint64_t* results       [[buffer(1)]],
                constant int& faces            [[buffer(2)]],
                uint id [[thread_position_in_grid]])
            {
                if (faces != 6) { results[id] = 0; return; }
                uint offset = id * 6;
                const float P[36] = {
                    0.25f, -0.25f,  0.00f,  0.25f, -0.25f,  0.00f,
                   -0.25f,  0.50f, -0.25f, -0.25f,  0.50f, -0.25f,
                    0.00f, -0.25f,  0.25f,  0.00f, -0.25f,  0.25f,
                    0.25f, -0.25f,  0.00f,  0.25f, -0.25f,  0.00f,
                   -0.25f,  0.50f, -0.25f, -0.25f,  0.50f, -0.25f,
                    0.00f, -0.25f,  0.25f,  0.00f, -0.25f,  0.25f
                };
                float harmonic[6] = {0};
                float norm_sq = 0.0f;
                for (int i = 0; i < 6; ++i) {
                    for (int j = 0; j < 6; ++j) {
                        harmonic[i] += P[i*6 + j] * uint64_to_float(data[offset + j]);
                    }
                    norm_sq += harmonic[i] * harmonic[i];
                }
                float norm = sqrt(norm_sq);
                float kappa = 0.5f * tanh(norm * 3.0f) + 0.3f;
                results[id] = float_to_uint64(kappa);
            }
            """
        }
    }
}

// MARK: - Tipo de dato auxiliar

public enum DataType {
    case uint64
    case float64
}
