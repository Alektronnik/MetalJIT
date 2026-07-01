// ==========================================================================
// MetalJITExample.cpp
// ==========================================================================
// Ejemplo completo de la API C de MetalJIT.
// Usa directamente las funciones mjit_* y los wrappers tipados de
// MetalJITBinding.h.
//
// Compilacion (macOS, Apple Silicon, Objective-C++):
//   clang++ -std=c++17 -O2 -fobjc-arc -x objective-c++ MetalJITExample.mm \
//           -I ../MetalJITCore/Headers \
//           -I ../MetalJITCore/Internal \
//           ../MetalJITCore/Modules/*.mm \
//           -framework Metal -framework Foundation \
//           -o MetalJITExample
//
// Ejecucion:
//   ./MetalJITExample
// ==========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include "MetalJITCore.h"

// Macro auxiliar para tests
#define CHECK(expr, msg) do { \
    if (!(expr)) { fprintf(stderr, "  ✗ FALLO: %s\n", msg); exit(1); } \
} while(0)

void ejemplo_1_cpu_only() {
    printf("[1] Pipeline CPU-only\n");

    int handle = mjit_compile_pipeline(nullptr, nullptr, nullptr, 0);
    CHECK(handle > 0, "Compilar pipeline CPU");

    uint64_t input[]  = {10, 20, 30, 40, 50};
    uint64_t output[] = {0, 0, 0, 0, 0};

    int r = mjit_dispatch_uint64(handle, input, output, 5);
    CHECK(r == MJIT_SUCCESS, "Despachar UInt64");

    printf("    Entrada: [%llu, %llu, %llu, %llu, %llu]\n",
           (unsigned long long)input[0], (unsigned long long)input[1],
           (unsigned long long)input[2], (unsigned long long)input[3],
           (unsigned long long)input[4]);
    printf("    Salida:  [%llu, %llu, %llu, %llu, %llu]\n",
           (unsigned long long)output[0], (unsigned long long)output[1],
           (unsigned long long)output[2], (unsigned long long)output[3],
           (unsigned long long)output[4]);

    for (int i = 0; i < 5; i++) CHECK(output[i] == input[i], "Coincidencia CPU");

    mjit_destroy_pipeline(handle);
    printf("    ✓ OK\n\n");
}

void ejemplo_2_compilar_shader() {
    printf("[2] Compilacion Metal JIT\n");

    const char* shader = R"MSL(
    #include <metal_stdlib>
    using namespace metal;
    kernel void doble(
        device const uint64_t* in  [[buffer(0)]],
        device uint64_t* out       [[buffer(1)]],
        uint id [[thread_position_in_grid]])
    {
        out[id] = in[id] * 2;
    }
    )MSL";

    char errBuf[512] = {0};
    int handle = mjit_compile_pipeline(shader, "doble", errBuf, sizeof(errBuf));
    CHECK(handle > 0, "Compilar shader doble");

    printf("    Shader 'doble' compilado: handle=%d\n", handle);
    printf("    ✓ OK\n\n");

    mjit_destroy_pipeline(handle);
}

void ejemplo_3_float64() {
    printf("[3] Despacho Float64 (Double)\n");

    int handle = mjit_compile_pipeline(nullptr, nullptr, nullptr, 0);
    CHECK(handle > 0, "Pipeline Float64");

    double input[]  = {3.1415926535, 2.7182818284, 1.4142135623};
    double output[] = {0.0, 0.0, 0.0};

    int r = mjit_dispatch_float64(handle, input, output, 3);
    CHECK(r == MJIT_SUCCESS, "Despachar Float64");

    printf("    Entrada: [%.10f, %.10f, %.10f]\n", input[0], input[1], input[2]);
    printf("    Salida:  [%.10f, %.10f, %.10f]\n", output[0], output[1], output[2]);

    for (int i = 0; i < 3; i++)
        CHECK(output[i] == input[i], "Coincidencia Float64");

    mjit_destroy_pipeline(handle);
    printf("    ✓ OK\n\n");
}

void ejemplo_4_batch() {
    printf("[4] Batch dispatch (3 lotes de 4 elementos)\n");

    int handle = mjit_compile_pipeline(nullptr, nullptr, nullptr, 0);
    CHECK(handle > 0, "Pipeline batch");

    uint64_t payloads[] = {
        1,  2,  3,  4,
        10, 20, 30, 40,
        100, 200, 300, 400
    };
    uint64_t results[] = {0, 0, 0};

    int r = mjit_dispatch_batch_uint64(handle, payloads, results, 3, 4);
    CHECK(r == MJIT_SUCCESS, "Batch dispatch");

    printf("    Payloads:  [1,2,3,4 | 10,20,30,40 | 100,200,300,400]\n");
    printf("    Resultados: [%llu, %llu, %llu]\n",
           (unsigned long long)results[0],
           (unsigned long long)results[1],
           (unsigned long long)results[2]);

    CHECK(results[0] == 1,   "Batch[0]");
    CHECK(results[1] == 10,  "Batch[1]");
    CHECK(results[2] == 100, "Batch[2]");

    mjit_destroy_pipeline(handle);
    printf("    ✓ OK\n\n");
}

void ejemplo_5_multi_pipeline() {
    printf("[5] Multi-pipeline simultaneo\n");

    const char* triple = R"MSL(
    #include <metal_stdlib>
    using namespace metal;
    kernel void triple(
        device const uint64_t* in  [[buffer(0)]],
        device uint64_t* out       [[buffer(1)]],
        uint id [[thread_position_in_grid]])
    { out[id] = in[id] * 3; }
    )MSL";

    char errBuf[512] = {0};
    int cpu = mjit_compile_pipeline(nullptr, nullptr, nullptr, 0);
    int gpu = mjit_compile_pipeline(triple, "triple", errBuf, sizeof(errBuf));

    CHECK(cpu > 0, "Pipeline CPU");
    CHECK(gpu > 0, "Pipeline GPU");
    CHECK(cpu != gpu, "Handles distintos");

    printf("    Pipeline CPU: handle=%d\n", cpu);
    printf("    Pipeline GPU: handle=%d\n", gpu);

    mjit_destroy_pipeline(cpu);
    printf("    Pipeline CPU destruido. GPU sigue activo.\n");

    mjit_destroy_pipeline(gpu);
    printf("    ✓ OK\n\n");
}

void ejemplo_6_compilacion_fallida() {
    printf("[6] Compilacion fallida con mensaje de error\n");

    const char* roto = "kernel void roto() { ESTO_NO_COMPILA }";
    char errBuf[512] = {0};

    int handle = mjit_compile_pipeline(roto, "roto", errBuf, sizeof(errBuf));
    CHECK(handle < 0, "Debe fallar compilacion");
    CHECK(errBuf[0] != '\0', "Debe haber mensaje de error");

    printf("    Codigo de error: %d\n", handle);
    printf("    Mensaje: %s\n", errBuf);
    printf("    ✓ OK (esperado)\n\n");
}

void ejemplo_7_wrappers_tipados() {
    printf("[7] Wrappers tipados C\n");

    char err[256] = {0};
    int handle = MJIT_COMPILE(nullptr, nullptr, err);
    CHECK(handle > 0, "Compilar con macro");

    uint64_t in[]  = {100, 200, 300};
    uint64_t out[] = {0, 0, 0};

    int r = mjit_dispatch_uint64(handle, in, out, 3);
    CHECK(r == MJIT_SUCCESS, "Despachar con wrapper");

    printf("    Resultado: [%llu, %llu, %llu]\n",
           (unsigned long long)out[0],
           (unsigned long long)out[1],
           (unsigned long long)out[2]);
    CHECK(out[0] == 100 && out[1] == 200 && out[2] == 300, "Resultados macro");

    mjit_destroy_pipeline(handle);
    printf("    ✓ OK\n\n");
}

void ejemplo_8_dispatch_result() {
    printf("[8] Struct MJITDispatchResult\n");

    int handle = mjit_compile_pipeline(nullptr, nullptr, nullptr, 0);
    uint64_t buf[] = {1, 2, 3};

    MJITDispatchResult res = mjit_make_result(
        mjit_dispatch_uint64(handle, buf, buf, 3), 3, MJIT_TYPE_UINT64);

    printf("    Error: %d\n", res.error_code);
    printf("    Procesados: %d elementos\n", res.elements_processed);
    printf("    Tipo: %s\n", res.data_type_name);

    CHECK(res.error_code == MJIT_SUCCESS, "Resultado exitoso");
    CHECK(res.elements_processed == 3, "3 elementos");
    CHECK(strcmp(res.data_type_name, "UInt64") == 0, "Tipo UInt64");

    mjit_destroy_pipeline(handle);
    printf("    ✓ OK\n\n");
}

// ======================================================================
// MAIN
// ======================================================================
int main() {
    printf("================================================================\n");
    printf(" MetalJIT — Ejemplo C++\n");
    printf("================================================================\n\n");

    ejemplo_1_cpu_only();
    ejemplo_2_compilar_shader();
    ejemplo_3_float64();
    ejemplo_4_batch();
    ejemplo_5_multi_pipeline();
    ejemplo_6_compilacion_fallida();
    ejemplo_7_wrappers_tipados();
    ejemplo_8_dispatch_result();

    printf("================================================================\n");
    printf(" MetalJIT C++ — Todos los ejemplos completados correctamente\n");
    printf("================================================================\n");

    return 0;
}
