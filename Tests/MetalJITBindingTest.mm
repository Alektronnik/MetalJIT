// ==========================================================================
// MetalJITBindingTest.mm — Test de integracion del binding C
// ==========================================================================
// Compilacion:
//   clang++ -std=c++17 -O0 -g -fobjc-arc -x objective-c++ MetalJITBindingTest.mm \
//           -I ../Sources/MetalJITCore/Headers \
//           ../Sources/MetalJITCore/Modules/*.mm \
//           -framework Metal -framework Foundation \
//           -o /tmp/MetalJITBindingTest
// Ejecucion: /tmp/MetalJITBindingTest
// ==========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "MetalJITCore.h"

static int passed = 0;
static int failed = 0;

#define CHECK(cond, msg) do { \
    if (cond) { passed++; printf("  ✓ %s\n", msg); } \
    else      { failed++; printf("  ✗ %s\n", msg); } \
} while(0)

#define CHECK_ERR(expr, expected_err, msg) do { \
    int _r = (expr); \
    if (_r == expected_err || _r == -expected_err) { passed++; printf("  ✓ %s\n", msg); } \
    else { failed++; printf("  ✗ %s (codigo=%d, esperado=%d)\n", msg, _r, expected_err); } \
} while(0)

int main() {
    printf("============================================================\n");
    printf(" MetalJIT Binding C — Test de integracion\n");
    printf("============================================================\n");

    // --- Compilacion ---
    printf("\n[Compilacion]\n");

    int cpu = mjit_compile_pipeline(nullptr, nullptr, nullptr, 0);
    CHECK(cpu > 0, "compile CPU devuelve handle > 0");

    const char* shader = "#include <metal_stdlib>\nusing namespace metal;\n"
                         "kernel void test_k(device uint64_t* out [[buffer(0)]], "
                         "uint id [[thread_position_in_grid]]) { out[id] = id; }";
    char err[512] = {0};
    int gpu = mjit_compile_pipeline(shader, "test_k", err, sizeof(err));
    CHECK(gpu > 0, "compile GPU devuelve handle > 0");

    int bad = mjit_compile_pipeline("basura", "x", err, sizeof(err));
    CHECK(bad < 0, "shader invalido devuelve error");

    // --- Dispatch ---
    printf("\n[Dispatch]\n");

    uint64_t in[]  = {10, 20, 30, 40, 50};
    uint64_t out[] = {0, 0, 0, 0, 0};
    int r = mjit_dispatch_uint64(cpu, in, out, 5);
    CHECK(r == MJIT_SUCCESS, "dispatch UInt64 OK");
    CHECK(out[0] == 10 && out[4] == 50, "datos correctos");

    double fin[]  = {3.14, 2.71, 1.41};
    double fout[] = {0, 0, 0};
    r = mjit_dispatch_float64(cpu, fin, fout, 3);
    CHECK(r == MJIT_SUCCESS, "dispatch Float64 OK");
    CHECK(fout[0] == 3.14 && fout[2] == 1.41, "datos Float64 correctos");

    // --- Batch ---
    printf("\n[Batch]\n");

    uint64_t pl[] = {1, 2, 3, 10, 20, 30, 100, 200, 300};
    uint64_t rl[] = {0, 0, 0};
    r = mjit_dispatch_batch_uint64(cpu, pl, rl, 3, 3);
    CHECK(r == MJIT_SUCCESS, "batch dispatch OK");
    CHECK(rl[0] == 1 && rl[1] == 10 && rl[2] == 100, "batch resultados correctos");

    // --- Multi-pipeline ---
    printf("\n[Multi-pipeline]\n");

    int h1 = mjit_compile_pipeline(nullptr, nullptr, nullptr, 0);
    int h2 = mjit_compile_pipeline(nullptr, nullptr, nullptr, 0);
    CHECK(h1 != h2, "handles unicos");
    r = mjit_destroy_pipeline(h1);
    CHECK(r == MJIT_SUCCESS, "destroy pipeline OK");

    // --- Utilidades ---
    printf("\n[Utilidades]\n");

    CHECK(mjit_data_type_size(MJIT_TYPE_UINT64) == 8, "UInt64 = 8 bytes");
    CHECK(mjit_data_type_size(MJIT_TYPE_FLOAT32) == 4, "Float32 = 4 bytes");
    CHECK(strcmp(mjit_data_type_name(MJIT_TYPE_FLOAT64), "Float64") == 0, "nombre Float64");

    // --- DispatchResult ---
    printf("\n[DispatchResult]\n");

    MJITDispatchResult dr = mjit_make_result(MJIT_SUCCESS, 42, MJIT_TYPE_UINT64);
    CHECK(dr.error_code == MJIT_SUCCESS, "result success code");
    CHECK(dr.elements_processed == 42, "result elements count");

    MJITDispatchResult dr_err = mjit_make_result(MJIT_ERR_INVALID_BUFFER, 10, MJIT_TYPE_FLOAT64);
    CHECK(dr_err.elements_processed == 0, "error => 0 processed");

    // --- Limpieza ---
    mjit_destroy_pipeline(cpu);
    mjit_destroy_pipeline(gpu);
    mjit_destroy_pipeline(h2);

    // --- Resumen ---
    printf("\n============================================================\n");
    int total = passed + failed;
    printf(" Resultado: %d/%d pasaron\n", passed, total);
    if (failed > 0) {
        printf(" %d tests FALLIDOS\n", failed);
        return 1;
    }
    printf(" Todos los tests pasaron correctamente\n");
    printf("============================================================\n");
    return 0;
}
