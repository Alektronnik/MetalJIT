// ==========================================================================
// MetalJITBenchmark.mm — Benchmark standalone C++ (Objective-C++)
// ==========================================================================
// Compilacion:
//   clang++ -std=c++17 -O2 -fobjc-arc -x objective-c++ MetalJITBenchmark.mm \
//           -I ../Sources/MetalJITCore/Headers \
//           -I ../Sources/MetalJITCore/Internal \
//           ../Sources/MetalJITCore/Modules/*.mm \
//           -framework Metal -framework Foundation \
//           -o /tmp/MetalJITBenchmark
// Ejecucion: /tmp/MetalJITBenchmark
// ==========================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include "MetalJITCore.h"

static int g_checks = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "  ✗ BENCH FAIL: %s\n", msg); exit(1); } \
    g_checks++; \
} while(0)

static double bench_now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define MEASURE(label, iterations, block) do { \
    double start = bench_now(); \
    for (int _i = 0; _i < iterations; ++_i) { block; } \
    double elapsed = bench_now() - start; \
    printf("  %-38s %8.4f s", label, elapsed / iterations); \
    if (iterations > 1) printf(" (x%d)", iterations); \
    printf("\n"); \
} while(0)

#define DIVIDER() printf("------------------------------------------------------------------\n")

int main() {
    const int COUNT   = 1000000;
    const int BATCHES = 100000;
    const int PER_B   = 6;

    printf("=================================================================\n");
    printf(" MetalJIT Benchmark C++ — %d elementos / %d lotes\n", COUNT, BATCHES);
    printf("=================================================================\n");

    char errBuf[512] = {0};

    // --- CORE ---
    printf("\n[CORE]\n");
    int cpu = mjit_compile_pipeline(nullptr, nullptr, nullptr, 0);
    CHECK(cpu > 0, "compile CPU pipeline");

    uint64_t* uin  = new uint64_t[COUNT];
    uint64_t* uout = new uint64_t[COUNT];
    for (int i = 0; i < COUNT; ++i) uin[i] = 1;
    MEASURE("dispatch UInt64", 1, {
        int r = mjit_dispatch_uint64(cpu, uin, uout, COUNT);
        CHECK(r == MJIT_SUCCESS, "dispatch UInt64 OK");
    });
    CHECK(uout[0] == 1, "UInt64 result valid");

    double* fin  = new double[COUNT];
    double* fout = new double[COUNT];
    for (int i = 0; i < COUNT; ++i) fin[i] = 3.14159;
    MEASURE("dispatch Float64", 1, {
        int r = mjit_dispatch_float64(cpu, fin, fout, COUNT);
        CHECK(r == MJIT_SUCCESS, "dispatch Float64 OK");
    });
    CHECK(fout[0] > 3.0, "Float64 result valid");

    delete[] uin; delete[] uout; delete[] fin; delete[] fout;
    DIVIDER();

    // --- BATCH ---
    printf("\n[BATCH]\n");
    int bp = mjit_compile_pipeline(nullptr, nullptr, nullptr, 0);
    CHECK(bp > 0, "compile batch pipeline");

    uint64_t* pl = new uint64_t[BATCHES * PER_B];
    uint64_t* rl = new uint64_t[BATCHES];
    for (int i = 0; i < BATCHES * PER_B; ++i) pl[i] = 1;

    char bLabel[64];
    snprintf(bLabel, sizeof(bLabel), "batch UInt64 (%dk x %d)", BATCHES/1000, PER_B);
    MEASURE(bLabel, 1, {
        int r = mjit_dispatch_batch_uint64(bp, pl, rl, BATCHES, PER_B);
        CHECK(r == MJIT_SUCCESS, "batch dispatch OK");
    });
    CHECK(rl[0] > 0, "batch result valid");

    delete[] pl; delete[] rl;
    mjit_destroy_pipeline(bp);
    DIVIDER();

    // --- COMPILER ---
    printf("\n[COMPILER]\n");
    const char* shader = R"MSL(
    #include <metal_stdlib>
    using namespace metal;
    kernel void bk(device const uint64_t* in [[buffer(0)]],
                   device uint64_t* out [[buffer(1)]],
                   uint id [[thread_position_in_grid]]) { out[id] = in[id] * 2; }
    )MSL";

    MEASURE("compilar JIT", 10, {
        int h = mjit_compile_pipeline(shader, "bk", errBuf, sizeof(errBuf));
        CHECK(h > 0, "JIT compile OK");
        mjit_destroy_pipeline(h);
    });

    MEASURE("compilar CPU-only", 10, {
        int h = mjit_compile_pipeline(nullptr, nullptr, nullptr, 0);
        CHECK(h > 0, "CPU compile OK");
        mjit_destroy_pipeline(h);
    });
    DIVIDER();

    // --- BINDING ---
    printf("\n[BINDING]\n");
    int bh = mjit_compile_pipeline(nullptr, nullptr, nullptr, 0);
    CHECK(bh > 0, "binding pipeline");
    uint64_t* bi = new uint64_t[COUNT];
    uint64_t* bo = new uint64_t[COUNT];
    for (int i = 0; i < COUNT; ++i) bi[i] = 1;
    MEASURE("dispatch C wrapper", 1, {
        int r = mjit_dispatch_uint64(bh, bi, bo, COUNT);
        CHECK(r == MJIT_SUCCESS, "wrapper dispatch OK");
    });
    CHECK(bo[0] == 1, "wrapper result valid");
    delete[] bi; delete[] bo;
    mjit_destroy_pipeline(bh);
    DIVIDER();

    // --- HEAP ---
    printf("\n[HEAP]\n");
    MEASURE("1000 allocs + reset", 5, {
        int h = mjit_heap_create(10 * 1024 * 1024);
        CHECK(h > 0, "heap create OK");
        for (int i = 0; i < 1000; ++i) mjit_heap_allocate(h, 1024, 256);
        int rr = mjit_heap_reset(h);
        CHECK(rr == 0, "heap reset OK");
        mjit_heap_destroy(h);
    });
    DIVIDER();

    // --- CACHE ---
    printf("\n[CACHE]\n");
    const char* cs = "#include <metal_stdlib>\nusing namespace metal;\n"
                     "kernel void cb(device uint64_t* out [[buffer(0)]], "
                     "uint id [[thread_position_in_grid]]) { out[id]=id; }";
    const char* cp = "/tmp/mjt_bench_cache.metallib";

    MEASURE("save to cache", 5, {
        int r = mjit_cache_compile_and_save(cs, cp, errBuf, sizeof(errBuf));
        CHECK(r == MJIT_SUCCESS, "cache save OK");
    });
    MEASURE("load from cache", 5, {
        int h = mjit_cache_load_library(cp, "cb", errBuf, sizeof(errBuf));
        CHECK(h > 0, "cache load OK");
        mjit_destroy_pipeline(h);
    });
    DIVIDER();

    // --- NATIVE (con checksum para evitar optimizacion fantasma) ---
    printf("\n[NATIVE]\n");
    float* nf = new float[COUNT];
    float* nfo = new float[COUNT];
    for (int i = 0; i < COUNT; ++i) nf[i] = 1.0f;
    float checksum_f = 0;
    MEASURE("Float * 2.0 bucle", 1, for (int i = 0; i < COUNT; ++i) { nfo[i] = nf[i] * 2.0f; checksum_f += nfo[i]; });
    printf("    (checksum: %.1f)\n", checksum_f);
    CHECK(checksum_f > 0, "float checksum valid");
    delete[] nf; delete[] nfo;

    double* nd = new double[COUNT];
    double* ndo = new double[COUNT];
    for (int i = 0; i < COUNT; ++i) nd[i] = 1.0;
    double checksum_d = 0;
    MEASURE("Double * 2.0 bucle", 1, for (int i = 0; i < COUNT; ++i) { ndo[i] = nd[i] * 2.0; checksum_d += ndo[i]; });
    printf("    (checksum: %.1f)\n", checksum_d);
    CHECK(checksum_d > 0, "double checksum valid");
    delete[] nd; delete[] ndo;

    mjit_destroy_pipeline(cpu);

    printf("\n=================================================================\n");
    printf(" MetalJIT Benchmark C++ — Completado (%d checks OK)\n", g_checks);
    printf("=================================================================\n");

    return 0;
}
