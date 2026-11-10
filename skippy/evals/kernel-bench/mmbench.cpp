// mmbench: times one ggml MUL_MAT (or MUL_MAT_ID) shape on the first GPU backend.
// usage: mmbench <type> <M rows> <K> <N cols> [n_expert n_expert_used] -> prints "MMBENCH type M K N us tflops"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static ggml_type parse_type(const char * s) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && strcmp(n, s) == 0) return (ggml_type) t;
    }
    fprintf(stderr, "unknown type %s\n", s); exit(1);
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: mmbench type M K N [n_expert n_used]\n"); return 1; }
    const ggml_type type = parse_type(argv[1]);
    const int64_t M = atoll(argv[2]), K = atoll(argv[3]), N = atoll(argv[4]);
    const int64_t E = argc > 6 ? atoll(argv[5]) : 0, U = argc > 6 ? atoll(argv[6]) : 0;

    ggml_backend_load_all();
    // Integrated GPUs (DGX Spark, Apple) can register as IGPU rather than GPU.
    ggml_backend_t be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    if (!be) {
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU, nullptr);
    }
    if (!be) { fprintf(stderr, "no GPU backend\n"); return 1; }

    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = E ? ggml_new_tensor_3d(ctx, type, K, M, E) : ggml_new_tensor_2d(ctx, type, K, M);
    ggml_tensor * b = E ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, U, N) : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
    ggml_tensor * ids = E ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, U, N) : nullptr;
    ggml_tensor * out = E ? ggml_mul_mat_id(ctx, a, b, ids) : ggml_mul_mat(ctx, a, b);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    {
        const int64_t n = K * M * (E ? E : 1);
        std::vector<float> f(n);
        for (auto & x : f) x = nd(rng) * 0.05f;
        std::vector<uint8_t> q(ggml_nbytes(a));
        ggml_quantize_chunk(type, f.data(), q.data(), 0, n / K, K, nullptr);
        ggml_backend_tensor_set(a, q.data(), 0, q.size());
    }
    {
        std::vector<float> f(ggml_nelements(b));
        for (auto & x : f) x = nd(rng);
        ggml_backend_tensor_set(b, f.data(), 0, f.size() * sizeof(float));
    }
    if (ids) {
        std::vector<int32_t> v(U * N);
        std::uniform_int_distribution<int> ud(0, (int) E - 1);
        for (int64_t t = 0; t < N; t++) {
            std::vector<int> pick;
            while ((int64_t) pick.size() < U) { int e = ud(rng); bool dup = false; for (int p : pick) dup |= p == e; if (!dup) pick.push_back(e); }
            for (int64_t u = 0; u < U; u++) v[t * U + u] = pick[u];
        }
        ggml_backend_tensor_set(ids, v.data(), 0, v.size() * sizeof(int32_t));
    }

    for (int i = 0; i < 3; i++) ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);
    int iters = 0;
    auto t0 = std::chrono::steady_clock::now();
    double el = 0;
    while (el < 0.5 || iters < 5) {
        ggml_backend_graph_compute(be, gf);
        ggml_backend_synchronize(be);
        iters++;
        el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    const double us = el / iters * 1e6;
    const double flops = 2.0 * M * K * N * (E ? U : 1);
    printf("MMBENCH %s %lld %lld %lld %lld %lld %.1f %.2f\n", argv[1], (long long) M, (long long) K, (long long) N,
           (long long) E, (long long) U, us, flops / us / 1e6);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;
}
