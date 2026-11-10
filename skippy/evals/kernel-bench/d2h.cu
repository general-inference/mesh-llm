// d2h: device-to-host copy bandwidth into pageable, registered and pinned memory, and staged
// through a pinned bounce buffer. usage: d2h [MiB per copy] [copies]
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    const size_t mib = argc > 1 ? atoi(argv[1]) : 16, n = argc > 2 ? atoi(argv[2]) : 16, bytes = mib << 20;
    void * dev; cudaMalloc(&dev, bytes); cudaMemset(dev, 1, bytes);
    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    auto run = [&](const char * name, void * dst) {
        cudaMemcpyAsync(dst, dev, bytes, cudaMemcpyDeviceToHost, s); cudaStreamSynchronize(s);
        double t = now();
        for (size_t i = 0; i < n; i++) { cudaMemcpyAsync(dst, dev, bytes, cudaMemcpyDeviceToHost, s); cudaStreamSynchronize(s); }
        double dt = now() - t;
        printf("D2H %-18s %5zu MiB x %zu: %8.2f GB/s\n", name, mib, n, bytes * n / dt / 1e9);
    };
    std::vector<char> pageable(bytes); memset(pageable.data(), 0, bytes);
    run("pageable", pageable.data());
    std::vector<char> reg(bytes); memset(reg.data(), 0, bytes);
    double t = now(); cudaHostRegister(reg.data(), bytes, cudaHostRegisterPortable); double treg = now() - t;
    run("registered", reg.data());
    printf("cudaHostRegister %zu MiB: %.1f ms\n", mib, treg * 1e3);
    cudaHostUnregister(reg.data());
    void * pinned; cudaMallocHost(&pinned, bytes);
    run("pinned", pinned);
    // staged: pinned bounce buffer of 8 MiB, CPU memcpy into the pageable destination
    const size_t chunk = 8 << 20; void * bounce[2]; cudaMallocHost(&bounce[0], chunk); cudaMallocHost(&bounce[1], chunk);
    cudaEvent_t ev[2]; cudaEventCreate(&ev[0]); cudaEventCreate(&ev[1]);
    t = now();
    for (size_t r = 0; r < n; r++) {
        size_t off = 0; int b = 0; size_t prev_len = 0, prev_off = 0; bool have = false;
        while (off < bytes || have) {
            size_t len = off < bytes ? std::min(chunk, bytes - off) : 0;
            if (len) { cudaMemcpyAsync(bounce[b], (char *) dev + off, len, cudaMemcpyDeviceToHost, s); cudaEventRecord(ev[b], s); }
            if (have) { cudaEventSynchronize(ev[b ^ 1]); memcpy(pageable.data() + prev_off, bounce[b ^ 1], prev_len); }
            have = len != 0; prev_len = len; prev_off = off; off += len; b ^= 1;
        }
    }
    double dt = now() - t;
    printf("D2H %-18s %5zu MiB x %zu: %8.2f GB/s\n", "staged-8MiB", mib, n, bytes * n / dt / 1e9);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("device %s integrated=%d pageableMemoryAccess=%d concurrentManagedAccess=%d\n", p.name, p.integrated, p.pageableMemoryAccess, p.concurrentManagedAccess);
    return 0;
}
