#include <cstdint>
#include <stdio.h>
#include <cuda_runtime.h>

constexpr uint64_t N = 1024 * 1024;
constexpr int TARGET_SM = 1;
constexpr int TOTAL_SMS = 96; // on A100D

__device__ __forceinline__ int load_global_strict(const int* ptr) {
    int ret;
    asm volatile (
        "ld.global.cg.u32 %0, [%1];"
        : "=r"(ret)
        : "l"(ptr)
        : "memory"
    );
    return ret;
}

__device__ __forceinline__ uint32_t get_smid() {
    uint32_t ret;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(ret));
    return ret;
}

__global__ void kernel(int *a, uint64_t *timings) {
    int smid = get_smid();
    if (smid != TARGET_SM) {
        return;
    }

    int acc = 0;
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        acc += load_global_strict(a + i);
    }

    __syncthreads();

    int acc2 = 0;
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        int *p = a + i;
        uint64_t start = clock64();
        int x = load_global_strict(p);
        acc += x; // loads are async on modern gpus, the data is required here so it's actually being waited on here ..
        uint64_t end = clock64();
        timings[i] = end-start;
    }

    a[1] = acc + acc2;

    a[0] = smid;
}

int main() {
    int smid;
    uint64_t *h_timings = (uint64_t*)malloc(N * sizeof(uint64_t));

    int *d_a;
    uint64_t *d_timings;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_timings, N * sizeof(uint64_t));

    kernel<<<TOTAL_SMS, 1>>>(d_a, d_timings);
    cudaMemcpy(h_timings, d_timings, N*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&smid, d_a, sizeof(int), cudaMemcpyDeviceToHost);

    printf("smid: %d\n", smid);
    printf("index;latency\n");
    for (int i = 0; i < N; i++) {
        printf("%d;%zu\n", i, h_timings[i]);
    }

    cudaFree(d_timings);
    cudaFree(d_a);
    free(h_timings);
    return 0;
}
