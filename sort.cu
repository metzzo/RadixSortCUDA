#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <chrono>

// #define CUDA_DEBUG
// #define CPU_DEBUG

#ifndef EXPERIMENT_REPETITION
#define EXPERIMENT_REPETITION (5)
#endif

#ifndef NUM_ELEMENTS
#define NUM_ELEMENTS (4096)
#endif

#ifndef SCAN_NUM_THREADS
#define SCAN_NUM_THREADS (512)
#endif

#ifdef CUDA_DEBUG
#define CHECK_CUDA
#endif

// #ifdef CHECK_CUDA
#define gpuErrchk()                    \
    {                                  \
        gpuAssert(__FILE__, __LINE__); \
    }
// #else
// #define gpuErrchk()
// #endif

typedef unsigned int uint;

void print_bits(size_t const size, void const *const ptr)
{
    unsigned char *b = (unsigned char *)ptr;
    unsigned char byte;
    int i, j;

    for (i = size - 1; i >= 0; i--)
    {
        for (j = 7; j >= 0; j--)
        {
            byte = (b[i] >> j) & 1;
            printf("%u", byte);
        }
    }
    puts("");
}

inline void gpuAssert(const char *file, int line, bool abort = true)
{
    cudaDeviceSynchronize();
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            getchar();
            exit(code);
        }
    }
}

std::vector<uint> generate_random_numbers(int n)
{
    std::vector<uint> v(n);
    for (int i = 0; i < n; i++)
    {
        v[i] = rand(); // % 100;
    }
    return v;
}

bool is_sorted(const std::vector<uint> &v)
{
    for (int i = 1; i < v.size(); i++)
    {
        if (v[i] < v[i - 1])
        {
            return false;
        }
    }
    return true;
}

void print_numbers(const std::vector<uint> &v)
{
    if (v.size() > 100)
    {
        return;
    }

    for (int i = 0; i < v.size(); i++)
    {
        printf("%u ", v[i]);
    }
    printf(" => ");
    if (is_sorted(v))
    {
        printf("Sorted!\n");
    }
    else
    {
        printf("NOT Sorted!\n");
    }
}

void sort(std::vector<uint> &v)
{
    std::sort(v.begin(), v.end());
}

void radix_sort_cpu(std::vector<uint> &v)
{
    std::vector<uint> v1(v.size());
    std::vector<uint> v2(v.size());
    // copy all to v1
    for (int i = 0; i < v.size(); i++)
    {
        v1[i] = v[i];
        v2[i] = v[i];
    }
    std::vector<uint> *v1_ptr = &v1;
    std::vector<uint> *v2_ptr = &v2;

    for (int bit = 0; bit <= 31; bit++)
    {
        int mask = 1 << bit;
#ifdef CPU_DEBUG
        print_bits(sizeof(mask), &mask);
#endif

        int ones = 0, zeros = 0;
        for (int i = 0; i < v1_ptr->size(); i++)
        {
            if ((*v1_ptr)[i] & mask)
            {
                ones++;
            }
            else
            {
                zeros++;
            }
        }
#ifdef CPU_DEBUG
        printf("bit %d: ones: %d, zeros: %d\n", bit, ones, zeros);
#endif

        // simulate prefix sum
        int cum_sum_0 = 0;
        int cum_sum_1 = 0;
        for (int i = 0; i < v1_ptr->size(); i++)
        {
            if ((*v1_ptr)[i] & mask)
            {
                (*v2_ptr)[zeros + cum_sum_1] = (*v1_ptr)[i];
                cum_sum_1++;
            }
            else
            {
                (*v2_ptr)[cum_sum_0] = (*v1_ptr)[i];
                cum_sum_0++;
            }
        }
#ifdef CPU_DEBUG
        print_random_numbers(*v2_ptr);
#endif

        // swap pointers
        std::vector<uint> *tmp = v1_ptr;
        v1_ptr = v2_ptr;
        v2_ptr = tmp;
    }

    // copy back
    memcpy(v.data(), v1_ptr->data(), v.size() * sizeof(uint));
}

typedef struct RadixSortCudaData
{
    uint *d_v1;
    uint *d_v2;
    uint *d_scanned_0;
    uint *d_scanned_1;
    uint *d_chunk_sum_0;
    uint *d_chunk_sum_1;
    uint size;

    uint *d_num_zeros;
} RadixSortCudaData;

void create_radix_sort_cuda_data(RadixSortCudaData &gpu_data, int size)
{
    gpu_data.size = size;

    cudaMalloc((void **)&gpu_data.d_v1, size * sizeof(uint));
    gpuErrchk();
    cudaMalloc((void **)&gpu_data.d_v2, size * sizeof(uint));
    gpuErrchk();

    cudaMalloc((void **)&gpu_data.d_num_zeros, sizeof(uint));
    gpuErrchk();

    cudaMalloc((void **)&gpu_data.d_scanned_0, size * sizeof(uint));
    gpuErrchk();

    cudaMalloc((void **)&gpu_data.d_scanned_1, size * sizeof(uint));
    gpuErrchk();

    cudaMalloc((void **)&gpu_data.d_chunk_sum_0, sizeof(uint));
    gpuErrchk();

    cudaMalloc((void **)&gpu_data.d_chunk_sum_1, sizeof(uint));
    gpuErrchk();
}

#ifdef CUDA_DEBUG
__global__ void debug_cuda_data(RadixSortCudaData data, int bit)
{
    uint mask = 1 << bit;
    printf("bit %d, size %u \n", bit, data.size);
    printf("v1:\t\t");
    for (int i = 0; i < data.size; i++)
    {
        printf("%u(%u)\t", data.d_v1[i], (data.d_v1[i] & mask) != 0);
    }
    printf("\n");
    printf("v2:\t\t");
    for (int i = 0; i < data.size; i++)
    {
        printf("%u\t", data.d_v2[i]);
    }
    printf("\n");
    printf("scanned 0:\t");
    for (int i = 0; i < data.size; i++)
    {
        printf("%u\t", data.d_scanned_0[i]);
    }
    printf("\n");
    printf("scanned 1:\t");
    for (int i = 0; i < data.size; i++)
    {
        printf("%u\t", data.d_scanned_1[i]);
    }
    printf("\n");

    printf("num zeros: %u, chunk sum 0: %u, chunk sum 1: %u\n", data.d_num_zeros[0], data.d_chunk_sum_0[0], data.d_chunk_sum_1[0]);
}
#endif

// inspired by: https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
__global__ void scan(RadixSortCudaData data, uint *target, int bit, int relevant_bit, int chunk_idx, uint *target_chunk_sum)
{
    // work efficient parallel scan with up-sweep and down-sweep phase
    __shared__ uint tmp[2 * SCAN_NUM_THREADS];
    int thread_idx = threadIdx.x;
    uint chunk_offset = chunk_idx * SCAN_NUM_THREADS;
    if (chunk_offset + 2 * thread_idx + 1 >= data.size)
    {
        return;
    }

    int offset = 1;
    int mask = 1 << bit;
    uint n = min(data.size, SCAN_NUM_THREADS * 2);
    uint prev_chunk_sum = target_chunk_sum[0]; // save prev_chunk_sum for later use
    __syncthreads();                           // wait for prev_chunk_sum to be loaded in all threads

    // load bit values into shared memory
    int val1 = (2 * thread_idx < data.size) ? ((data.d_v1[chunk_offset + 2 * thread_idx] & mask) != 0) == relevant_bit : 0;
    int val2 = (2 * thread_idx + 1 < data.size) ? ((data.d_v1[chunk_offset + 2 * thread_idx + 1] & mask) != 0) == relevant_bit : 0;
    tmp[2 * thread_idx] = val1;
    tmp[2 * thread_idx + 1] = val2;
    atomicAdd(target_chunk_sum, val1 + val2);

    // up sweep phase
    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (thread_idx < d)
        {
            int ai = offset * (2 * thread_idx + 1) - 1;
            int bi = offset * (2 * thread_idx + 2) - 1;
            tmp[bi] += tmp[ai];
        }
        offset *= 2;
    }

    // reset last element
    if (thread_idx == 0)
    {
        tmp[n - 1] = 0;
    }

    // down sweep phase
    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (thread_idx < d)
        {
            int ai = offset * (2 * thread_idx + 1) - 1;
            int bi = offset * (2 * thread_idx + 2) - 1;
            uint t = tmp[ai];
            tmp[ai] = tmp[bi];
            tmp[bi] += t;
        }
    }
    __syncthreads();

    // result into memory => add largest value of previous chunk
    target[chunk_offset + 2 * thread_idx] = tmp[2 * thread_idx] + prev_chunk_sum;
    target[chunk_offset + 2 * thread_idx + 1] = tmp[2 * thread_idx + 1] + prev_chunk_sum;
}

__global__ void merge_scanned(RadixSortCudaData data, uint bit)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= data.size)
    {
        return;
    }
    uint mask = 1 << bit;
    uint value = data.d_v1[thread_id];
    uint idx = (value & mask) ? (data.d_scanned_1[thread_id] + data.d_num_zeros[0]) : (data.d_scanned_0[thread_id]);
    data.d_v2[idx] = value;

    data.d_scanned_0[thread_id] = 0;
    data.d_scanned_1[thread_id] = 0;
}

__global__ void count_zeros(RadixSortCudaData data, int bit)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= data.size)
    {
        return;
    }
    uint mask = 1 << bit;
    uint masked = data.d_v1[thread_id] & mask;
    if (masked == 0)
    {
        atomicAdd(data.d_num_zeros, 1);
    }
}

__global__ void reset(RadixSortCudaData data)
{
    data.d_num_zeros[0] = 0;
    data.d_chunk_sum_0[0] = 0;
    data.d_chunk_sum_1[0] = 0;
}

void radix_sort_gpu(std::vector<uint> &v, RadixSortCudaData &gpu_data)
{
    cudaMemcpy(gpu_data.d_v1, v.data(), gpu_data.size * sizeof(uint), cudaMemcpyDefault);
    gpuErrchk();

    for (int bit = 0; bit <= 31; bit++)
    {
#ifdef CUDA_DEBUG
        printf("bit %d\n", bit);
#endif
        reset<<<1, 1>>>(gpu_data);
        gpuErrchk();

        count_zeros<<<v.size() / 256 + 1, 256>>>(gpu_data, bit);
        gpuErrchk();

#ifdef CUDA_DEBUG
        printf("counted zeros\n");
        debug_cuda_data<<<1, 1>>>(gpu_data, bit);
        gpuErrchk();
#endif
        for (int chunk_idx = 0; chunk_idx < v.size() / SCAN_NUM_THREADS + 1; chunk_idx++)
        {
            scan<<<1, SCAN_NUM_THREADS / 2>>>(gpu_data, gpu_data.d_scanned_0, bit, 0, chunk_idx, gpu_data.d_chunk_sum_0);
            gpuErrchk();

#ifdef CUDA_DEBUG
            printf("scanned 0\n");
            debug_cuda_data<<<1, 1>>>(gpu_data, bit);
            gpuErrchk();
#endif
            scan<<<1, SCAN_NUM_THREADS / 2>>>(gpu_data, gpu_data.d_scanned_1, bit, 1, chunk_idx, gpu_data.d_chunk_sum_1);
            gpuErrchk();
#ifdef CUDA_DEBUG
            printf("scanned 1\n");
            debug_cuda_data<<<1, 1>>>(gpu_data, bit);
            gpuErrchk();
#endif
        }

        merge_scanned<<<v.size() / 256 + 1, 256>>>(gpu_data, bit);
        gpuErrchk();

#ifdef CUDA_DEBUG
        printf("merged\n");
        debug_cuda_data<<<1, 1>>>(gpu_data, bit);
        gpuErrchk();
#endif

        // ping pong buffer
        uint *tmp = gpu_data.d_v1;
        gpu_data.d_v1 = gpu_data.d_v2;
        gpu_data.d_v2 = tmp;
    }

    cudaMemcpy(v.data(), gpu_data.d_v1, gpu_data.size * sizeof(uint), cudaMemcpyDefault);
    gpuErrchk();
}

int main(int argc, char *argv[])
{
    // without loss of generality I assume power of two size array
    // in case it is not a power of 2, we can
    // a) pad with missing numbers or
    // b) write a custom prefix sum kernel handling the delta of the residual between the lowest fitting power of 2 and the actual size
    assert(ceil(log2(NUM_ELEMENTS)) == floor(log2(NUM_ELEMENTS)));

    RadixSortCudaData gpu_data;
    create_radix_sort_cuda_data(gpu_data, NUM_ELEMENTS);

    std::vector<uint> v_original(generate_random_numbers(NUM_ELEMENTS));
    std::vector<uint> v_cpu, v_gpu;

    print_numbers(v_original);

    printf("Run CPU...\n");
    std::vector<std::chrono::duration<int64_t, std::nano>> sequential_times(EXPERIMENT_REPETITION);
    for (int repeat = 0; repeat < EXPERIMENT_REPETITION; repeat++)
    {
        v_cpu = std::vector<uint>(v_original);

        auto start = std::chrono::high_resolution_clock::now();
        // radix_sort_cpu(v_cpu);
        sort(v_cpu);
        sequential_times[repeat] = std::chrono::high_resolution_clock::now() - start;
    }
    std::sort(sequential_times.begin(), sequential_times.end());

    printf("Run GPU...\n");
    std::vector<std::chrono::duration<int64_t, std::nano>> parallel_times(EXPERIMENT_REPETITION);
    for (int repeat = 0; repeat < EXPERIMENT_REPETITION; repeat++)
    {
        v_gpu = std::vector<uint>(v_original);
        auto start = std::chrono::high_resolution_clock::now();
        radix_sort_gpu(v_gpu, gpu_data);
        parallel_times[repeat] = std::chrono::high_resolution_clock::now() - start;
    }
    std::sort(parallel_times.begin(), parallel_times.end());

    // ensure that both algorithms produce the same result
    int any_mismatch = 0;
    for (int i = 0; i < v_cpu.size(); i++)
    {
        if (v_cpu[i] != v_gpu[i])
        {
            printf("Error at index %d: %u %u\n", i, v_cpu[i], v_gpu[i]);
            any_mismatch = 1;
        }
    }
    if (any_mismatch == 0)
    {
        printf("Success!\n");
    }
    else
    {
        print_numbers(v_gpu);
        printf("Failure!\n");
        assert(0);
    }

    assert(is_sorted(v_cpu));
    assert(is_sorted(v_gpu));

    double median_sequential_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sequential_times[sequential_times.size() / 2]).count();
    double median_parallel_ms = std::chrono::duration_cast<std::chrono::milliseconds>(parallel_times[parallel_times.size() / 2]).count();
    printf("Median CPU time: %f ms\n", median_sequential_ms);
    printf("Median GPU time: %f ms\n", median_parallel_ms);
}