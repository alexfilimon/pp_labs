#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <cmath>

using namespace std;

#define GRID_SIZE 64
#define BLOCK_SIZE 32

const float degreesToRadiansCoefficient = 0.0174533;
const int minValue = 0;
const int maxValue = 360;

cudaError_t calculateWithCuda(float* c, float* a, float* b, unsigned int size);

__global__ void globalCalculateKernel(float* c, float* a, float* b)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    c[i * j] = sin(a[i * j]) * sin(a[i * j]) + cos(b[i * j]) * cos(b[i * j]) * cos(b[i * j]);
}

void initRandom(int arraySize, float* a)
{
    for (int i = 0; i < arraySize; i++)
        for (int j = 0; j < arraySize; j++)
            a[i * arraySize + j] = minValue + rand() % maxValue * degreesToRadiansCoefficient;

}

void initNull(int arraySize, float* a)
{
    for (int i = 0; i < arraySize; i++)
        for (int j = 0; j < arraySize; j++)
            a[i * arraySize + j] = 0;
}

void display(int arraySize, float* a)
{
    for (int i = 0; i < arraySize; i++)
    {
        for (int j = 0; j < arraySize; j++)
            cout << a[i * arraySize + j] << " ";
        cout << endl;
    }
}


// Helper function for using CUDA to add matrixes in parallel.
cudaError_t calculateWithCuda(float* c, float* a, float* b, unsigned int size)
{
    float* dev_a;
    float* dev_b;
    float* dev_c;

    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    float KernelTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three matrixes (two input, one output).
    cudaEventRecord(start, 0);
    cudaStatus = cudaMalloc((void**)&dev_c, (BLOCK_SIZE * BLOCK_SIZE) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_a, (BLOCK_SIZE * BLOCK_SIZE) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMalloc((void**)&dev_b, (BLOCK_SIZE * BLOCK_SIZE) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    printf("\nAllocating GPU buffers time:  %0.2f ms \n", KernelTime);

    // Copy input matrixes from host memory to GPU buffers.
    cudaEventRecord(start, 0);
    cudaStatus = cudaMemcpy(dev_a, a, (BLOCK_SIZE * BLOCK_SIZE) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, (BLOCK_SIZE * BLOCK_SIZE) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    printf("\nCopying input matrixes: host -> GPU  time:  %0.2f ms \n", KernelTime);

    // Launch a kernel on the GPU with one thread for each element.
    int numBlocks = GRID_SIZE;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    cout << "\nConfig settings: arraySize = " << size << ", numBlocks = " << numBlocks << ", threadsPerBlock(" << BLOCK_SIZE << ", " << BLOCK_SIZE << ")\n";

    // Global memory
    cudaEventRecord(start, 0);
    globalCalculateKernel <<< numBlocks, threadsPerBlock >>> (dev_c, dev_a, dev_b);
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    // cout << "\nGlobal result: " << KernelTime <<  " milliseconds\n";
    printf("\nGlobal memory work's time:  %0.2f ms \n", KernelTime);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "globalCalculateKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching globalCalculateKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output matrix from GPU buffer to host memory.
    cudaEventRecord(start, 0);
    cudaStatus = cudaMemcpy(c, dev_c, (BLOCK_SIZE * BLOCK_SIZE) * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&KernelTime, start, stop);
    printf("\nCopying output matri: GPU -> host time:  %0.2f ms \n", KernelTime);

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}



int main()
{
    srand(time(NULL));

    cout << "Enter array size: ";
    int arraySize = 0;
    cin >> arraySize;

    // Initialize arrays
    float* a = new float[arraySize * arraySize];
    float* b = new float[arraySize * arraySize];
    float* c = new float[arraySize * arraySize];
    initRandom(arraySize, a);
    initRandom(arraySize, b);
    initNull(arraySize, c);

    // Add matrixes in parallel
    cudaError_t cudaStatus = calculateWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        cout << "calculateWithCuda failed!\n";
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        cout << "cudaDeviceReset failed!\n";
        return 1;
    }

    return 0;
}