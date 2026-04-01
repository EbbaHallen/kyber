#include "opencl.h"
#include "ntt_kernels.h"
#include "params.h"
#include <stdio.h>

gpu_ctx g_ctx;

#define CHECK(err) if (err != CL_SUCCESS) { \
    printf("Error %d at line %d\n", err, __LINE__); exit(1); }

void opencl_init() {
    cl_platform_id platform;
    cl_device_id device;
    cl_int err;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);

    g_ctx.context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue_properties props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0
    };

    g_ctx.queue = clCreateCommandQueueWithProperties(
        g_ctx.context,
        device,
        props,
        &err
    );
    CHECK(err);
    //g_ctx.queue = clCreateCommandQueue(g_ctx.context, device, 0, &err);

    cl_program program = clCreateProgramWithSource(g_ctx.context, 1, &source, NULL, &err);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    g_ctx.kernel = clCreateKernel(program, "ntt", &err);

    g_ctx.buffer = clCreateBuffer(g_ctx.context, CL_MEM_READ_WRITE,
                                 sizeof(int16_t) * 256 * BATCH_SIZE, NULL, &err);
}


void opencl_cleanup() {
    clReleaseMemObject(g_ctx.buffer);
    clReleaseKernel(g_ctx.kernel);
    clReleaseCommandQueue(g_ctx.queue);
    clReleaseContext(g_ctx.context);
}

