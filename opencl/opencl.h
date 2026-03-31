#ifndef OPENCL_H
#define OPENCL_H
#define CL_TARGET_OPENCL_VERSION 200

#include <CL/cl.h>

typedef struct {
  cl_context context;
  cl_command_queue queue;
  cl_kernel kernel;
  cl_mem buffer;
  cl_event event;
} gpu_ctx;

extern gpu_ctx g_ctx;

void opencl_init(void);
void opencl_cleanup(void);

#endif