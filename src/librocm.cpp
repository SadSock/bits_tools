#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

// union m32{
//     uint32_t u32;
//     int32_t  s32;
//     float    f32;
// };

// union m64 {
//   uint64_t   u64;
//   int64_t    s64;
//   double     f64;
//   struct {
//     m32 lo;
//     m32 hi;
//   };
//   void* ptr;
// };

__global__ void _v_fma_f32(uint32_t *r, uint32_t a, uint32_t b, uint32_t c) {

  int ret;
  asm volatile("v_fma_f32 %0, %1, %2, %3;"
               : "=v"(ret)
               : "v"(a), "v"(b), "v"(c));
  *r = ret;
}

extern "C" {

uint32_t v_fma_f32(uint32_t a, uint32_t b, uint32_t c) {

  int N = 1;
  uint32_t ret;
  uint32_t *h_R;
  h_R = (uint32_t *)malloc(N * sizeof(uint32_t));
  uint32_t *d_R;
  hipMalloc((void **)&d_R, N * sizeof(uint32_t));

  dim3 block(N);
  dim3 grid(1);
  _v_fma_f32<<<grid, block>>>(d_R, a, b, c);

  hipMemcpy(h_R, d_R, N * sizeof(int), hipMemcpyDeviceToHost);

  ret = *h_R;
  free(h_R);
  hipFree(d_R);
  return ret;
}
}

__global__ void _v_mul_lo_u32(uint32_t *r, uint32_t a, uint32_t b) {

  int ret;
  asm volatile("v_mul_lo_u32 %0, %1, %2;" : "=v"(ret) : "v"(a), "v"(b));
  *r = ret;
}

extern "C" {

uint32_t v_mul_lo_u32(uint32_t a, uint32_t b) {

  int N = 1;
  uint32_t ret;
  uint32_t *h_R;
  h_R = (uint32_t *)malloc(N * sizeof(uint32_t));
  uint32_t *d_R;
  hipMalloc((void **)&d_R, N * sizeof(uint32_t));

  dim3 block(N);
  dim3 grid(1);
  _v_mul_lo_u32<<<grid, block>>>(d_R, a, b);

  hipMemcpy(h_R, d_R, N * sizeof(int), hipMemcpyDeviceToHost);

  ret = *h_R;
  free(h_R);
  hipFree(d_R);

  return ret;
}
}

__global__ void _v_add_u32(uint32_t *r, uint32_t a, uint32_t b) {
  int ret;
  asm volatile("v_add_u32_e32 %0, %1, %2;" : "=v"(ret) : "v"(a), "v"(b));
  *r = ret;
}

extern "C" {

uint32_t v_add_u32(uint32_t a, uint32_t b) {
  int N = 1;
  uint32_t ret;
  uint32_t *h_R;
  h_R = (uint32_t *)malloc(N * sizeof(uint32_t));
  uint32_t *d_R;
  hipMalloc((void **)&d_R, N * sizeof(uint32_t));

  dim3 block(N);
  dim3 grid(1);
  _v_add_u32<<<grid, block>>>(d_R, a, b);
  hipMemcpy(h_R, d_R, N * sizeof(int), hipMemcpyDeviceToHost);
  ret = *h_R;
  free(h_R);
  hipFree(d_R);
  return ret;
}
}

__global__ void _v_add_i32(uint32_t *r, uint32_t a, uint32_t b) {
  int ret;
  asm volatile("v_add_i32 %0, %1, %2;" : "=v"(ret) : "v"(a), "v"(b));
  *r = ret;
}

extern "C" {

uint32_t v_add_i32(uint32_t a, uint32_t b) {
  int N = 1;
  uint32_t ret;
  uint32_t *h_R;
  h_R = (uint32_t *)malloc(N * sizeof(uint32_t));
  uint32_t *d_R;
  hipMalloc((void **)&d_R, N * sizeof(uint32_t));

  dim3 block(N);
  dim3 grid(1);
  _v_add_i32<<<grid, block>>>(d_R, a, b);
  hipMemcpy(h_R, d_R, N * sizeof(int), hipMemcpyDeviceToHost);
  ret = *h_R;
  free(h_R);
  hipFree(d_R);
  return ret;
}
}
