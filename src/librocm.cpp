#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdint.h>


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



// 定义设备端加法Kernel
__global__ void _fma_f32(uint32_t *r, uint32_t a, uint32_t b, uint32_t c) {

  int ret;
  asm volatile("v_fma_f32 %0, %1, %2, %3;"
               : "=v"(ret)
               : "v"(a), "v"(b), "v"(c));
  *r = ret;
}


extern "C" {
  
uint32_t fma_f32(uint32_t a, uint32_t b, uint32_t c) {

  // 数组长度
  int N = 1;
  uint32_t ret;
  // 在Host端分配内存
  uint32_t *h_R;
  h_R = (uint32_t *)malloc(N * sizeof(uint32_t));
  // 在Device端分配内存
  uint32_t *d_R;
  hipMalloc((void **)&d_R, N * sizeof(uint32_t));

  // 配置和执行Kernel
  dim3 block(N);
  dim3 grid(1);
  _fma_f32<<<grid, block>>>(d_R, a, b, c);

  // 拷贝结果回Host端
  hipMemcpy(h_R, d_R, N * sizeof(int), hipMemcpyDeviceToHost);

  // 释放资源
  ret = *h_R;
  free(h_R);
  hipFree(d_R);

  return ret;
}

}