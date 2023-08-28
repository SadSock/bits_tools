#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>


union m32{
    uint32_t u32;
    int32_t  s32;
    float    f32;
};

union m64 {
  uint64_t   u64;
  int64_t    s64;
  double     f64;
  struct {
    m32 lo;
    m32 hi;
  };
  void* ptr;
};



// 定义设备端加法Kernel
__global__ void AsmKernel(uint32_t *r, uint32_t a, uint32_t b, uint32_t c) {

  int ret;

  asm volatile("v_fma_f32 %0, %1, %2, %3;"
               : "=v"(ret)
               : "v"(a), "v"(b), "v"(c));

  *r = ret;
}

int main()
{
  // 数组长度
  int N = 64;
  
  // 在Host端分配内存
  uint32_t *h_R;
  h_R = (uint32_t*)malloc(N * sizeof(uint32_t));


  // 在Device端分配内存
  uint32_t *d_R; 
  hipMalloc((void**)&d_R, N * sizeof(uint32_t));


  m32 a,b,c,r;
  a.f32 = b.f32 = c.f32 = 1.0;


  // 配置和执行Kernel
  dim3 block(N);
  dim3 grid(1);
  AsmKernel<<<grid,block>>>(d_R, a.u32, b.u32, c.u32);

  // 拷贝结果回Host端
  hipMemcpy(h_R, d_R, N * sizeof(int), hipMemcpyDeviceToHost);

  r.u32 = *h_R;
    
  // 释放资源
  printf("%f\n", r.f32);
  // std::cout<<r.f32<<std::endl;
  free(h_R);
  hipFree(d_R);

  return 0;
}
