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





int main()
{

  m32 a,b,c,r;
  a.f32 = b.f32 = c.f32 = 1.0;
  r.u32 = fma_f32(a.u32, b.u32, c.u32);
  printf("%f\n", r.f32);
  return 0;
}

