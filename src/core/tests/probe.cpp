#include <cuda_runtime.h>
#include <cstdio>
int main() {
  int n=-1; auto e = cudaGetDeviceCount(&n);
  std::printf("cudaGetDeviceCount -> %s (count=%d)\n", cudaGetErrorString(e), n);
  if (e!=cudaSuccess || n<1) return 1;
  std::printf("setDevice -> %s\n", cudaGetErrorString(cudaSetDevice(0)));
  std::printf("free(0)   -> %s\n", cudaGetErrorString(cudaFree(0)));
  return 0;
}

