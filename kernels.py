
# B = alpha*A.*conj(A)
matrix_square_kernel = """
#include <pycuda-complex.hpp>
__global__ void MatrixSquareKernel(pycuda::complex<float> *A,
                                float *B,
                                float alpha,
                                int matSizeX,
                                int matSizeY)
{
  unsigned int idx = blockDim.x*blockIdx.x + threadIdx.x;
  unsigned int idy = blockDim.y*blockIdx.y + threadIdx.y;

  if (idx < matSizeX && idy < matSizeY) {
      unsigned int index = matSizeX*idy + idx;
      pycuda::complex<float> elem = A[index];
      B[index] = alpha*(elem.real()*elem.real()+elem.imag()*elem.imag());
  }
}
"""

# B = sum(A, 1), sum over each row
# B[i] = eps if B[i] == 0
matrix_sum_kernel = """
__global__ void MatrixSumKernel(float *A,
                                float *B,
                                int matSizeX,
                                int matSizeY,
                                float eps)
{
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if (idx < matSizeX) {
        float sum = 0.0;
        for (int j=0; j<matSizeY, j++) {
            sum += A[idx*matSizeY + j];
        }
        B[idx] = (sum > 0.0) ? sum : eps;
    }
}
"""



