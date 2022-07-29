#include <stdlib.h>

int main()
{
  int *A = (int *)malloc(2147483647 * sizeof(int));
  int *B = (int *)malloc(2147483647 * sizeof(int));
  int *C = (int *)malloc(2147483647 * sizeof(int));

  for (int i1 = 0; i1 < 2147483647; ++i1)
  {
    A[i1] += 5;
  }

  for (int i2 = 0; i2 < A[0]; ++i2)
  {
    B[0] = 5;
  }

  return B[0];
}
