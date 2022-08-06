#include <stdlib.h>

int main()
{
  int *A = (int *)malloc(2147483647 * sizeof(int)); // ~ 8GB of memory
  int *B = (int *)malloc(21 * sizeof(int));

  for (int i = 0; i < 2147483647; ++i)
  {
    A[i] = 5;

    for (int j = 0; j < 21; ++j)
    {
      B[j] = A[i];
      A[j] = A[i];
    }
  }

  return B[0];
}
