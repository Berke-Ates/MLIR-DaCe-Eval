#include <stdlib.h>

int main()
{
  int *A = (int *)malloc(1000000000 * sizeof(int));
  int *B = (int *)malloc(1000000000 * sizeof(int));

  for (int i = 0; i < 1000000000; ++i)
  {
    A[i] = 5;

    for (int j = 0; j < 1000000000; ++j)
    {
      B[j] = A[i];
    }

    for (int j = 0; j < 1000; ++j)
    {
      A[j] = A[i];
    }
  }

  return B[0];
}
