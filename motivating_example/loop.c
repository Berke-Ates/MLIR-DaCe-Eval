#include <stdlib.h>

int main()
{
  int *A = (int *)malloc(2147483645 * sizeof(int));
  int *B = (int *)malloc(2147483645 * sizeof(int));

  for (int i = 0; i < 2147483645; ++i)
  {
    A[i] = 5;

    for (int j = 0; j < 2147483645; ++j)
    {
      B[j] = A[i];
    }
  }

  return B[0];
}
