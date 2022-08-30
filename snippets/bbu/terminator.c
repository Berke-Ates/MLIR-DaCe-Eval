#include <stdlib.h>

#define epsilon 9999

int main(void)
{
  int set1[10] = {1, 1, 1, 2, 3, epsilon, epsilon, epsilon, epsilon, epsilon};
  int set2[10] = {2, 2, 2, 1, 1, 1, 1, 1, 1, 1};

  int m = sizeof(set1) / sizeof(int);
  int n = sizeof(set2) / sizeof(int);
  int size = m;

  // https://github.com/momalab/TERMinatorSuite/blob/master/Kernels/PSI/setIntersection.c
  int *res = calloc(size, sizeof(int));
  int *exist = calloc(size, sizeof(int));

  for (int i = 0; i < size; i++)
  {
    for (int j = 0; j < size; j++)
    {
      int eq = (set1[i] == set2[j]) ? 1 : 0;
      exist[i] += eq;
      res[i] += eq * set1[i];
      set1[i] += eq * epsilon;
      set2[j] += eq * epsilon;
    }
  }
  for (int i = 0; i < size; i++)
  {
    res[i] = (1 - exist[i]) * epsilon + exist[i] * res[i];
  }

  //
  return res[0];
}
