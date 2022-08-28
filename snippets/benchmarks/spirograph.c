#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

// https://github.com/TheAlgorithms/C/blob/master/graphics/spirograph.c
void spirograph(double *x, double *y, double l, double k, size_t N, double rot)
{
  double dt = rot * 2.f * M_PI / N;
  double t = 0.f, R = 1.f;
  const double k1 = 1.f - k;

  for (size_t dk = 0; dk < N; dk++, t += dt)
  {
    x[dk] = R * (k1 * cos(t) + l * k * cos(k1 * t / k));
    y[dk] = R * (k1 * sin(t) - l * k * sin(k1 * t / k));
  }
}

/**
 * @brief Test function to save resulting points to a CSV file.
 *
 */
static inline int test(void)
{
  size_t N = 10000000;
  double l = 0.3, k = 0.75, rot = 10.;

  double *x = (double *)malloc(N * sizeof(double));
  double *y = (double *)malloc(N * sizeof(double));

  spirograph(x, y, l, k, N, rot);

  int a = (int)(x[10] * 10 + y[10] * 10);

  free(x);
  free(y);

  return a;
}

/*********************************** DRIVER ***********************************/

int main()
{
  return test();
}
