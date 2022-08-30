

#define _USE_MATH_DEFINES /**< required for MS Visual C */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef max
/** shorthand for maximum value */
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
/** shorthand for minimum value */
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

/**
 * Get minimum value and index of the value in a vector
 * \param[in] X vector to search
 * \param[in] N number of points in the vector
 * \param[out] val minimum value found
 * \param[out] idx index where minimum value was found
 */
void kohonen_get_min_1d(double const *X, int N, double *val, int *idx)
{
  val[0] = -1; // initial min value

  for (int i = 0; i < N; i++) // check each value
  {
    if (X[i] < val[0]) // if a lower value is found
    {                  // save the value and its index
      idx[0] = i;
      val[0] = X[i];
    }
  }
}

int main()
{
  int N = 200;
  int features = 30000;
  int num_out = 20000;

  double *x = (double *)malloc(features * sizeof(double));
  double *W = (double *)malloc(num_out * features * sizeof(double));

  for (int i = 0; i < num_out; i++) // loop till max(N, num_out)
  {
    for (int j = 0; j < features; j++)
    {
      W[i * features + j] = 0.5;
      x[j] = 0.5;
    }
  }

  int num_features = features;
  double alpha = 1.f;
  double *D = (double *)malloc(num_out * sizeof(double));
  int R = num_out >> 2;

  // https://github.com/TheAlgorithms/C/blob/master/machine_learning/kohonen_som_trace.c
  int j, k;

#ifdef _OPENMP
#pragma omp for
#endif
  // step 1: for each output point
  for (j = 0; j < num_out; j++)
  {
    D[j] = 0.f;
    // compute Euclidian distance of each output
    // point from the current sample
    for (k = 0; k < num_features; k++)
      D[j] += (W[j * num_features + k] - x[k]) * (W[j * num_features + k] - x[k]);
  }

  // step 2:  get closest node i.e., node with smallest Euclidian distance to
  // the current pattern
  int d_min_idx;
  double d_min;
  kohonen_get_min_1d(D, num_out, &d_min, &d_min_idx);

  // step 3a: get the neighborhood range
  int from_node = max(0, d_min_idx - R);
  int to_node = min(num_out, d_min_idx + R + 1);

  // step 3b: update the weights of nodes in the
  // neighborhood
#ifdef _OPENMP
#pragma omp for
#endif
  for (j = from_node; j < to_node; j++)
    for (k = 0; k < num_features; k++)
      // update weights of nodes in the neighborhood
      W[j * num_features + k] += alpha * (x[k] - W[j * num_features + k]);

  //----------------------------------------------------------------------------
  int res = (int)W[0];
  free(x);
  free(W);
  free(D);

  return res;
}
