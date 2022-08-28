// https://github.com/TheAlgorithms/C/blob/master/searching/floyd_cycle_detection_algorithm.c

#include <assert.h>   /// for assert
#include <inttypes.h> /// for uint32_t
#include <stdio.h>    /// for IO operations

/**
 * @brief The main function implements the search algorithm
 * @tparam T type of array
 * @param in_arr the input array
 * @param n size of the array
 * @returns the duplicate number
 */
uint32_t duplicateNumber(const uint32_t *in_arr, size_t n)
{
  if (n <= 1)
  { // to find duplicate in an array its size should be atleast 2
    return -1;
  }
  uint32_t tortoise = in_arr[0]; ///< variable tortoise is used for the longer
                                 ///< jumps in the array
  uint32_t hare = in_arr[0];     ///< variable hare is used for shorter jumps in the array
  do
  {                              // loop to enter the cycle
    tortoise = in_arr[tortoise]; // tortoise is moving by one step
    hare = in_arr[in_arr[hare]]; // hare is moving by two steps
  } while (tortoise != hare);
  tortoise = in_arr[0];
  while (tortoise != hare)
  { // loop to find the entry point of cycle
    tortoise = in_arr[tortoise];
    hare = in_arr[hare];
  }
  return tortoise;
}

/**
 * @brief Main function
 * @returns 0 on exit
 */
int main()
{
  uint32_t arr[] = {1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610};
  size_t n = sizeof(arr) / sizeof(int);

  return duplicateNumber(arr, n);
}
