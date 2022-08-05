#ifndef EXTERNAL_ALLOCATE
#define EXTERNAL_ALLOCATE

#include <stdlib.h>

template <typename T>
T *Allocate(size_t size)
{
  return static_cast<T *>(malloc(sizeof(T) * size));
}

template <typename T>
void Release(T **ptr)
{
  // if (*ptr != NULL)
  // {
  free(*ptr);
  //   *ptr = NULL;
  // }
}

#endif
