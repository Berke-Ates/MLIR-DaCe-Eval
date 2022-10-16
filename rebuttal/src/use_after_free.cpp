#include <stdlib.h>
int main()
{
  int *buffer = (int *)malloc(sizeof(int32_t));

  for (int i = 0; i < 10; ++i)
    buffer[0] = i;

  free(buffer);

  return buffer[0];
}
