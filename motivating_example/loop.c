#include <stdlib.h>

char main()
{
  char *A = malloc(sizeof(char));
  char *B = malloc(sizeof(char));

  for (char i = 0; i < 10000000; ++i)
  {
    *A = 5;
    *B = i;
  }

  return *A;
}
