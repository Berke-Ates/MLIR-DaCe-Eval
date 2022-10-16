#include <vector>
#include <stdio.h>
#include <stdlib.h>

struct mystruct
{
  int a[5];
  int b[5];
  int c[5];
};

int strct(mystruct s)
{
  return s.b[0];
}
