#include <time.h>
#include <stdio.h>

void char_num(int dig[13], int chr[2], int *ch, const int length)
{
  int j;
  int tenl;
  int base;
  int save;

  base = 8;
  *ch = 0;
  tenl = 1;
  for (j = 0; j < length / 2; j++)
    tenl = tenl * base;

  chr[0] = 0;
  chr[1] = 0;

  save = 1;
  for (j = 0; j < length / 2; j++)
  {
    chr[0] += dig[j] * save;
    save *= base;
  }

  save = 1;
  for (j = length / 2; j < length; j++)
  {
    chr[1] += dig[j] * save;
    save *= base;
  }

  return;
}

int main()
{
  clock_t start = clock();

  int length = 13;
  int dig[13] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13};
  int chr[2];
  int ch;

  char_num(dig, chr, &ch, length);

  clock_t diff = clock() - start;
  printf("%lf\n", ((double)diff * 1000) / CLOCKS_PER_SEC);

  return chr[0] + chr[1];
}
