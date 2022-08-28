void *cpy(void *dest, const void *src, int count)
{
  char *dst8 = (char *)dest;
  char *src8 = (char *)src;

  while (count--)
  {
    *dst8++ = *src8++;
  }
  return dest;
}

int main()
{
  const int M = 1024 * 1024;

  char a[M];
  char b[M];

  for (int i = 0; i < M; ++i)
    b[i] = 0;

  for (int i = 0; i < M; ++i)
    a[i] = i % 127;

  cpy(b, a, M);

  return b[6];
}
