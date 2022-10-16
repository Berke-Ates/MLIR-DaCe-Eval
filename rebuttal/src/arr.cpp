#include <array>

void arr()
{
  std::array<float, 10> a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::array<float, 10> b = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  for (int i = 0; i < a.size(); ++i)
  {
    std::array<float, 10> tmp = a;
    b[i] = tmp[i];
  }
}
