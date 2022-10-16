#include <vector>
void vec(std::vector<float> &a, std::vector<float> &b)
{
  for (int i = 0; i < a.size(); ++i)
  {
    std::vector<float> &tmp = a;
    b[i] = tmp[i];
  }
}
