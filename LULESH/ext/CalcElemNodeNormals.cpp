#include "SumElemFaceNormal.cpp"

void CalcElemNodeNormals_Extern(double pfx[8],
                                double pfy[8],
                                double pfz[8],
                                const double x[8],
                                const double y[8],
                                const double z[8])
{
  for (signed int i = 0; i < 8; ++i)
  {
    pfx[i] = double(0.0);
    pfy[i] = double(0.0);
    pfz[i] = double(0.0);
  }
  /* evaluate face one: nodes 0, 1, 2, 3 */
  SumElemFaceNormal_Extern(&pfx[0], &pfy[0], &pfz[0],
                           &pfx[1], &pfy[1], &pfz[1],
                           &pfx[2], &pfy[2], &pfz[2],
                           &pfx[3], &pfy[3], &pfz[3],
                           x[0], y[0], z[0], x[1], y[1], z[1],
                           x[2], y[2], z[2], x[3], y[3], z[3]);
  /* evaluate face two: nodes 0, 4, 5, 1 */
  SumElemFaceNormal_Extern(&pfx[0], &pfy[0], &pfz[0],
                           &pfx[4], &pfy[4], &pfz[4],
                           &pfx[5], &pfy[5], &pfz[5],
                           &pfx[1], &pfy[1], &pfz[1],
                           x[0], y[0], z[0], x[4], y[4], z[4],
                           x[5], y[5], z[5], x[1], y[1], z[1]);
  /* evaluate face three: nodes 1, 5, 6, 2 */
  SumElemFaceNormal_Extern(&pfx[1], &pfy[1], &pfz[1],
                           &pfx[5], &pfy[5], &pfz[5],
                           &pfx[6], &pfy[6], &pfz[6],
                           &pfx[2], &pfy[2], &pfz[2],
                           x[1], y[1], z[1], x[5], y[5], z[5],
                           x[6], y[6], z[6], x[2], y[2], z[2]);
  /* evaluate face four: nodes 2, 6, 7, 3 */
  SumElemFaceNormal_Extern(&pfx[2], &pfy[2], &pfz[2],
                           &pfx[6], &pfy[6], &pfz[6],
                           &pfx[7], &pfy[7], &pfz[7],
                           &pfx[3], &pfy[3], &pfz[3],
                           x[2], y[2], z[2], x[6], y[6], z[6],
                           x[7], y[7], z[7], x[3], y[3], z[3]);
  /* evaluate face five: nodes 3, 7, 4, 0 */
  SumElemFaceNormal_Extern(&pfx[3], &pfy[3], &pfz[3],
                           &pfx[7], &pfy[7], &pfz[7],
                           &pfx[4], &pfy[4], &pfz[4],
                           &pfx[0], &pfy[0], &pfz[0],
                           x[3], y[3], z[3], x[7], y[7], z[7],
                           x[4], y[4], z[4], x[0], y[0], z[0]);
  /* evaluate face six: nodes 4, 7, 6, 5 */
  SumElemFaceNormal_Extern(&pfx[4], &pfy[4], &pfz[4],
                           &pfx[7], &pfy[7], &pfz[7],
                           &pfx[6], &pfy[6], &pfz[6],
                           &pfx[5], &pfy[5], &pfz[5],
                           x[4], y[4], z[4], x[7], y[7], z[7],
                           x[6], y[6], z[6], x[5], y[5], z[5]);
}
