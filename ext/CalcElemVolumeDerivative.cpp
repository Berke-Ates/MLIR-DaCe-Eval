#insert VoluDer.cpp

void CalcElemVolumeDerivative_Extern(double dvdx[8],
                                     double dvdy[8],
                                     double dvdz[8],
                                     const double x[8],
                                     const double y[8],
                                     const double z[8])
{
  VoluDer_Extern(x[1], x[2], x[3], x[4], x[5], x[7],
                 y[1], y[2], y[3], y[4], y[5], y[7],
                 z[1], z[2], z[3], z[4], z[5], z[7],
                 &dvdx[0], &dvdy[0], &dvdz[0]);
  VoluDer_Extern(x[0], x[1], x[2], x[7], x[4], x[6],
                 y[0], y[1], y[2], y[7], y[4], y[6],
                 z[0], z[1], z[2], z[7], z[4], z[6],
                 &dvdx[3], &dvdy[3], &dvdz[3]);
  VoluDer_Extern(x[3], x[0], x[1], x[6], x[7], x[5],
                 y[3], y[0], y[1], y[6], y[7], y[5],
                 z[3], z[0], z[1], z[6], z[7], z[5],
                 &dvdx[2], &dvdy[2], &dvdz[2]);
  VoluDer_Extern(x[2], x[3], x[0], x[5], x[6], x[4],
                 y[2], y[3], y[0], y[5], y[6], y[4],
                 z[2], z[3], z[0], z[5], z[6], z[4],
                 &dvdx[1], &dvdy[1], &dvdz[1]);
  VoluDer_Extern(x[7], x[6], x[5], x[0], x[3], x[1],
                 y[7], y[6], y[5], y[0], y[3], y[1],
                 z[7], z[6], z[5], z[0], z[3], z[1],
                 &dvdx[4], &dvdy[4], &dvdz[4]);
  VoluDer_Extern(x[4], x[7], x[6], x[1], x[0], x[2],
                 y[4], y[7], y[6], y[1], y[0], y[2],
                 z[4], z[7], z[6], z[1], z[0], z[2],
                 &dvdx[5], &dvdy[5], &dvdz[5]);
  VoluDer_Extern(x[5], x[4], x[7], x[2], x[1], x[3],
                 y[5], y[4], y[7], y[2], y[1], y[3],
                 z[5], z[4], z[7], z[2], z[1], z[3],
                 &dvdx[6], &dvdy[6], &dvdz[6]);
  VoluDer_Extern(x[6], x[5], x[4], x[3], x[2], x[0],
                 y[6], y[5], y[4], y[3], y[2], y[0],
                 z[6], z[5], z[4], z[3], z[2], z[0],
                 &dvdx[7], &dvdy[7], &dvdz[7]);
}
