void SumElemFaceNormal_Extern(double *normalX0, double *normalY0, double *normalZ0,
                              double *normalX1, double *normalY1, double *normalZ1,
                              double *normalX2, double *normalY2, double *normalZ2,
                              double *normalX3, double *normalY3, double *normalZ3,
                              const double x0, const double y0, const double z0,
                              const double x1, const double y1, const double z1,
                              const double x2, const double y2, const double z2,
                              const double x3, const double y3, const double z3)
{
  double bisectX0 = double(0.5) * (x3 + x2 - x1 - x0);
  double bisectY0 = double(0.5) * (y3 + y2 - y1 - y0);
  double bisectZ0 = double(0.5) * (z3 + z2 - z1 - z0);
  double bisectX1 = double(0.5) * (x2 + x1 - x3 - x0);
  double bisectY1 = double(0.5) * (y2 + y1 - y3 - y0);
  double bisectZ1 = double(0.5) * (z2 + z1 - z3 - z0);
  double areaX = double(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
  double areaY = double(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
  double areaZ = double(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

  *normalX0 += areaX;
  *normalX1 += areaX;
  *normalX2 += areaX;
  *normalX3 += areaX;

  *normalY0 += areaY;
  *normalY1 += areaY;
  *normalY2 += areaY;
  *normalY3 += areaY;

  *normalZ0 += areaZ;
  *normalZ1 += areaZ;
  *normalZ2 += areaZ;
  *normalZ3 += areaZ;
}
