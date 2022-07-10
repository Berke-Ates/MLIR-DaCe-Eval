void VoluDer_Extern(const double x0, const double x1, const double x2,
                    const double x3, const double x4, const double x5,
                    const double y0, const double y1, const double y2,
                    const double y3, const double y4, const double y5,
                    const double z0, const double z1, const double z2,
                    const double z3, const double z4, const double z5,
                    double *dvdx, double *dvdy, double *dvdz)
{
  const double twelfth = 1.0 / 12.0;

  *dvdx =
      (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
      (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
      (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);
  *dvdy =
      -(x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
      (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
      (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);

  *dvdz =
      -(y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
      (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
      (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);

  *dvdx *= twelfth;
  *dvdy *= twelfth;
  *dvdz *= twelfth;
}
