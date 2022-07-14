void CalcElemShapeFunctionDerivatives_Extern(double const x[],
                                             double const y[],
                                             double const z[],
                                             double b[][8],
                                             double *const volume)
{
  const double x0 = x[0];
  const double x1 = x[1];
  const double x2 = x[2];
  const double x3 = x[3];
  const double x4 = x[4];
  const double x5 = x[5];
  const double x6 = x[6];
  const double x7 = x[7];

  const double y0 = y[0];
  const double y1 = y[1];
  const double y2 = y[2];
  const double y3 = y[3];
  const double y4 = y[4];
  const double y5 = y[5];
  const double y6 = y[6];
  const double y7 = y[7];

  const double z0 = z[0];
  const double z1 = z[1];
  const double z2 = z[2];
  const double z3 = z[3];
  const double z4 = z[4];
  const double z5 = z[5];
  const double z6 = z[6];
  const double z7 = z[7];

  double fjxxi, fjxet, fjxze;
  double fjyxi, fjyet, fjyze;
  double fjzxi, fjzet, fjzze;
  double cjxxi, cjxet, cjxze;
  double cjyxi, cjyet, cjyze;
  double cjzxi, cjzet, cjzze;

  fjxxi = double(.125) * ((x6 - x0) + (x5 - x3) - (x7 - x1) - (x4 - x2));
  fjxet = double(.125) * ((x6 - x0) - (x5 - x3) + (x7 - x1) - (x4 - x2));
  fjxze = double(.125) * ((x6 - x0) + (x5 - x3) + (x7 - x1) + (x4 - x2));

  fjyxi = double(.125) * ((y6 - y0) + (y5 - y3) - (y7 - y1) - (y4 - y2));
  fjyet = double(.125) * ((y6 - y0) - (y5 - y3) + (y7 - y1) - (y4 - y2));
  fjyze = double(.125) * ((y6 - y0) + (y5 - y3) + (y7 - y1) + (y4 - y2));

  fjzxi = double(.125) * ((z6 - z0) + (z5 - z3) - (z7 - z1) - (z4 - z2));
  fjzet = double(.125) * ((z6 - z0) - (z5 - z3) + (z7 - z1) - (z4 - z2));
  fjzze = double(.125) * ((z6 - z0) + (z5 - z3) + (z7 - z1) + (z4 - z2));

  /* compute cofactors */
  cjxxi = (fjyet * fjzze) - (fjzet * fjyze);
  cjxet = -(fjyxi * fjzze) + (fjzxi * fjyze);
  cjxze = (fjyxi * fjzet) - (fjzxi * fjyet);

  cjyxi = -(fjxet * fjzze) + (fjzet * fjxze);
  cjyet = (fjxxi * fjzze) - (fjzxi * fjxze);
  cjyze = -(fjxxi * fjzet) + (fjzxi * fjxet);

  cjzxi = (fjxet * fjyze) - (fjyet * fjxze);
  cjzet = -(fjxxi * fjyze) + (fjyxi * fjxze);
  cjzze = (fjxxi * fjyet) - (fjyxi * fjxet);

  /* calculate partials :
     this need only be done for l = 0,1,2,3   since , by symmetry ,
     (6,7,4,5) = - (0,1,2,3) .
  */
  b[0][0] = -cjxxi - cjxet - cjxze;
  b[0][1] = cjxxi - cjxet - cjxze;
  b[0][2] = cjxxi + cjxet - cjxze;
  b[0][3] = -cjxxi + cjxet - cjxze;
  b[0][4] = -b[0][2];
  b[0][5] = -b[0][3];
  b[0][6] = -b[0][0];
  b[0][7] = -b[0][1];

  b[1][0] = -cjyxi - cjyet - cjyze;
  b[1][1] = cjyxi - cjyet - cjyze;
  b[1][2] = cjyxi + cjyet - cjyze;
  b[1][3] = -cjyxi + cjyet - cjyze;
  b[1][4] = -b[1][2];
  b[1][5] = -b[1][3];
  b[1][6] = -b[1][0];
  b[1][7] = -b[1][1];

  b[2][0] = -cjzxi - cjzet - cjzze;
  b[2][1] = cjzxi - cjzet - cjzze;
  b[2][2] = cjzxi + cjzet - cjzze;
  b[2][3] = -cjzxi + cjzet - cjzze;
  b[2][4] = -b[2][2];
  b[2][5] = -b[2][3];
  b[2][6] = -b[2][0];
  b[2][7] = -b[2][1];

  /* calculate jacobian determinant (volume) */
  *volume = double(8.) * (fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}
