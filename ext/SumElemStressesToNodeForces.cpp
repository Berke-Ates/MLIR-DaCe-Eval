void SumElemStressesToNodeForces(const double B[][8],
                                 const double stress_xx,
                                 const double stress_yy,
                                 const double stress_zz,
                                 double fx[], double fy[], double fz[])
{
  for (signed int i = 0; i < 8; i++)
  {
    fx[i] = -(stress_xx * B[0][i]);
    fy[i] = -(stress_yy * B[1][i]);
    fz[i] = -(stress_zz * B[2][i]);
  }
}
