void CalcElemFBHourglassForce_Extern(double *xd, double *yd, double *zd,
                                     double hourgam[][4], double coefficient,
                                     double *hgfx, double *hgfy, double *hgfz)
{
  double hxx[4];
  for (signed int i = 0; i < 4; i++)
  {
    hxx[i] = hourgam[0][i] * xd[0] + hourgam[1][i] * xd[1] +
             hourgam[2][i] * xd[2] + hourgam[3][i] * xd[3] +
             hourgam[4][i] * xd[4] + hourgam[5][i] * xd[5] +
             hourgam[6][i] * xd[6] + hourgam[7][i] * xd[7];
  }
  for (signed int i = 0; i < 8; i++)
  {
    hgfx[i] = coefficient *
              (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
               hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
  }
  for (signed int i = 0; i < 4; i++)
  {
    hxx[i] = hourgam[0][i] * yd[0] + hourgam[1][i] * yd[1] +
             hourgam[2][i] * yd[2] + hourgam[3][i] * yd[3] +
             hourgam[4][i] * yd[4] + hourgam[5][i] * yd[5] +
             hourgam[6][i] * yd[6] + hourgam[7][i] * yd[7];
  }
  for (signed int i = 0; i < 8; i++)
  {
    hgfy[i] = coefficient *
              (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
               hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
  }
  for (signed int i = 0; i < 4; i++)
  {
    hxx[i] = hourgam[0][i] * zd[0] + hourgam[1][i] * zd[1] +
             hourgam[2][i] * zd[2] + hourgam[3][i] * zd[3] +
             hourgam[4][i] * zd[4] + hourgam[5][i] * zd[5] +
             hourgam[6][i] * zd[6] + hourgam[7][i] * zd[7];
  }
  for (signed int i = 0; i < 8; i++)
  {
    hgfz[i] = coefficient *
              (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
               hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
  }
}
