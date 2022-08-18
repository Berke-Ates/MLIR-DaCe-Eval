#include <stdlib.h>
#include <math.h>

inline double CBRT(double arg) { return cbrt(arg); }

template <typename T>
static inline T *Allocate(size_t size)
{
  return static_cast<T *>(malloc(sizeof(T) * size));
}

template <typename T>
static inline void Release(T **ptr)
{
  // if (*ptr != NULL)
  // {
  free(*ptr);
  //   *ptr = NULL;
  // }
}

static inline void InitStressTermsForElems_Extern(double *m_p,
                                                  double *m_q,
                                                  double *sigxx, double *sigyy, double *sigzz,
                                                  signed int numElem)
{
  //
  // pull in the stresses appropriate to the hydro integration
  //

#pragma omp parallel for firstprivate(numElem)
  for (signed int i = 0; i < numElem; ++i)
  {
    sigxx[i] = sigyy[i] = sigzz[i] = -m_p[i] - m_q[i];
  }
}

static inline void CollectDomainNodesToElemNodes_Extern(double *m_x,
                                                        double *m_y,
                                                        double *m_z,
                                                        const signed int *elemToNode,
                                                        double elemX[8],
                                                        double elemY[8],
                                                        double elemZ[8])
{
  signed int nd0i = elemToNode[0];
  signed int nd1i = elemToNode[1];
  signed int nd2i = elemToNode[2];
  signed int nd3i = elemToNode[3];
  signed int nd4i = elemToNode[4];
  signed int nd5i = elemToNode[5];
  signed int nd6i = elemToNode[6];
  signed int nd7i = elemToNode[7];

  elemX[0] = m_x[nd0i];
  elemX[1] = m_x[nd1i];
  elemX[2] = m_x[nd2i];
  elemX[3] = m_x[nd3i];
  elemX[4] = m_x[nd4i];
  elemX[5] = m_x[nd5i];
  elemX[6] = m_x[nd6i];
  elemX[7] = m_x[nd7i];

  elemY[0] = m_y[nd0i];
  elemY[1] = m_y[nd1i];
  elemY[2] = m_y[nd2i];
  elemY[3] = m_y[nd3i];
  elemY[4] = m_y[nd4i];
  elemY[5] = m_y[nd5i];
  elemY[6] = m_y[nd6i];
  elemY[7] = m_y[nd7i];

  elemZ[0] = m_z[nd0i];
  elemZ[1] = m_z[nd1i];
  elemZ[2] = m_z[nd2i];
  elemZ[3] = m_z[nd3i];
  elemZ[4] = m_z[nd4i];
  elemZ[5] = m_z[nd5i];
  elemZ[6] = m_z[nd6i];
  elemZ[7] = m_z[nd7i];
}

static inline void CalcElemShapeFunctionDerivatives_Extern(double const x[],
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

static inline void SumElemFaceNormal_Extern(double *normalX0, double *normalY0, double *normalZ0,
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

static inline void CalcElemNodeNormals_Extern(double pfx[8],
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

static inline void SumElemStressesToNodeForces_Extern(const double B[][8],
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

static inline void IntegrateStressForElems_Extern(double *m_x,
                                                  double *m_y,
                                                  double *m_z,
                                                  double *m_fx,
                                                  double *m_fy,
                                                  double *m_fz,
                                                  signed int *m_nodelist,
                                                  signed int *m_nodeElemStart,
                                                  signed int *m_nodeElemCornerList,
                                                  double *sigxx, double *sigyy, double *sigzz,
                                                  double *determ, signed int numElem, signed int numNode)
{
#if _OPENMP
  signed int numthreads = omp_get_max_threads();
#else
  signed int numthreads = 1;
#endif

  signed int numElem8 = numElem * 8;
  double *fx_elem;
  double *fy_elem;
  double *fz_elem;
  double fx_local[8];
  double fy_local[8];
  double fz_local[8];

  if (numthreads > 1)
  {
    fx_elem = Allocate<double>(numElem8);
    fy_elem = Allocate<double>(numElem8);
    fz_elem = Allocate<double>(numElem8);
  }
  // loop over all elements

#pragma omp parallel for firstprivate(numElem)
  for (signed int k = 0; k < numElem; ++k)
  {
    const signed int *const elemToNode = &m_nodelist[/* signed int( */ 8 /* ) */ * k];
    double B[3][8]; // shape function derivatives
    double x_local[8];
    double y_local[8];
    double z_local[8];

    // get nodal coordinates from global arrays and copy into local arrays.
    CollectDomainNodesToElemNodes_Extern(m_x, m_y, m_z,
                                         elemToNode,
                                         x_local, y_local, z_local);

    // Volume calculation involves extra work for numerical consistency
    CalcElemShapeFunctionDerivatives_Extern(x_local, y_local, z_local,
                                            B, &determ[k]);

    CalcElemNodeNormals_Extern(B[0], B[1], B[2],
                               x_local, y_local, z_local);

    if (numthreads > 1)
    {
      // Eliminate thread writing conflicts at the nodes by giving
      // each element its own copy to write to
      SumElemStressesToNodeForces_Extern(B, sigxx[k], sigyy[k], sigzz[k],
                                         &fx_elem[k * 8],
                                         &fy_elem[k * 8],
                                         &fz_elem[k * 8]);
    }
    else
    {
      SumElemStressesToNodeForces_Extern(B, sigxx[k], sigyy[k], sigzz[k],
                                         fx_local, fy_local, fz_local);

      // copy nodal force contributions to global force arrray.
      for (signed int lnode = 0; lnode < 8; ++lnode)
      {
        signed int gnode = elemToNode[lnode];
        m_fx[gnode] += fx_local[lnode];
        m_fy[gnode] += fy_local[lnode];
        m_fz[gnode] += fz_local[lnode];
      }
    }
  }

  if (numthreads > 1)
  {
    // If threaded, then we need to copy the data out of the temporary
    // arrays used above into the final forces field
#pragma omp parallel for firstprivate(numNode)
    for (signed int gnode = 0; gnode < numNode; ++gnode)
    {
      signed int count = m_nodeElemStart[gnode + 1] - m_nodeElemStart[gnode];
      signed int *cornerList = &m_nodeElemCornerList[m_nodeElemStart[gnode]];
      double fx_tmp = double(0.0);
      double fy_tmp = double(0.0);
      double fz_tmp = double(0.0);
      for (signed int i = 0; i < count; ++i)
      {
        signed int ielem = cornerList[i];
        fx_tmp += fx_elem[ielem];
        fy_tmp += fy_elem[ielem];
        fz_tmp += fz_elem[ielem];
      }
      m_fx[gnode] = fx_tmp;
      m_fy[gnode] = fy_tmp;
      m_fz[gnode] = fz_tmp;
    }
    Release(&fz_elem);
    Release(&fy_elem);
    Release(&fx_elem);
  }
}

static inline void VoluDer_Extern(const double x0, const double x1, const double x2,
                                  const double x3, const double x4, const double x5,
                                  const double y0, const double y1, const double y2,
                                  const double y3, const double y4, const double y5,
                                  const double z0, const double z1, const double z2,
                                  const double z3, const double z4, const double z5,
                                  double *dvdx, double *dvdy, double *dvdz)
{
  const double twelfth = double(1.0) / double(12.0);

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

static inline void CalcElemVolumeDerivative_Extern(double dvdx[8],
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

static inline void CalcElemFBHourglassForce_Extern(double *xd, double *yd, double *zd,
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

static inline void CalcFBHourglassForceForElems_Extern(signed int *m_nodelist,
                                                       double *m_ss,
                                                       double *m_elemMass,
                                                       double *m_xd,
                                                       double *m_yd,
                                                       double *m_zd,
                                                       double *m_fx,
                                                       double *m_fy,
                                                       double *m_fz,
                                                       signed int *m_nodeElemStart,
                                                       signed int *m_nodeElemCornerList,
                                                       double *determ,
                                                       double *x8n, double *y8n, double *z8n,
                                                       double *dvdx, double *dvdy, double *dvdz,
                                                       double hourg, signed int numElem,
                                                       signed int numNode)
{

#if _OPENMP
  signed int numthreads = omp_get_max_threads();
#else
  signed int numthreads = 1;
#endif
  /*************************************************
   *
   *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
   *               force.
   *
   *************************************************/

  signed int numElem8 = numElem * 8;

  double *fx_elem;
  double *fy_elem;
  double *fz_elem;

  if (numthreads > 1)
  {
    fx_elem = Allocate<double>(numElem8);
    fy_elem = Allocate<double>(numElem8);
    fz_elem = Allocate<double>(numElem8);
  }

  double gamma[4][8];

  gamma[0][0] = double(1.);
  gamma[0][1] = double(1.);
  gamma[0][2] = double(-1.);
  gamma[0][3] = double(-1.);
  gamma[0][4] = double(-1.);
  gamma[0][5] = double(-1.);
  gamma[0][6] = double(1.);
  gamma[0][7] = double(1.);
  gamma[1][0] = double(1.);
  gamma[1][1] = double(-1.);
  gamma[1][2] = double(-1.);
  gamma[1][3] = double(1.);
  gamma[1][4] = double(-1.);
  gamma[1][5] = double(1.);
  gamma[1][6] = double(1.);
  gamma[1][7] = double(-1.);
  gamma[2][0] = double(1.);
  gamma[2][1] = double(-1.);
  gamma[2][2] = double(1.);
  gamma[2][3] = double(-1.);
  gamma[2][4] = double(1.);
  gamma[2][5] = double(-1.);
  gamma[2][6] = double(1.);
  gamma[2][7] = double(-1.);
  gamma[3][0] = double(-1.);
  gamma[3][1] = double(1.);
  gamma[3][2] = double(-1.);
  gamma[3][3] = double(1.);
  gamma[3][4] = double(1.);
  gamma[3][5] = double(-1.);
  gamma[3][6] = double(1.);
  gamma[3][7] = double(-1.);

  /*************************************************/
  /*    compute the hourglass modes */

#pragma omp parallel for firstprivate(numElem, hourg)
  for (signed int i2 = 0; i2 < numElem; ++i2)
  {
    double *fx_local, *fy_local, *fz_local;
    double hgfx[8], hgfy[8], hgfz[8];

    double coefficient;

    double hourgam[8][4];
    double xd1[8], yd1[8], zd1[8];

    const signed int *elemToNode = &m_nodelist[/* signed int( */ 8 /* ) */ * i2];
    signed int i3 = 8 * i2;
    double volinv = double(1.0) / determ[i2];
    double ss1, mass1, volume13;
    for (signed int i1 = 0; i1 < 4; ++i1)
    {

      double hourmodx =
          x8n[i3] * gamma[i1][0] + x8n[i3 + 1] * gamma[i1][1] +
          x8n[i3 + 2] * gamma[i1][2] + x8n[i3 + 3] * gamma[i1][3] +
          x8n[i3 + 4] * gamma[i1][4] + x8n[i3 + 5] * gamma[i1][5] +
          x8n[i3 + 6] * gamma[i1][6] + x8n[i3 + 7] * gamma[i1][7];

      double hourmody =
          y8n[i3] * gamma[i1][0] + y8n[i3 + 1] * gamma[i1][1] +
          y8n[i3 + 2] * gamma[i1][2] + y8n[i3 + 3] * gamma[i1][3] +
          y8n[i3 + 4] * gamma[i1][4] + y8n[i3 + 5] * gamma[i1][5] +
          y8n[i3 + 6] * gamma[i1][6] + y8n[i3 + 7] * gamma[i1][7];

      double hourmodz =
          z8n[i3] * gamma[i1][0] + z8n[i3 + 1] * gamma[i1][1] +
          z8n[i3 + 2] * gamma[i1][2] + z8n[i3 + 3] * gamma[i1][3] +
          z8n[i3 + 4] * gamma[i1][4] + z8n[i3 + 5] * gamma[i1][5] +
          z8n[i3 + 6] * gamma[i1][6] + z8n[i3 + 7] * gamma[i1][7];

      hourgam[0][i1] = gamma[i1][0] - volinv * (dvdx[i3] * hourmodx +
                                                dvdy[i3] * hourmody +
                                                dvdz[i3] * hourmodz);

      hourgam[1][i1] = gamma[i1][1] - volinv * (dvdx[i3 + 1] * hourmodx +
                                                dvdy[i3 + 1] * hourmody +
                                                dvdz[i3 + 1] * hourmodz);

      hourgam[2][i1] = gamma[i1][2] - volinv * (dvdx[i3 + 2] * hourmodx +
                                                dvdy[i3 + 2] * hourmody +
                                                dvdz[i3 + 2] * hourmodz);

      hourgam[3][i1] = gamma[i1][3] - volinv * (dvdx[i3 + 3] * hourmodx +
                                                dvdy[i3 + 3] * hourmody +
                                                dvdz[i3 + 3] * hourmodz);

      hourgam[4][i1] = gamma[i1][4] - volinv * (dvdx[i3 + 4] * hourmodx +
                                                dvdy[i3 + 4] * hourmody +
                                                dvdz[i3 + 4] * hourmodz);

      hourgam[5][i1] = gamma[i1][5] - volinv * (dvdx[i3 + 5] * hourmodx +
                                                dvdy[i3 + 5] * hourmody +
                                                dvdz[i3 + 5] * hourmodz);

      hourgam[6][i1] = gamma[i1][6] - volinv * (dvdx[i3 + 6] * hourmodx +
                                                dvdy[i3 + 6] * hourmody +
                                                dvdz[i3 + 6] * hourmodz);

      hourgam[7][i1] = gamma[i1][7] - volinv * (dvdx[i3 + 7] * hourmodx +
                                                dvdy[i3 + 7] * hourmody +
                                                dvdz[i3 + 7] * hourmodz);
    }

    /* compute forces */
    /* store forces into h arrays (force arrays) */

    ss1 = m_ss[i2];
    mass1 = m_elemMass[i2];
    volume13 = CBRT(determ[i2]);

    signed int n0si2 = elemToNode[0];
    signed int n1si2 = elemToNode[1];
    signed int n2si2 = elemToNode[2];
    signed int n3si2 = elemToNode[3];
    signed int n4si2 = elemToNode[4];
    signed int n5si2 = elemToNode[5];
    signed int n6si2 = elemToNode[6];
    signed int n7si2 = elemToNode[7];

    xd1[0] = m_xd[n0si2];
    xd1[1] = m_xd[n1si2];
    xd1[2] = m_xd[n2si2];
    xd1[3] = m_xd[n3si2];
    xd1[4] = m_xd[n4si2];
    xd1[5] = m_xd[n5si2];
    xd1[6] = m_xd[n6si2];
    xd1[7] = m_xd[n7si2];

    yd1[0] = m_yd[n0si2];
    yd1[1] = m_yd[n1si2];
    yd1[2] = m_yd[n2si2];
    yd1[3] = m_yd[n3si2];
    yd1[4] = m_yd[n4si2];
    yd1[5] = m_yd[n5si2];
    yd1[6] = m_yd[n6si2];
    yd1[7] = m_yd[n7si2];

    zd1[0] = m_zd[n0si2];
    zd1[1] = m_zd[n1si2];
    zd1[2] = m_zd[n2si2];
    zd1[3] = m_zd[n3si2];
    zd1[4] = m_zd[n4si2];
    zd1[5] = m_zd[n5si2];
    zd1[6] = m_zd[n6si2];
    zd1[7] = m_zd[n7si2];

    coefficient = -hourg * double(0.01) * ss1 * mass1 / volume13;

    CalcElemFBHourglassForce_Extern(xd1, yd1, zd1,
                                    hourgam,
                                    coefficient, hgfx, hgfy, hgfz);

    // With the threaded version, we write into local arrays per elem
    // so we don't have to worry about race conditions
    if (numthreads > 1)
    {
      fx_local = &fx_elem[i3];
      fx_local[0] = hgfx[0];
      fx_local[1] = hgfx[1];
      fx_local[2] = hgfx[2];
      fx_local[3] = hgfx[3];
      fx_local[4] = hgfx[4];
      fx_local[5] = hgfx[5];
      fx_local[6] = hgfx[6];
      fx_local[7] = hgfx[7];

      fy_local = &fy_elem[i3];
      fy_local[0] = hgfy[0];
      fy_local[1] = hgfy[1];
      fy_local[2] = hgfy[2];
      fy_local[3] = hgfy[3];
      fy_local[4] = hgfy[4];
      fy_local[5] = hgfy[5];
      fy_local[6] = hgfy[6];
      fy_local[7] = hgfy[7];

      fz_local = &fz_elem[i3];
      fz_local[0] = hgfz[0];
      fz_local[1] = hgfz[1];
      fz_local[2] = hgfz[2];
      fz_local[3] = hgfz[3];
      fz_local[4] = hgfz[4];
      fz_local[5] = hgfz[5];
      fz_local[6] = hgfz[6];
      fz_local[7] = hgfz[7];
    }
    else
    {
      m_fx[n0si2] += hgfx[0];
      m_fy[n0si2] += hgfy[0];
      m_fz[n0si2] += hgfz[0];

      m_fx[n1si2] += hgfx[1];
      m_fy[n1si2] += hgfy[1];
      m_fz[n1si2] += hgfz[1];

      m_fx[n2si2] += hgfx[2];
      m_fy[n2si2] += hgfy[2];
      m_fz[n2si2] += hgfz[2];

      m_fx[n3si2] += hgfx[3];
      m_fy[n3si2] += hgfy[3];
      m_fz[n3si2] += hgfz[3];

      m_fx[n4si2] += hgfx[4];
      m_fy[n4si2] += hgfy[4];
      m_fz[n4si2] += hgfz[4];

      m_fx[n5si2] += hgfx[5];
      m_fy[n5si2] += hgfy[5];
      m_fz[n5si2] += hgfz[5];

      m_fx[n6si2] += hgfx[6];
      m_fy[n6si2] += hgfy[6];
      m_fz[n6si2] += hgfz[6];

      m_fx[n7si2] += hgfx[7];
      m_fy[n7si2] += hgfy[7];
      m_fz[n7si2] += hgfz[7];
    }
  }

  if (numthreads > 1)
  {
    // Collect the data from the local arrays into the final force arrays
#pragma omp parallel for firstprivate(numNode)
    for (signed int gnode = 0; gnode < numNode; ++gnode)
    {
      signed int count = m_nodeElemStart[gnode + 1] - m_nodeElemStart[gnode];
      signed int *cornerList = &m_nodeElemCornerList[m_nodeElemStart[gnode]];
      double fx_tmp = double(0.0);
      double fy_tmp = double(0.0);
      double fz_tmp = double(0.0);
      for (signed int i = 0; i < count; ++i)
      {
        signed int ielem = cornerList[i];
        fx_tmp += fx_elem[ielem];
        fy_tmp += fy_elem[ielem];
        fz_tmp += fz_elem[ielem];
      }
      m_fx[gnode] += fx_tmp;
      m_fy[gnode] += fy_tmp;
      m_fz[gnode] += fz_tmp;
    }
    Release(&fz_elem);
    Release(&fy_elem);
    Release(&fx_elem);
  }
}

static inline void CalcHourglassControlForElems_Extern(double *m_x,
                                                       double *m_y,
                                                       double *m_z,
                                                       signed int m_numElem,
                                                       double *m_volo,
                                                       double *m_v,
                                                       signed int *m_nodelist,
                                                       double *m_ss,
                                                       double *m_elemMass,
                                                       double *m_xd,
                                                       double *m_yd,
                                                       double *m_zd,
                                                       double *m_fx,
                                                       double *m_fy,
                                                       double *m_fz,
                                                       signed int *m_nodeElemStart,
                                                       signed int *m_nodeElemCornerList,
                                                       signed int m_numNode,
                                                       double determ[], double hgcoef)
{
  signed int numElem = m_numElem;
  signed int numElem8 = numElem * 8;
  double *dvdx = Allocate<double>(numElem8);
  double *dvdy = Allocate<double>(numElem8);
  double *dvdz = Allocate<double>(numElem8);
  double *x8n = Allocate<double>(numElem8);
  double *y8n = Allocate<double>(numElem8);
  double *z8n = Allocate<double>(numElem8);

  /* start loop over elements */
#pragma omp parallel for firstprivate(numElem)
  for (signed int i = 0; i < numElem; ++i)
  {
    double x1[8], y1[8], z1[8];
    double pfx[8], pfy[8], pfz[8];

    signed int *elemToNode = &m_nodelist[/* signed int( */ 8 /* ) */ * i];
    CollectDomainNodesToElemNodes_Extern(m_x, m_y, m_z, elemToNode, x1, y1, z1);

    CalcElemVolumeDerivative_Extern(pfx, pfy, pfz, x1, y1, z1);

    /* load into temporary storage for FB Hour Glass control */
    for (signed int ii = 0; ii < 8; ++ii)
    {
      signed int jj = 8 * i + ii;

      dvdx[jj] = pfx[ii];
      dvdy[jj] = pfy[ii];
      dvdz[jj] = pfz[ii];
      x8n[jj] = x1[ii];
      y8n[jj] = y1[ii];
      z8n[jj] = z1[ii];
    }

    determ[i] = m_volo[i] * m_v[i];

    /* Do a check for negative volumes */
    if (m_v[i] <= double(0.0))
    {
#if USE_MPI
      MPI_Abort(MPI_COMM_WORLD, -1);
#else
      exit(-1);
#endif
    }
  }

  if (hgcoef > double(0.))
  {
    CalcFBHourglassForceForElems_Extern(m_nodelist,
                                        m_ss,
                                        m_elemMass,
                                        m_xd,
                                        m_yd,
                                        m_zd,
                                        m_fx,
                                        m_fy,
                                        m_fz,
                                        m_nodeElemStart,
                                        m_nodeElemCornerList,
                                        determ,
                                        x8n, y8n, z8n, dvdx, dvdy, dvdz,
                                        hgcoef, numElem, m_numNode);
  }

  Release(&z8n);
  Release(&y8n);
  Release(&x8n);
  Release(&dvdz);
  Release(&dvdy);
  Release(&dvdx);

  return;
}

void CalcVolumeForceForElems_Extern(double *m_p,
                                    double *m_q,
                                    double *m_x,
                                    double *m_y,
                                    double *m_z,
                                    double *m_fx,
                                    double *m_fy,
                                    double *m_fz,
                                    signed int *m_nodelist,
                                    signed int *m_nodeElemStart,
                                    signed int *m_nodeElemCornerList,
                                    signed int m_numElem,
                                    double *m_volo,
                                    double *m_v,
                                    double *m_ss,
                                    double *m_elemMass,
                                    double *m_xd,
                                    double *m_yd,
                                    double *m_zd,
                                    signed int m_numNode,
                                    double m_hgcoef)
{
  signed int numElem = m_numElem;
  if (numElem != 0)
  {
    double hgcoef = m_hgcoef;
    double *sigxx = Allocate<double>(numElem);
    double *sigyy = Allocate<double>(numElem);
    double *sigzz = Allocate<double>(numElem);
    double *determ = Allocate<double>(numElem);

    /* Sum contributions to total stress tensor */
    InitStressTermsForElems_Extern(m_p, m_q, sigxx, sigyy, sigzz, numElem);

    // call elemlib stress integration loop to produce nodal forces from
    // material stresses.
    IntegrateStressForElems_Extern(m_x,
                                   m_y,
                                   m_z,
                                   m_fx,
                                   m_fy,
                                   m_fz,
                                   m_nodelist,
                                   m_nodeElemStart,
                                   m_nodeElemCornerList,
                                   sigxx, sigyy, sigzz, determ, numElem,
                                   m_numNode);

    // check for negative element volume
#pragma omp parallel for firstprivate(numElem)
    for (signed int k = 0; k < numElem; ++k)
    {
      if (determ[k] <= double(0.0))
      {
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#else
        exit(-1);
#endif
      }
    }

    CalcHourglassControlForElems_Extern(m_x,
                                        m_y,
                                        m_z,
                                        m_numElem,
                                        m_volo,
                                        m_v,
                                        m_nodelist,
                                        m_ss,
                                        m_elemMass,
                                        m_xd,
                                        m_yd,
                                        m_zd,
                                        m_fx,
                                        m_fy,
                                        m_fz,
                                        m_nodeElemStart,
                                        m_nodeElemCornerList,
                                        m_numNode,
                                        determ, hgcoef);

    Release(&determ);
    Release(&sigzz);
    Release(&sigyy);
    Release(&sigxx);
  }
}
