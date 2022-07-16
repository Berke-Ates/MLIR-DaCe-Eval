#include <vector>
#include <math.h>

#include "CalcElemFBHourglassForce.cpp"
#include "Alloc.cpp"

// Probably only need one of those
inline float CBRT(float arg) { return cbrtf(arg); }
inline double CBRT(double arg) { return cbrt(arg); }
inline long double CBRT(long double arg) { return cbrtl(arg); }

void CalcFBHourglassForceForElems_Extern(std::vector<signed int> m_nodelist,
                                         std::vector<double> m_ss,
                                         std::vector<double> m_elemMass,
                                         std::vector<double> m_xd,
                                         std::vector<double> m_yd,
                                         std::vector<double> m_zd,
                                         std::vector<double> m_fx,
                                         std::vector<double> m_fy,
                                         std::vector<double> m_fz,
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
