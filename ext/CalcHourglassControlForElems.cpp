#include "CollectDomainNodesToElemNodes.cpp"
#include "CalcElemVolumeDerivative.cpp"
#include "CalcFBHourglassForceForElems.cpp"

#include "Alloc.cpp"

void CalcHourglassControlForElems_Extern(std::vector<double> &m_x,
                                         std::vector<double> &m_y,
                                         std::vector<double> &m_z,
                                         signed int m_numElem,
                                         std::vector<double> &m_volo,
                                         std::vector<double> &m_v,
                                         std::vector<signed int> &m_nodelist,
                                         std::vector<double> &m_ss,
                                         std::vector<double> &m_elemMass,
                                         std::vector<double> &m_xd,
                                         std::vector<double> &m_yd,
                                         std::vector<double> &m_zd,
                                         std::vector<double> &m_fx,
                                         std::vector<double> &m_fy,
                                         std::vector<double> &m_fz,
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
