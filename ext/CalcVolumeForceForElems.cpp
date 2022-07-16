#include <vector>

#include "InitStressTermsForElems.cpp"
#include "IntegrateStressForElems.cpp"
#include "CalcHourglassControlForElems.cpp"
#include "Alloc.cpp"

void CalcVolumeForceForElems_Extern(std::vector<double> &m_p,
                                    std::vector<double> &m_q,
                                    std::vector<double> &m_x,
                                    std::vector<double> &m_y,
                                    std::vector<double> &m_z,
                                    std::vector<double> &m_fx,
                                    std::vector<double> &m_fy,
                                    std::vector<double> &m_fz,
                                    std::vector<signed int> &m_nodelist,
                                    signed int *m_nodeElemStart,
                                    signed int *m_nodeElemCornerList,
                                    signed int m_numElem,
                                    std::vector<double> &m_volo,
                                    std::vector<double> &m_v,
                                    std::vector<double> &m_ss,
                                    std::vector<double> &m_elemMass,
                                    std::vector<double> &m_xd,
                                    std::vector<double> &m_yd,
                                    std::vector<double> &m_zd,
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
