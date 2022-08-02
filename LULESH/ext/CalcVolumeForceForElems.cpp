#include "InitStressTermsForElems.cpp"
#include "IntegrateStressForElems.cpp"
#include "CalcHourglassControlForElems.cpp"
#include "Alloc.cpp"

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
