void InitStressTermsForElems_Extern(double *m_p,
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
