#include <vector>
#include <stdlib.h>

#include "CollectDomainNodesToElemNodes.cpp"
#include "CalcElemShapeFunctionDerivatives.cpp"
#include "CalcElemNodeNormals.cpp"
#include "SumElemStressesToNodeForces.cpp"

template <typename T>
T *Allocate(size_t size)
{
  return static_cast<T *>(malloc(sizeof(T) * size));
}

template <typename T>
void Release(T **ptr)
{
  if (*ptr != NULL)
  {
    free(*ptr);
    *ptr = NULL;
  }
}

void IntegrateStressForElems_Extern(std::vector<double> &m_x,
                                    std::vector<double> &m_y,
                                    std::vector<double> &m_z,
                                    std::vector<double> &m_fx,
                                    std::vector<double> &m_fy,
                                    std::vector<double> &m_fz,
                                    std::vector<signed int> &m_nodelist,
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
