#include <vector>

void CollectDomainNodesToElemNodes_Extern(std::vector<double> &m_x,
                                          std::vector<double> &m_y,
                                          std::vector<double> &m_z,
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
