#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <vector>

#if !defined(USE_MPI)
#error "You should specify USE_MPI=0 or USE_MPI=1 on the compile line"
#endif

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// Precision specification
typedef float real4;
typedef double real8;
typedef long double real10; // 10 bytes on x86

typedef int32_t Int4_t;
typedef int64_t Int8_t;
typedef Int4_t Index_t; // array subscript and loop index
typedef real8 Real_t;   // floating point representation
typedef Int4_t Int_t;   // integer representation

enum
{
  VolumeError = -1,
  QStopError = -2
};

inline real4 SQRT(real4 arg) { return sqrtf(arg); }
inline real8 SQRT(real8 arg) { return sqrt(arg); }
inline real10 SQRT(real10 arg) { return sqrtl(arg); }

inline real4 CBRT(real4 arg) { return cbrtf(arg); }
inline real8 CBRT(real8 arg) { return cbrt(arg); }
inline real10 CBRT(real10 arg) { return cbrtl(arg); }

inline real4 FABS(real4 arg) { return fabsf(arg); }
inline real8 FABS(real8 arg) { return fabs(arg); }
inline real10 FABS(real10 arg) { return fabsl(arg); }

/* might want to add access methods so that memory can be */
/* better managed, as in luleshFT */

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

class Domain
{

public:
  // Constructor
  Domain(Int_t numRanks, Index_t colLoc,
         Index_t rowLoc, Index_t planeLoc,
         Index_t nx, Int_t tp, Int_t nr, Int_t balance, Int_t cost);

  // Destructor
  ~Domain();

  //
  // ALLOCATION
  //

  void AllocateNodePersistent(Int_t numNode) // Node-centered
  {
    m_x.resize(numNode); // coordinates
    m_y.resize(numNode);
    m_z.resize(numNode);

    m_xd.resize(numNode); // velocities
    m_yd.resize(numNode);
    m_zd.resize(numNode);

    m_xdd.resize(numNode); // accelerations
    m_ydd.resize(numNode);
    m_zdd.resize(numNode);

    m_fx.resize(numNode); // forces
    m_fy.resize(numNode);
    m_fz.resize(numNode);

    m_nodalMass.resize(numNode); // mass
  }

  void AllocateElemPersistent(Int_t numElem) // Elem-centered
  {
    m_nodelist.resize(8 * numElem);

    // elem connectivities through face
    m_lxim.resize(numElem);
    m_lxip.resize(numElem);
    m_letam.resize(numElem);
    m_letap.resize(numElem);
    m_lzetam.resize(numElem);
    m_lzetap.resize(numElem);

    m_elemBC.resize(numElem);

    m_e.resize(numElem);
    m_p.resize(numElem);

    m_q.resize(numElem);
    m_ql.resize(numElem);
    m_qq.resize(numElem);

    m_v.resize(numElem);

    m_volo.resize(numElem);
    m_delv.resize(numElem);
    m_vdov.resize(numElem);

    m_arealg.resize(numElem);

    m_ss.resize(numElem);

    m_elemMass.resize(numElem);

    m_vnew.resize(numElem);
  }

  void AllocateGradients(Int_t numElem, Int_t allElem)
  {
    // Position gradients
    m_delx_xi = Allocate<Real_t>(numElem);
    m_delx_eta = Allocate<Real_t>(numElem);
    m_delx_zeta = Allocate<Real_t>(numElem);

    // Velocity gradients
    m_delv_xi = Allocate<Real_t>(allElem);
    m_delv_eta = Allocate<Real_t>(allElem);
    m_delv_zeta = Allocate<Real_t>(allElem);
  }

  void DeallocateGradients()
  {
    Release(&m_delx_zeta);
    Release(&m_delx_eta);
    Release(&m_delx_xi);

    Release(&m_delv_zeta);
    Release(&m_delv_eta);
    Release(&m_delv_xi);
  }

  void AllocateStrains(Int_t numElem)
  {
    m_dxx = Allocate<Real_t>(numElem);
    m_dyy = Allocate<Real_t>(numElem);
    m_dzz = Allocate<Real_t>(numElem);
  }

  void DeallocateStrains()
  {
    Release(&m_dzz);
    Release(&m_dyy);
    Release(&m_dxx);
  }

  //
  // ACCESSORS
  //

  // Node-centered

  // Nodal coordinates
  Real_t &x(Index_t idx) { return m_x[idx]; }
  Real_t &y(Index_t idx) { return m_y[idx]; }
  Real_t &z(Index_t idx) { return m_z[idx]; }

  // Nodal velocities
  Real_t &xd(Index_t idx) { return m_xd[idx]; }
  Real_t &yd(Index_t idx) { return m_yd[idx]; }
  Real_t &zd(Index_t idx) { return m_zd[idx]; }

  // Nodal accelerations
  Real_t &xdd(Index_t idx) { return m_xdd[idx]; }
  Real_t &ydd(Index_t idx) { return m_ydd[idx]; }
  Real_t &zdd(Index_t idx) { return m_zdd[idx]; }

  // Nodal forces
  Real_t &fx(Index_t idx) { return m_fx[idx]; }
  Real_t &fy(Index_t idx) { return m_fy[idx]; }
  Real_t &fz(Index_t idx) { return m_fz[idx]; }

  // Nodal mass
  Real_t &nodalMass(Index_t idx) { return m_nodalMass[idx]; }

  // Nodes on symmertry planes
  Index_t symmX(Index_t idx) { return m_symmX[idx]; }
  Index_t symmY(Index_t idx) { return m_symmY[idx]; }
  Index_t symmZ(Index_t idx) { return m_symmZ[idx]; }
  bool symmXempty() { return m_symmX.empty(); }
  bool symmYempty() { return m_symmY.empty(); }
  bool symmZempty() { return m_symmZ.empty(); }

  //
  // Element-centered
  //
  Index_t &regElemSize(Index_t idx) { return m_regElemSize[idx]; }
  Index_t &regNumList(Index_t idx) { return m_regNumList[idx]; }
  Index_t *regNumList() { return &m_regNumList[0]; }
  Index_t *regElemlist(Int_t r) { return m_regElemlist[r]; }
  Index_t &regElemlist(Int_t r, Index_t idx) { return m_regElemlist[r][idx]; }

  Index_t *nodelist(Index_t idx) { return &m_nodelist[Index_t(8) * idx]; }

  // elem connectivities through face
  Index_t &lxim(Index_t idx) { return m_lxim[idx]; }
  Index_t &lxip(Index_t idx) { return m_lxip[idx]; }
  Index_t &letam(Index_t idx) { return m_letam[idx]; }
  Index_t &letap(Index_t idx) { return m_letap[idx]; }
  Index_t &lzetam(Index_t idx) { return m_lzetam[idx]; }
  Index_t &lzetap(Index_t idx) { return m_lzetap[idx]; }

  // elem face symm/free-surface flag
  Int_t &elemBC(Index_t idx) { return m_elemBC[idx]; }

  // Principal strains - temporary
  Real_t &dxx(Index_t idx) { return m_dxx[idx]; }
  Real_t &dyy(Index_t idx) { return m_dyy[idx]; }
  Real_t &dzz(Index_t idx) { return m_dzz[idx]; }

  // New relative volume - temporary
  Real_t &vnew(Index_t idx) { return m_vnew[idx]; }

  // Velocity gradient - temporary
  Real_t &delv_xi(Index_t idx) { return m_delv_xi[idx]; }
  Real_t &delv_eta(Index_t idx) { return m_delv_eta[idx]; }
  Real_t &delv_zeta(Index_t idx) { return m_delv_zeta[idx]; }

  // Position gradient - temporary
  Real_t &delx_xi(Index_t idx) { return m_delx_xi[idx]; }
  Real_t &delx_eta(Index_t idx) { return m_delx_eta[idx]; }
  Real_t &delx_zeta(Index_t idx) { return m_delx_zeta[idx]; }

  // Energy
  Real_t &e(Index_t idx) { return m_e[idx]; }

  // Pressure
  Real_t &p(Index_t idx) { return m_p[idx]; }

  // Artificial viscosity
  Real_t &q(Index_t idx) { return m_q[idx]; }

  // Linear term for q
  Real_t &ql(Index_t idx) { return m_ql[idx]; }
  // Quadratic term for q
  Real_t &qq(Index_t idx) { return m_qq[idx]; }

  // Relative volume
  Real_t &v(Index_t idx) { return m_v[idx]; }
  Real_t &delv(Index_t idx) { return m_delv[idx]; }

  // Reference volume
  Real_t &volo(Index_t idx) { return m_volo[idx]; }

  // volume derivative over volume
  Real_t &vdov(Index_t idx) { return m_vdov[idx]; }

  // Element characteristic length
  Real_t &arealg(Index_t idx) { return m_arealg[idx]; }

  // Sound speed
  Real_t &ss(Index_t idx) { return m_ss[idx]; }

  // Element mass
  Real_t &elemMass(Index_t idx) { return m_elemMass[idx]; }

  Index_t nodeElemCount(Index_t idx)
  {
    return m_nodeElemStart[idx + 1] - m_nodeElemStart[idx];
  }

  Index_t *nodeElemCornerList(Index_t idx)
  {
    return &m_nodeElemCornerList[m_nodeElemStart[idx]];
  }

  // Parameters

  // Cutoffs
  Real_t u_cut() const { return m_u_cut; }
  Real_t e_cut() const { return m_e_cut; }
  Real_t p_cut() const { return m_p_cut; }
  Real_t q_cut() const { return m_q_cut; }
  Real_t v_cut() const { return m_v_cut; }

  // Other constants (usually are settable via input file in real codes)
  Real_t hgcoef() const { return m_hgcoef; }
  Real_t qstop() const { return m_qstop; }
  Real_t monoq_max_slope() const { return m_monoq_max_slope; }
  Real_t monoq_limiter_mult() const { return m_monoq_limiter_mult; }
  Real_t ss4o3() const { return m_ss4o3; }
  Real_t qlc_monoq() const { return m_qlc_monoq; }
  Real_t qqc_monoq() const { return m_qqc_monoq; }
  Real_t qqc() const { return m_qqc; }

  Real_t eosvmax() const { return m_eosvmax; }
  Real_t eosvmin() const { return m_eosvmin; }
  Real_t pmin() const { return m_pmin; }
  Real_t emin() const { return m_emin; }
  Real_t dvovmax() const { return m_dvovmax; }
  Real_t refdens() const { return m_refdens; }

  // Timestep controls, etc...
  Real_t &time() { return m_time; }
  Real_t &deltatime() { return m_deltatime; }
  Real_t &deltatimemultlb() { return m_deltatimemultlb; }
  Real_t &deltatimemultub() { return m_deltatimemultub; }
  Real_t &stoptime() { return m_stoptime; }
  Real_t &dtcourant() { return m_dtcourant; }
  Real_t &dthydro() { return m_dthydro; }
  Real_t &dtmax() { return m_dtmax; }
  Real_t &dtfixed() { return m_dtfixed; }

  Int_t &cycle() { return m_cycle; }
  Index_t &numRanks() { return m_numRanks; }

  Index_t &colLoc() { return m_colLoc; }
  Index_t &rowLoc() { return m_rowLoc; }
  Index_t &planeLoc() { return m_planeLoc; }
  Index_t &tp() { return m_tp; }

  Index_t &sizeX() { return m_sizeX; }
  Index_t &sizeY() { return m_sizeY; }
  Index_t &sizeZ() { return m_sizeZ; }
  Index_t &numReg() { return m_numReg; }
  Int_t &cost() { return m_cost; }
  Index_t &numElem() { return m_numElem; }
  Index_t &numNode() { return m_numNode; }

  Index_t &maxPlaneSize() { return m_maxPlaneSize; }
  Index_t &maxEdgeSize() { return m_maxEdgeSize; }

  //
  // MPI-Related additional data
  //

#if USE_MPI
  // Communication Work space
  Real_t *commDataSend;
  Real_t *commDataRecv;

  // Maximum number of block neighbors
  MPI_Request recvRequest[26]; // 6 faces + 12 edges + 8 corners
  MPI_Request sendRequest[26]; // 6 faces + 12 edges + 8 corners
#endif

private:
  void BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems);
  void SetupThreadSupportStructures();
  void CreateRegionIndexSets(Int_t nreg, Int_t balance);
  void SetupCommBuffers(Int_t edgeNodes);
  void SetupSymmetryPlanes(Int_t edgeNodes);
  void SetupElementConnectivities(Int_t edgeElems);
  void SetupBoundaryConditions(Int_t edgeElems);

  //
  // IMPLEMENTATION
  //

  /* Node-centered */
  std::vector<Real_t> m_x; /* coordinates */
  std::vector<Real_t> m_y;
  std::vector<Real_t> m_z;

  std::vector<Real_t> m_xd; /* velocities */
  std::vector<Real_t> m_yd;
  std::vector<Real_t> m_zd;

  std::vector<Real_t> m_xdd; /* accelerations */
  std::vector<Real_t> m_ydd;
  std::vector<Real_t> m_zdd;

  std::vector<Real_t> m_fx; /* forces */
  std::vector<Real_t> m_fy;
  std::vector<Real_t> m_fz;

  std::vector<Real_t> m_nodalMass; /* mass */

  std::vector<Index_t> m_symmX; /* symmetry plane nodesets */
  std::vector<Index_t> m_symmY;
  std::vector<Index_t> m_symmZ;

  // Element-centered

  // Region information
  Int_t m_numReg;
  Int_t m_cost;            // imbalance cost
  Index_t *m_regElemSize;  // Size of region sets
  Index_t *m_regNumList;   // Region number per domain element
  Index_t **m_regElemlist; // region indexset

  std::vector<Index_t> m_nodelist; /* elemToNode connectivity */

  std::vector<Index_t> m_lxim; /* element connectivity across each face */
  std::vector<Index_t> m_lxip;
  std::vector<Index_t> m_letam;
  std::vector<Index_t> m_letap;
  std::vector<Index_t> m_lzetam;
  std::vector<Index_t> m_lzetap;

  std::vector<Int_t> m_elemBC; /* symmetry/free-surface flags for each elem face */

  Real_t *m_dxx; /* principal strains -- temporary */
  Real_t *m_dyy;
  Real_t *m_dzz;

  Real_t *m_delv_xi; /* velocity gradient -- temporary */
  Real_t *m_delv_eta;
  Real_t *m_delv_zeta;

  Real_t *m_delx_xi; /* coordinate gradient -- temporary */
  Real_t *m_delx_eta;
  Real_t *m_delx_zeta;

  std::vector<Real_t> m_e; /* energy */

  std::vector<Real_t> m_p;  /* pressure */
  std::vector<Real_t> m_q;  /* q */
  std::vector<Real_t> m_ql; /* linear term for q */
  std::vector<Real_t> m_qq; /* quadratic term for q */

  std::vector<Real_t> m_v;    /* relative volume */
  std::vector<Real_t> m_volo; /* reference volume */
  std::vector<Real_t> m_vnew; /* new relative volume -- temporary */
  std::vector<Real_t> m_delv; /* m_vnew - m_v */
  std::vector<Real_t> m_vdov; /* volume derivative over volume */

  std::vector<Real_t> m_arealg; /* characteristic length of an element */

  std::vector<Real_t> m_ss; /* "sound speed" */

  std::vector<Real_t> m_elemMass; /* mass */

  // Cutoffs (treat as constants)
  const Real_t m_e_cut; // energy tolerance
  const Real_t m_p_cut; // pressure tolerance
  const Real_t m_q_cut; // q tolerance
  const Real_t m_v_cut; // relative volume tolerance
  const Real_t m_u_cut; // velocity tolerance

  // Other constants (usually setable, but hardcoded in this proxy app)

  const Real_t m_hgcoef; // hourglass control
  const Real_t m_ss4o3;
  const Real_t m_qstop; // excessive q indicator
  const Real_t m_monoq_max_slope;
  const Real_t m_monoq_limiter_mult;
  const Real_t m_qlc_monoq; // linear term coef for q
  const Real_t m_qqc_monoq; // quadratic term coef for q
  const Real_t m_qqc;
  const Real_t m_eosvmax;
  const Real_t m_eosvmin;
  const Real_t m_pmin;    // pressure floor
  const Real_t m_emin;    // energy floor
  const Real_t m_dvovmax; // maximum allowable volume change
  const Real_t m_refdens; // reference density

  // Variables to keep track of timestep, simulation time, and cycle
  Real_t m_dtcourant; // courant constraint
  Real_t m_dthydro;   // volume change constraint
  Int_t m_cycle;      // iteration count for simulation
  Real_t m_dtfixed;   // fixed time increment
  Real_t m_time;      // current time
  Real_t m_deltatime; // variable time increment
  Real_t m_deltatimemultlb;
  Real_t m_deltatimemultub;
  Real_t m_dtmax;    // maximum allowable time increment
  Real_t m_stoptime; // end time for simulation

  Int_t m_numRanks;

  Index_t m_colLoc;
  Index_t m_rowLoc;
  Index_t m_planeLoc;
  Index_t m_tp;

  Index_t m_sizeX;
  Index_t m_sizeY;
  Index_t m_sizeZ;
  Index_t m_numElem;
  Index_t m_numNode;

  Index_t m_maxPlaneSize;
  Index_t m_maxEdgeSize;

  // OMP hack
  Index_t *m_nodeElemStart;
  Index_t *m_nodeElemCornerList;

  // Used in setup
  Index_t m_rowMin, m_rowMax;
  Index_t m_colMin, m_colMax;
  Index_t m_planeMin, m_planeMax;
};

typedef Real_t &(Domain::*Domain_member)(Index_t);

//**************************************************
// Declaration
//**************************************************

static inline void CalcElemShapeFunctionDerivatives(Real_t const x[],
                                                    Real_t const y[],
                                                    Real_t const z[],
                                                    Real_t b[][8],
                                                    Real_t *const volume);

static inline void CalcElemNodeNormals(Real_t pfx[8],
                                       Real_t pfy[8],
                                       Real_t pfz[8],
                                       const Real_t x[8],
                                       const Real_t y[8],
                                       const Real_t z[8]);

//**************************************************
// Misc
//**************************************************

static inline void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
                           const Real_t x3, const Real_t x4, const Real_t x5,
                           const Real_t y0, const Real_t y1, const Real_t y2,
                           const Real_t y3, const Real_t y4, const Real_t y5,
                           const Real_t z0, const Real_t z1, const Real_t z2,
                           const Real_t z3, const Real_t z4, const Real_t z5,
                           Real_t *dvdx, Real_t *dvdy, Real_t *dvdz)
{
  const Real_t twelfth = Real_t(1.0) / Real_t(12.0);

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

//**************************************************
// Collect & Sum
//**************************************************

static inline void CollectDomainNodesToElemNodes(Domain &domain,
                                                 const Index_t *elemToNode,
                                                 Real_t elemX[8],
                                                 Real_t elemY[8],
                                                 Real_t elemZ[8])
{
  Index_t nd0i = elemToNode[0];
  Index_t nd1i = elemToNode[1];
  Index_t nd2i = elemToNode[2];
  Index_t nd3i = elemToNode[3];
  Index_t nd4i = elemToNode[4];
  Index_t nd5i = elemToNode[5];
  Index_t nd6i = elemToNode[6];
  Index_t nd7i = elemToNode[7];

  elemX[0] = domain.x(nd0i);
  elemX[1] = domain.x(nd1i);
  elemX[2] = domain.x(nd2i);
  elemX[3] = domain.x(nd3i);
  elemX[4] = domain.x(nd4i);
  elemX[5] = domain.x(nd5i);
  elemX[6] = domain.x(nd6i);
  elemX[7] = domain.x(nd7i);

  elemY[0] = domain.y(nd0i);
  elemY[1] = domain.y(nd1i);
  elemY[2] = domain.y(nd2i);
  elemY[3] = domain.y(nd3i);
  elemY[4] = domain.y(nd4i);
  elemY[5] = domain.y(nd5i);
  elemY[6] = domain.y(nd6i);
  elemY[7] = domain.y(nd7i);

  elemZ[0] = domain.z(nd0i);
  elemZ[1] = domain.z(nd1i);
  elemZ[2] = domain.z(nd2i);
  elemZ[3] = domain.z(nd3i);
  elemZ[4] = domain.z(nd4i);
  elemZ[5] = domain.z(nd5i);
  elemZ[6] = domain.z(nd6i);
  elemZ[7] = domain.z(nd7i);
}

static inline void SumElemStressesToNodeForces(const Real_t B[][8],
                                               const Real_t stress_xx,
                                               const Real_t stress_yy,
                                               const Real_t stress_zz,
                                               Real_t fx[], Real_t fy[], Real_t fz[])
{
  for (Index_t i = 0; i < 8; i++)
  {
    fx[i] = -(stress_xx * B[0][i]);
    fy[i] = -(stress_yy * B[1][i]);
    fz[i] = -(stress_zz * B[2][i]);
  }
}

static inline void SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
                                     Real_t *normalX1, Real_t *normalY1, Real_t *normalZ1,
                                     Real_t *normalX2, Real_t *normalY2, Real_t *normalZ2,
                                     Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3,
                                     const Real_t x0, const Real_t y0, const Real_t z0,
                                     const Real_t x1, const Real_t y1, const Real_t z1,
                                     const Real_t x2, const Real_t y2, const Real_t z2,
                                     const Real_t x3, const Real_t y3, const Real_t z3)
{
  Real_t bisectX0 = Real_t(0.5) * (x3 + x2 - x1 - x0);
  Real_t bisectY0 = Real_t(0.5) * (y3 + y2 - y1 - y0);
  Real_t bisectZ0 = Real_t(0.5) * (z3 + z2 - z1 - z0);
  Real_t bisectX1 = Real_t(0.5) * (x2 + x1 - x3 - x0);
  Real_t bisectY1 = Real_t(0.5) * (y2 + y1 - y3 - y0);
  Real_t bisectZ1 = Real_t(0.5) * (z2 + z1 - z3 - z0);
  Real_t areaX = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
  Real_t areaY = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
  Real_t areaZ = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

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

//**************************************************
// Init & Integrate
//**************************************************

static inline void InitStressTermsForElems(Domain &domain,
                                           Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
                                           Index_t numElem)
{
  //
  // pull in the stresses appropriate to the hydro integration
  //

#pragma omp parallel for firstprivate(numElem)
  for (Index_t i = 0; i < numElem; ++i)
  {
    sigxx[i] = sigyy[i] = sigzz[i] = -domain.p(i) - domain.q(i);
  }
}

static inline void IntegrateStressForElems(Domain &domain,
                                           Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
                                           Real_t *determ, Index_t numElem, Index_t numNode)
{
#if _OPENMP
  Index_t numthreads = omp_get_max_threads();
#else
  Index_t numthreads = 1;
#endif

  Index_t numElem8 = numElem * 8;
  Real_t *fx_elem;
  Real_t *fy_elem;
  Real_t *fz_elem;
  Real_t fx_local[8];
  Real_t fy_local[8];
  Real_t fz_local[8];

  if (numthreads > 1)
  {
    fx_elem = Allocate<Real_t>(numElem8);
    fy_elem = Allocate<Real_t>(numElem8);
    fz_elem = Allocate<Real_t>(numElem8);
  }
  // loop over all elements

#pragma omp parallel for firstprivate(numElem)
  for (Index_t k = 0; k < numElem; ++k)
  {
    const Index_t *const elemToNode = domain.nodelist(k);
    Real_t B[3][8]; // shape function derivatives
    Real_t x_local[8];
    Real_t y_local[8];
    Real_t z_local[8];

    // get nodal coordinates from global arrays and copy into local arrays.
    CollectDomainNodesToElemNodes(domain, elemToNode, x_local, y_local, z_local);

    // Volume calculation involves extra work for numerical consistency
    CalcElemShapeFunctionDerivatives(x_local, y_local, z_local,
                                     B, &determ[k]);

    CalcElemNodeNormals(B[0], B[1], B[2],
                        x_local, y_local, z_local);

    if (numthreads > 1)
    {
      // Eliminate thread writing conflicts at the nodes by giving
      // each element its own copy to write to
      SumElemStressesToNodeForces(B, sigxx[k], sigyy[k], sigzz[k],
                                  &fx_elem[k * 8],
                                  &fy_elem[k * 8],
                                  &fz_elem[k * 8]);
    }
    else
    {
      SumElemStressesToNodeForces(B, sigxx[k], sigyy[k], sigzz[k],
                                  fx_local, fy_local, fz_local);

      // copy nodal force contributions to global force arrray.
      for (Index_t lnode = 0; lnode < 8; ++lnode)
      {
        Index_t gnode = elemToNode[lnode];
        domain.fx(gnode) += fx_local[lnode];
        domain.fy(gnode) += fy_local[lnode];
        domain.fz(gnode) += fz_local[lnode];
      }
    }
  }

  if (numthreads > 1)
  {
    // If threaded, then we need to copy the data out of the temporary
    // arrays used above into the final forces field
#pragma omp parallel for firstprivate(numNode)
    for (Index_t gnode = 0; gnode < numNode; ++gnode)
    {
      Index_t count = domain.nodeElemCount(gnode);
      Index_t *cornerList = domain.nodeElemCornerList(gnode);
      Real_t fx_tmp = Real_t(0.0);
      Real_t fy_tmp = Real_t(0.0);
      Real_t fz_tmp = Real_t(0.0);
      for (Index_t i = 0; i < count; ++i)
      {
        Index_t ielem = cornerList[i];
        fx_tmp += fx_elem[ielem];
        fy_tmp += fy_elem[ielem];
        fz_tmp += fz_elem[ielem];
      }
      domain.fx(gnode) = fx_tmp;
      domain.fy(gnode) = fy_tmp;
      domain.fz(gnode) = fz_tmp;
    }
    Release(&fz_elem);
    Release(&fy_elem);
    Release(&fx_elem);
  }
}

//**************************************************
// Calcs
//**************************************************

static inline void CalcElemShapeFunctionDerivatives(Real_t const x[],
                                                    Real_t const y[],
                                                    Real_t const z[],
                                                    Real_t b[][8],
                                                    Real_t *const volume)
{
  const Real_t x0 = x[0];
  const Real_t x1 = x[1];
  const Real_t x2 = x[2];
  const Real_t x3 = x[3];
  const Real_t x4 = x[4];
  const Real_t x5 = x[5];
  const Real_t x6 = x[6];
  const Real_t x7 = x[7];

  const Real_t y0 = y[0];
  const Real_t y1 = y[1];
  const Real_t y2 = y[2];
  const Real_t y3 = y[3];
  const Real_t y4 = y[4];
  const Real_t y5 = y[5];
  const Real_t y6 = y[6];
  const Real_t y7 = y[7];

  const Real_t z0 = z[0];
  const Real_t z1 = z[1];
  const Real_t z2 = z[2];
  const Real_t z3 = z[3];
  const Real_t z4 = z[4];
  const Real_t z5 = z[5];
  const Real_t z6 = z[6];
  const Real_t z7 = z[7];

  Real_t fjxxi, fjxet, fjxze;
  Real_t fjyxi, fjyet, fjyze;
  Real_t fjzxi, fjzet, fjzze;
  Real_t cjxxi, cjxet, cjxze;
  Real_t cjyxi, cjyet, cjyze;
  Real_t cjzxi, cjzet, cjzze;

  fjxxi = Real_t(.125) * ((x6 - x0) + (x5 - x3) - (x7 - x1) - (x4 - x2));
  fjxet = Real_t(.125) * ((x6 - x0) - (x5 - x3) + (x7 - x1) - (x4 - x2));
  fjxze = Real_t(.125) * ((x6 - x0) + (x5 - x3) + (x7 - x1) + (x4 - x2));

  fjyxi = Real_t(.125) * ((y6 - y0) + (y5 - y3) - (y7 - y1) - (y4 - y2));
  fjyet = Real_t(.125) * ((y6 - y0) - (y5 - y3) + (y7 - y1) - (y4 - y2));
  fjyze = Real_t(.125) * ((y6 - y0) + (y5 - y3) + (y7 - y1) + (y4 - y2));

  fjzxi = Real_t(.125) * ((z6 - z0) + (z5 - z3) - (z7 - z1) - (z4 - z2));
  fjzet = Real_t(.125) * ((z6 - z0) - (z5 - z3) + (z7 - z1) - (z4 - z2));
  fjzze = Real_t(.125) * ((z6 - z0) + (z5 - z3) + (z7 - z1) + (z4 - z2));

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
  *volume = Real_t(8.) * (fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

static inline void CalcElemNodeNormals(Real_t pfx[8],
                                       Real_t pfy[8],
                                       Real_t pfz[8],
                                       const Real_t x[8],
                                       const Real_t y[8],
                                       const Real_t z[8])
{
  for (Index_t i = 0; i < 8; ++i)
  {
    pfx[i] = Real_t(0.0);
    pfy[i] = Real_t(0.0);
    pfz[i] = Real_t(0.0);
  }
  /* evaluate face one: nodes 0, 1, 2, 3 */
  SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                    &pfx[1], &pfy[1], &pfz[1],
                    &pfx[2], &pfy[2], &pfz[2],
                    &pfx[3], &pfy[3], &pfz[3],
                    x[0], y[0], z[0], x[1], y[1], z[1],
                    x[2], y[2], z[2], x[3], y[3], z[3]);
  /* evaluate face two: nodes 0, 4, 5, 1 */
  SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
                    &pfx[4], &pfy[4], &pfz[4],
                    &pfx[5], &pfy[5], &pfz[5],
                    &pfx[1], &pfy[1], &pfz[1],
                    x[0], y[0], z[0], x[4], y[4], z[4],
                    x[5], y[5], z[5], x[1], y[1], z[1]);
  /* evaluate face three: nodes 1, 5, 6, 2 */
  SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
                    &pfx[5], &pfy[5], &pfz[5],
                    &pfx[6], &pfy[6], &pfz[6],
                    &pfx[2], &pfy[2], &pfz[2],
                    x[1], y[1], z[1], x[5], y[5], z[5],
                    x[6], y[6], z[6], x[2], y[2], z[2]);
  /* evaluate face four: nodes 2, 6, 7, 3 */
  SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
                    &pfx[6], &pfy[6], &pfz[6],
                    &pfx[7], &pfy[7], &pfz[7],
                    &pfx[3], &pfy[3], &pfz[3],
                    x[2], y[2], z[2], x[6], y[6], z[6],
                    x[7], y[7], z[7], x[3], y[3], z[3]);
  /* evaluate face five: nodes 3, 7, 4, 0 */
  SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
                    &pfx[7], &pfy[7], &pfz[7],
                    &pfx[4], &pfy[4], &pfz[4],
                    &pfx[0], &pfy[0], &pfz[0],
                    x[3], y[3], z[3], x[7], y[7], z[7],
                    x[4], y[4], z[4], x[0], y[0], z[0]);
  /* evaluate face six: nodes 4, 7, 6, 5 */
  SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
                    &pfx[7], &pfy[7], &pfz[7],
                    &pfx[6], &pfy[6], &pfz[6],
                    &pfx[5], &pfy[5], &pfz[5],
                    x[4], y[4], z[4], x[7], y[7], z[7],
                    x[6], y[6], z[6], x[5], y[5], z[5]);
}

static inline void CalcElemVolumeDerivative(Real_t dvdx[8],
                                            Real_t dvdy[8],
                                            Real_t dvdz[8],
                                            const Real_t x[8],
                                            const Real_t y[8],
                                            const Real_t z[8])
{
  VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
          y[1], y[2], y[3], y[4], y[5], y[7],
          z[1], z[2], z[3], z[4], z[5], z[7],
          &dvdx[0], &dvdy[0], &dvdz[0]);
  VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
          y[0], y[1], y[2], y[7], y[4], y[6],
          z[0], z[1], z[2], z[7], z[4], z[6],
          &dvdx[3], &dvdy[3], &dvdz[3]);
  VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
          y[3], y[0], y[1], y[6], y[7], y[5],
          z[3], z[0], z[1], z[6], z[7], z[5],
          &dvdx[2], &dvdy[2], &dvdz[2]);
  VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
          y[2], y[3], y[0], y[5], y[6], y[4],
          z[2], z[3], z[0], z[5], z[6], z[4],
          &dvdx[1], &dvdy[1], &dvdz[1]);
  VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
          y[7], y[6], y[5], y[0], y[3], y[1],
          z[7], z[6], z[5], z[0], z[3], z[1],
          &dvdx[4], &dvdy[4], &dvdz[4]);
  VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
          y[4], y[7], y[6], y[1], y[0], y[2],
          z[4], z[7], z[6], z[1], z[0], z[2],
          &dvdx[5], &dvdy[5], &dvdz[5]);
  VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
          y[5], y[4], y[7], y[2], y[1], y[3],
          z[5], z[4], z[7], z[2], z[1], z[3],
          &dvdx[6], &dvdy[6], &dvdz[6]);
  VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
          y[6], y[5], y[4], y[3], y[2], y[0],
          z[6], z[5], z[4], z[3], z[2], z[0],
          &dvdx[7], &dvdy[7], &dvdz[7]);
}

static inline void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd, Real_t hourgam[][4],
                                            Real_t coefficient,
                                            Real_t *hgfx, Real_t *hgfy, Real_t *hgfz)
{
  Real_t hxx[4];
  for (Index_t i = 0; i < 4; i++)
  {
    hxx[i] = hourgam[0][i] * xd[0] + hourgam[1][i] * xd[1] +
             hourgam[2][i] * xd[2] + hourgam[3][i] * xd[3] +
             hourgam[4][i] * xd[4] + hourgam[5][i] * xd[5] +
             hourgam[6][i] * xd[6] + hourgam[7][i] * xd[7];
  }
  for (Index_t i = 0; i < 8; i++)
  {
    hgfx[i] = coefficient *
              (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
               hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
  }
  for (Index_t i = 0; i < 4; i++)
  {
    hxx[i] = hourgam[0][i] * yd[0] + hourgam[1][i] * yd[1] +
             hourgam[2][i] * yd[2] + hourgam[3][i] * yd[3] +
             hourgam[4][i] * yd[4] + hourgam[5][i] * yd[5] +
             hourgam[6][i] * yd[6] + hourgam[7][i] * yd[7];
  }
  for (Index_t i = 0; i < 8; i++)
  {
    hgfy[i] = coefficient *
              (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
               hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
  }
  for (Index_t i = 0; i < 4; i++)
  {
    hxx[i] = hourgam[0][i] * zd[0] + hourgam[1][i] * zd[1] +
             hourgam[2][i] * zd[2] + hourgam[3][i] * zd[3] +
             hourgam[4][i] * zd[4] + hourgam[5][i] * zd[5] +
             hourgam[6][i] * zd[6] + hourgam[7][i] * zd[7];
  }
  for (Index_t i = 0; i < 8; i++)
  {
    hgfz[i] = coefficient *
              (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
               hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
  }
}

static inline void CalcFBHourglassForceForElems(Domain &domain,
                                                Real_t *determ,
                                                Real_t *x8n, Real_t *y8n, Real_t *z8n,
                                                Real_t *dvdx, Real_t *dvdy, Real_t *dvdz,
                                                Real_t hourg, Index_t numElem,
                                                Index_t numNode)
{

#if _OPENMP
  Index_t numthreads = omp_get_max_threads();
#else
  Index_t numthreads = 1;
#endif
  /*************************************************
   *
   *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
   *               force.
   *
   *************************************************/

  Index_t numElem8 = numElem * 8;

  Real_t *fx_elem;
  Real_t *fy_elem;
  Real_t *fz_elem;

  if (numthreads > 1)
  {
    fx_elem = Allocate<Real_t>(numElem8);
    fy_elem = Allocate<Real_t>(numElem8);
    fz_elem = Allocate<Real_t>(numElem8);
  }

  Real_t gamma[4][8];

  gamma[0][0] = Real_t(1.);
  gamma[0][1] = Real_t(1.);
  gamma[0][2] = Real_t(-1.);
  gamma[0][3] = Real_t(-1.);
  gamma[0][4] = Real_t(-1.);
  gamma[0][5] = Real_t(-1.);
  gamma[0][6] = Real_t(1.);
  gamma[0][7] = Real_t(1.);
  gamma[1][0] = Real_t(1.);
  gamma[1][1] = Real_t(-1.);
  gamma[1][2] = Real_t(-1.);
  gamma[1][3] = Real_t(1.);
  gamma[1][4] = Real_t(-1.);
  gamma[1][5] = Real_t(1.);
  gamma[1][6] = Real_t(1.);
  gamma[1][7] = Real_t(-1.);
  gamma[2][0] = Real_t(1.);
  gamma[2][1] = Real_t(-1.);
  gamma[2][2] = Real_t(1.);
  gamma[2][3] = Real_t(-1.);
  gamma[2][4] = Real_t(1.);
  gamma[2][5] = Real_t(-1.);
  gamma[2][6] = Real_t(1.);
  gamma[2][7] = Real_t(-1.);
  gamma[3][0] = Real_t(-1.);
  gamma[3][1] = Real_t(1.);
  gamma[3][2] = Real_t(-1.);
  gamma[3][3] = Real_t(1.);
  gamma[3][4] = Real_t(1.);
  gamma[3][5] = Real_t(-1.);
  gamma[3][6] = Real_t(1.);
  gamma[3][7] = Real_t(-1.);

  /*************************************************/
  /*    compute the hourglass modes */

#pragma omp parallel for firstprivate(numElem, hourg)
  for (Index_t i2 = 0; i2 < numElem; ++i2)
  {
    Real_t *fx_local, *fy_local, *fz_local;
    Real_t hgfx[8], hgfy[8], hgfz[8];

    Real_t coefficient;

    Real_t hourgam[8][4];
    Real_t xd1[8], yd1[8], zd1[8];

    const Index_t *elemToNode = domain.nodelist(i2);
    Index_t i3 = 8 * i2;
    Real_t volinv = Real_t(1.0) / determ[i2];
    Real_t ss1, mass1, volume13;
    for (Index_t i1 = 0; i1 < 4; ++i1)
    {

      Real_t hourmodx =
          x8n[i3] * gamma[i1][0] + x8n[i3 + 1] * gamma[i1][1] +
          x8n[i3 + 2] * gamma[i1][2] + x8n[i3 + 3] * gamma[i1][3] +
          x8n[i3 + 4] * gamma[i1][4] + x8n[i3 + 5] * gamma[i1][5] +
          x8n[i3 + 6] * gamma[i1][6] + x8n[i3 + 7] * gamma[i1][7];

      Real_t hourmody =
          y8n[i3] * gamma[i1][0] + y8n[i3 + 1] * gamma[i1][1] +
          y8n[i3 + 2] * gamma[i1][2] + y8n[i3 + 3] * gamma[i1][3] +
          y8n[i3 + 4] * gamma[i1][4] + y8n[i3 + 5] * gamma[i1][5] +
          y8n[i3 + 6] * gamma[i1][6] + y8n[i3 + 7] * gamma[i1][7];

      Real_t hourmodz =
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

    ss1 = domain.ss(i2);
    mass1 = domain.elemMass(i2);
    volume13 = CBRT(determ[i2]);

    Index_t n0si2 = elemToNode[0];
    Index_t n1si2 = elemToNode[1];
    Index_t n2si2 = elemToNode[2];
    Index_t n3si2 = elemToNode[3];
    Index_t n4si2 = elemToNode[4];
    Index_t n5si2 = elemToNode[5];
    Index_t n6si2 = elemToNode[6];
    Index_t n7si2 = elemToNode[7];

    xd1[0] = domain.xd(n0si2);
    xd1[1] = domain.xd(n1si2);
    xd1[2] = domain.xd(n2si2);
    xd1[3] = domain.xd(n3si2);
    xd1[4] = domain.xd(n4si2);
    xd1[5] = domain.xd(n5si2);
    xd1[6] = domain.xd(n6si2);
    xd1[7] = domain.xd(n7si2);

    yd1[0] = domain.yd(n0si2);
    yd1[1] = domain.yd(n1si2);
    yd1[2] = domain.yd(n2si2);
    yd1[3] = domain.yd(n3si2);
    yd1[4] = domain.yd(n4si2);
    yd1[5] = domain.yd(n5si2);
    yd1[6] = domain.yd(n6si2);
    yd1[7] = domain.yd(n7si2);

    zd1[0] = domain.zd(n0si2);
    zd1[1] = domain.zd(n1si2);
    zd1[2] = domain.zd(n2si2);
    zd1[3] = domain.zd(n3si2);
    zd1[4] = domain.zd(n4si2);
    zd1[5] = domain.zd(n5si2);
    zd1[6] = domain.zd(n6si2);
    zd1[7] = domain.zd(n7si2);

    coefficient = -hourg * Real_t(0.01) * ss1 * mass1 / volume13;

    CalcElemFBHourglassForce(xd1, yd1, zd1,
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
      domain.fx(n0si2) += hgfx[0];
      domain.fy(n0si2) += hgfy[0];
      domain.fz(n0si2) += hgfz[0];

      domain.fx(n1si2) += hgfx[1];
      domain.fy(n1si2) += hgfy[1];
      domain.fz(n1si2) += hgfz[1];

      domain.fx(n2si2) += hgfx[2];
      domain.fy(n2si2) += hgfy[2];
      domain.fz(n2si2) += hgfz[2];

      domain.fx(n3si2) += hgfx[3];
      domain.fy(n3si2) += hgfy[3];
      domain.fz(n3si2) += hgfz[3];

      domain.fx(n4si2) += hgfx[4];
      domain.fy(n4si2) += hgfy[4];
      domain.fz(n4si2) += hgfz[4];

      domain.fx(n5si2) += hgfx[5];
      domain.fy(n5si2) += hgfy[5];
      domain.fz(n5si2) += hgfz[5];

      domain.fx(n6si2) += hgfx[6];
      domain.fy(n6si2) += hgfy[6];
      domain.fz(n6si2) += hgfz[6];

      domain.fx(n7si2) += hgfx[7];
      domain.fy(n7si2) += hgfy[7];
      domain.fz(n7si2) += hgfz[7];
    }
  }

  if (numthreads > 1)
  {
    // Collect the data from the local arrays into the final force arrays
#pragma omp parallel for firstprivate(numNode)
    for (Index_t gnode = 0; gnode < numNode; ++gnode)
    {
      Index_t count = domain.nodeElemCount(gnode);
      Index_t *cornerList = domain.nodeElemCornerList(gnode);
      Real_t fx_tmp = Real_t(0.0);
      Real_t fy_tmp = Real_t(0.0);
      Real_t fz_tmp = Real_t(0.0);
      for (Index_t i = 0; i < count; ++i)
      {
        Index_t ielem = cornerList[i];
        fx_tmp += fx_elem[ielem];
        fy_tmp += fy_elem[ielem];
        fz_tmp += fz_elem[ielem];
      }
      domain.fx(gnode) += fx_tmp;
      domain.fy(gnode) += fy_tmp;
      domain.fz(gnode) += fz_tmp;
    }
    Release(&fz_elem);
    Release(&fy_elem);
    Release(&fx_elem);
  }
}

static inline void CalcHourglassControlForElems(Domain &domain,
                                                Real_t determ[], Real_t hgcoef)
{
  Index_t numElem = domain.numElem();
  Index_t numElem8 = numElem * 8;
  Real_t *dvdx = Allocate<Real_t>(numElem8);
  Real_t *dvdy = Allocate<Real_t>(numElem8);
  Real_t *dvdz = Allocate<Real_t>(numElem8);
  Real_t *x8n = Allocate<Real_t>(numElem8);
  Real_t *y8n = Allocate<Real_t>(numElem8);
  Real_t *z8n = Allocate<Real_t>(numElem8);

  /* start loop over elements */
#pragma omp parallel for firstprivate(numElem)
  for (Index_t i = 0; i < numElem; ++i)
  {
    Real_t x1[8], y1[8], z1[8];
    Real_t pfx[8], pfy[8], pfz[8];

    Index_t *elemToNode = domain.nodelist(i);
    CollectDomainNodesToElemNodes(domain, elemToNode, x1, y1, z1);

    CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

    /* load into temporary storage for FB Hour Glass control */
    for (Index_t ii = 0; ii < 8; ++ii)
    {
      Index_t jj = 8 * i + ii;

      dvdx[jj] = pfx[ii];
      dvdy[jj] = pfy[ii];
      dvdz[jj] = pfz[ii];
      x8n[jj] = x1[ii];
      y8n[jj] = y1[ii];
      z8n[jj] = z1[ii];
    }

    determ[i] = domain.volo(i) * domain.v(i);

    /* Do a check for negative volumes */
    if (domain.v(i) <= Real_t(0.0))
    {
#if USE_MPI
      MPI_Abort(MPI_COMM_WORLD, VolumeError);
#else
      exit(VolumeError);
#endif
    }
  }

  if (hgcoef > Real_t(0.))
  {
    CalcFBHourglassForceForElems(domain,
                                 determ, x8n, y8n, z8n, dvdx, dvdy, dvdz,
                                 hgcoef, numElem, domain.numNode());
  }

  Release(&z8n);
  Release(&y8n);
  Release(&x8n);
  Release(&dvdz);
  Release(&dvdy);
  Release(&dvdx);

  return;
}

void CalcVolumeForceForElems_Extern(Domain &domain)
{
  Index_t numElem = domain.numElem();
  if (numElem != 0)
  {
    Real_t hgcoef = domain.hgcoef();
    Real_t *sigxx = Allocate<Real_t>(numElem);
    Real_t *sigyy = Allocate<Real_t>(numElem);
    Real_t *sigzz = Allocate<Real_t>(numElem);
    Real_t *determ = Allocate<Real_t>(numElem);

    /* Sum contributions to total stress tensor */
    InitStressTermsForElems(domain, sigxx, sigyy, sigzz, numElem);

    // call elemlib stress integration loop to produce nodal forces from
    // material stresses.
    IntegrateStressForElems(domain,
                            sigxx, sigyy, sigzz, determ, numElem,
                            domain.numNode());

    // check for negative element volume
#pragma omp parallel for firstprivate(numElem)
    for (Index_t k = 0; k < numElem; ++k)
    {
      if (determ[k] <= Real_t(0.0))
      {
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, VolumeError);
#else
        exit(VolumeError);
#endif
      }
    }

    CalcHourglassControlForElems(domain, determ, hgcoef);

    Release(&determ);
    Release(&sigzz);
    Release(&sigyy);
    Release(&sigxx);
  }
}
