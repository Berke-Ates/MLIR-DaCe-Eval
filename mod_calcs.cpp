#include <math.h>
#include <vector>

#if !defined(USE_MPI)
#error "You should specify USE_MPI=0 or USE_MPI=1 on the compile line"
#endif

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
  Domain(int32_t numRanks, int32_t colLoc,
         int32_t rowLoc, int32_t planeLoc,
         int32_t nx, int32_t tp, int32_t nr, int32_t balance, int32_t cost);

  // Destructor
  ~Domain();

  //
  // ALLOCATION
  //

  void AllocateNodePersistent(int32_t numNode) // Node-centered
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

  void AllocateElemPersistent(int32_t numElem) // Elem-centered
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

  void AllocateGradients(int32_t numElem, int32_t allElem)
  {
    // Position gradients
    m_delx_xi = Allocate<double>(numElem);
    m_delx_eta = Allocate<double>(numElem);
    m_delx_zeta = Allocate<double>(numElem);

    // Velocity gradients
    m_delv_xi = Allocate<double>(allElem);
    m_delv_eta = Allocate<double>(allElem);
    m_delv_zeta = Allocate<double>(allElem);
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

  void AllocateStrains(int32_t numElem)
  {
    m_dxx = Allocate<double>(numElem);
    m_dyy = Allocate<double>(numElem);
    m_dzz = Allocate<double>(numElem);
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
  double &x(int32_t idx) { return m_x[idx]; }
  double &y(int32_t idx) { return m_y[idx]; }
  double &z(int32_t idx) { return m_z[idx]; }

  // Nodal velocities
  double &xd(int32_t idx) { return m_xd[idx]; }
  double &yd(int32_t idx) { return m_yd[idx]; }
  double &zd(int32_t idx) { return m_zd[idx]; }

  // Nodal accelerations
  double &xdd(int32_t idx) { return m_xdd[idx]; }
  double &ydd(int32_t idx) { return m_ydd[idx]; }
  double &zdd(int32_t idx) { return m_zdd[idx]; }

  // Nodal forces
  double &fx(int32_t idx) { return m_fx[idx]; }
  double &fy(int32_t idx) { return m_fy[idx]; }
  double &fz(int32_t idx) { return m_fz[idx]; }

  // Nodal mass
  double &nodalMass(int32_t idx) { return m_nodalMass[idx]; }

  // Nodes on symmertry planes
  int32_t symmX(int32_t idx) { return m_symmX[idx]; }
  int32_t symmY(int32_t idx) { return m_symmY[idx]; }
  int32_t symmZ(int32_t idx) { return m_symmZ[idx]; }
  bool symmXempty() { return m_symmX.empty(); }
  bool symmYempty() { return m_symmY.empty(); }
  bool symmZempty() { return m_symmZ.empty(); }

  //
  // Element-centered
  //
  int32_t &regElemSize(int32_t idx) { return m_regElemSize[idx]; }
  int32_t &regNumList(int32_t idx) { return m_regNumList[idx]; }
  int32_t *regNumList() { return &m_regNumList[0]; }
  int32_t *regElemlist(int32_t r) { return m_regElemlist[r]; }
  int32_t &regElemlist(int32_t r, int32_t idx) { return m_regElemlist[r][idx]; }

  int32_t *nodelist(int32_t idx) { return &m_nodelist[int32_t(8) * idx]; }

  // elem connectivities through face
  int32_t &lxim(int32_t idx) { return m_lxim[idx]; }
  int32_t &lxip(int32_t idx) { return m_lxip[idx]; }
  int32_t &letam(int32_t idx) { return m_letam[idx]; }
  int32_t &letap(int32_t idx) { return m_letap[idx]; }
  int32_t &lzetam(int32_t idx) { return m_lzetam[idx]; }
  int32_t &lzetap(int32_t idx) { return m_lzetap[idx]; }

  // elem face symm/free-surface flag
  int32_t &elemBC(int32_t idx) { return m_elemBC[idx]; }

  // Principal strains - temporary
  double &dxx(int32_t idx) { return m_dxx[idx]; }
  double &dyy(int32_t idx) { return m_dyy[idx]; }
  double &dzz(int32_t idx) { return m_dzz[idx]; }

  // New relative volume - temporary
  double &vnew(int32_t idx) { return m_vnew[idx]; }

  // Velocity gradient - temporary
  double &delv_xi(int32_t idx) { return m_delv_xi[idx]; }
  double &delv_eta(int32_t idx) { return m_delv_eta[idx]; }
  double &delv_zeta(int32_t idx) { return m_delv_zeta[idx]; }

  // Position gradient - temporary
  double &delx_xi(int32_t idx) { return m_delx_xi[idx]; }
  double &delx_eta(int32_t idx) { return m_delx_eta[idx]; }
  double &delx_zeta(int32_t idx) { return m_delx_zeta[idx]; }

  // Energy
  double &e(int32_t idx) { return m_e[idx]; }

  // Pressure
  double &p(int32_t idx) { return m_p[idx]; }

  // Artificial viscosity
  double &q(int32_t idx) { return m_q[idx]; }

  // Linear term for q
  double &ql(int32_t idx) { return m_ql[idx]; }
  // Quadratic term for q
  double &qq(int32_t idx) { return m_qq[idx]; }

  // Relative volume
  double &v(int32_t idx) { return m_v[idx]; }
  double &delv(int32_t idx) { return m_delv[idx]; }

  // Reference volume
  double &volo(int32_t idx) { return m_volo[idx]; }

  // volume derivative over volume
  double &vdov(int32_t idx) { return m_vdov[idx]; }

  // Element characteristic length
  double &arealg(int32_t idx) { return m_arealg[idx]; }

  // Sound speed
  double &ss(int32_t idx) { return m_ss[idx]; }

  // Element mass
  double &elemMass(int32_t idx) { return m_elemMass[idx]; }

  int32_t nodeElemCount(int32_t idx)
  {
    return m_nodeElemStart[idx + 1] - m_nodeElemStart[idx];
  }

  int32_t *nodeElemCornerList(int32_t idx)
  {
    return &m_nodeElemCornerList[m_nodeElemStart[idx]];
  }

  // Parameters

  // Cutoffs
  double u_cut() const { return m_u_cut; }
  double e_cut() const { return m_e_cut; }
  double p_cut() const { return m_p_cut; }
  double q_cut() const { return m_q_cut; }
  double v_cut() const { return m_v_cut; }

  // Other constants (usually are settable via input file in real codes)
  double hgcoef() const { return m_hgcoef; }
  double qstop() const { return m_qstop; }
  double monoq_max_slope() const { return m_monoq_max_slope; }
  double monoq_limiter_mult() const { return m_monoq_limiter_mult; }
  double ss4o3() const { return m_ss4o3; }
  double qlc_monoq() const { return m_qlc_monoq; }
  double qqc_monoq() const { return m_qqc_monoq; }
  double qqc() const { return m_qqc; }

  double eosvmax() const { return m_eosvmax; }
  double eosvmin() const { return m_eosvmin; }
  double pmin() const { return m_pmin; }
  double emin() const { return m_emin; }
  double dvovmax() const { return m_dvovmax; }
  double refdens() const { return m_refdens; }

  // Timestep controls, etc...
  double &time() { return m_time; }
  double &deltatime() { return m_deltatime; }
  double &deltatimemultlb() { return m_deltatimemultlb; }
  double &deltatimemultub() { return m_deltatimemultub; }
  double &stoptime() { return m_stoptime; }
  double &dtcourant() { return m_dtcourant; }
  double &dthydro() { return m_dthydro; }
  double &dtmax() { return m_dtmax; }
  double &dtfixed() { return m_dtfixed; }

  int32_t &cycle() { return m_cycle; }
  int32_t &numRanks() { return m_numRanks; }

  int32_t &colLoc() { return m_colLoc; }
  int32_t &rowLoc() { return m_rowLoc; }
  int32_t &planeLoc() { return m_planeLoc; }
  int32_t &tp() { return m_tp; }

  int32_t &sizeX() { return m_sizeX; }
  int32_t &sizeY() { return m_sizeY; }
  int32_t &sizeZ() { return m_sizeZ; }
  int32_t &numReg() { return m_numReg; }
  int32_t &cost() { return m_cost; }
  int32_t &numElem() { return m_numElem; }
  int32_t &numNode() { return m_numNode; }

  int32_t &maxPlaneSize() { return m_maxPlaneSize; }
  int32_t &maxEdgeSize() { return m_maxEdgeSize; }

  //
  // MPI-Related additional data
  //

#if USE_MPI
  // Communication Work space
  double *commDataSend;
  double *commDataRecv;

  // Maximum number of block neighbors
  MPI_Request recvRequest[26]; // 6 faces + 12 edges + 8 corners
  MPI_Request sendRequest[26]; // 6 faces + 12 edges + 8 corners
#endif

private:
  void BuildMesh(int32_t nx, int32_t edgeNodes, int32_t edgeElems);
  void SetupThreadSupportStructures();
  void CreateRegionIndexSets(int32_t nreg, int32_t balance);
  void SetupCommBuffers(int32_t edgeNodes);
  void SetupSymmetryPlanes(int32_t edgeNodes);
  void SetupElementConnectivities(int32_t edgeElems);
  void SetupBoundaryConditions(int32_t edgeElems);

  //
  // IMPLEMENTATION
  //

  /* Node-centered */
  std::vector<double> m_x; /* coordinates */
  std::vector<double> m_y;
  std::vector<double> m_z;

  std::vector<double> m_xd; /* velocities */
  std::vector<double> m_yd;
  std::vector<double> m_zd;

  std::vector<double> m_xdd; /* accelerations */
  std::vector<double> m_ydd;
  std::vector<double> m_zdd;

  std::vector<double> m_fx; /* forces */
  std::vector<double> m_fy;
  std::vector<double> m_fz;

  std::vector<double> m_nodalMass; /* mass */

  std::vector<int32_t> m_symmX; /* symmetry plane nodesets */
  std::vector<int32_t> m_symmY;
  std::vector<int32_t> m_symmZ;

  // Element-centered

  // Region information
  int32_t m_numReg;
  int32_t m_cost;          // imbalance cost
  int32_t *m_regElemSize;  // Size of region sets
  int32_t *m_regNumList;   // Region number per domain element
  int32_t **m_regElemlist; // region indexset

  std::vector<int32_t> m_nodelist; /* elemToNode connectivity */

  std::vector<int32_t> m_lxim; /* element connectivity across each face */
  std::vector<int32_t> m_lxip;
  std::vector<int32_t> m_letam;
  std::vector<int32_t> m_letap;
  std::vector<int32_t> m_lzetam;
  std::vector<int32_t> m_lzetap;

  std::vector<int32_t> m_elemBC; /* symmetry/free-surface flags for each elem face */

  double *m_dxx; /* principal strains -- temporary */
  double *m_dyy;
  double *m_dzz;

  double *m_delv_xi; /* velocity gradient -- temporary */
  double *m_delv_eta;
  double *m_delv_zeta;

  double *m_delx_xi; /* coordinate gradient -- temporary */
  double *m_delx_eta;
  double *m_delx_zeta;

  std::vector<double> m_e; /* energy */

  std::vector<double> m_p;  /* pressure */
  std::vector<double> m_q;  /* q */
  std::vector<double> m_ql; /* linear term for q */
  std::vector<double> m_qq; /* quadratic term for q */

  std::vector<double> m_v;    /* relative volume */
  std::vector<double> m_volo; /* reference volume */
  std::vector<double> m_vnew; /* new relative volume -- temporary */
  std::vector<double> m_delv; /* m_vnew - m_v */
  std::vector<double> m_vdov; /* volume derivative over volume */

  std::vector<double> m_arealg; /* characteristic length of an element */

  std::vector<double> m_ss; /* "sound speed" */

  std::vector<double> m_elemMass; /* mass */

  // Cutoffs (treat as constants)
  const double m_e_cut; // energy tolerance
  const double m_p_cut; // pressure tolerance
  const double m_q_cut; // q tolerance
  const double m_v_cut; // relative volume tolerance
  const double m_u_cut; // velocity tolerance

  // Other constants (usually setable, but hardcoded in this proxy app)

  const double m_hgcoef; // hourglass control
  const double m_ss4o3;
  const double m_qstop; // excessive q indicator
  const double m_monoq_max_slope;
  const double m_monoq_limiter_mult;
  const double m_qlc_monoq; // linear term coef for q
  const double m_qqc_monoq; // quadratic term coef for q
  const double m_qqc;
  const double m_eosvmax;
  const double m_eosvmin;
  const double m_pmin;    // pressure floor
  const double m_emin;    // energy floor
  const double m_dvovmax; // maximum allowable volume change
  const double m_refdens; // reference density

  // Variables to keep track of timestep, simulation time, and cycle
  double m_dtcourant; // courant constraint
  double m_dthydro;   // volume change constraint
  int32_t m_cycle;    // iteration count for simulation
  double m_dtfixed;   // fixed time increment
  double m_time;      // current time
  double m_deltatime; // variable time increment
  double m_deltatimemultlb;
  double m_deltatimemultub;
  double m_dtmax;    // maximum allowable time increment
  double m_stoptime; // end time for simulation

  int32_t m_numRanks;

  int32_t m_colLoc;
  int32_t m_rowLoc;
  int32_t m_planeLoc;
  int32_t m_tp;

  int32_t m_sizeX;
  int32_t m_sizeY;
  int32_t m_sizeZ;
  int32_t m_numElem;
  int32_t m_numNode;

  int32_t m_maxPlaneSize;
  int32_t m_maxEdgeSize;

  // OMP hack
  int32_t *m_nodeElemStart;
  int32_t *m_nodeElemCornerList;

  // Used in setup
  int32_t m_rowMin, m_rowMax;
  int32_t m_colMin, m_colMax;
  int32_t m_planeMin, m_planeMax;
};

//**************************************************
// Declaration
//**************************************************

static inline void CalcElemShapeFunctionDerivatives(double const x[],
                                                    double const y[],
                                                    double const z[],
                                                    double b[][8],
                                                    double *const volume);

static inline void CalcElemNodeNormals(double pfx[8],
                                       double pfy[8],
                                       double pfz[8],
                                       const double x[8],
                                       const double y[8],
                                       const double z[8]);

//**************************************************
// Misc
//**************************************************

static inline void VoluDer(const double x0, const double x1, const double x2,
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

//**************************************************
// Collect & Sum
//**************************************************

static inline void CollectDomainNodesToElemNodes(Domain &domain,
                                                 const int32_t *elemToNode,
                                                 double elemX[8],
                                                 double elemY[8],
                                                 double elemZ[8])
{
  int32_t nd0i = elemToNode[0];
  int32_t nd1i = elemToNode[1];
  int32_t nd2i = elemToNode[2];
  int32_t nd3i = elemToNode[3];
  int32_t nd4i = elemToNode[4];
  int32_t nd5i = elemToNode[5];
  int32_t nd6i = elemToNode[6];
  int32_t nd7i = elemToNode[7];

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

static inline void SumElemStressesToNodeForces(const double B[][8],
                                               const double stress_xx,
                                               const double stress_yy,
                                               const double stress_zz,
                                               double fx[], double fy[], double fz[])
{
  for (int32_t i = 0; i < 8; i++)
  {
    fx[i] = -(stress_xx * B[0][i]);
    fy[i] = -(stress_yy * B[1][i]);
    fz[i] = -(stress_zz * B[2][i]);
  }
}

static inline void SumElemFaceNormal(double *normalX0, double *normalY0, double *normalZ0,
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

//**************************************************
// Init & Integrate
//**************************************************

static inline void InitStressTermsForElems(Domain &domain,
                                           double *sigxx, double *sigyy, double *sigzz,
                                           int32_t numElem)
{
  //
  // pull in the stresses appropriate to the hydro integration
  //

#pragma omp parallel for firstprivate(numElem)
  for (int32_t i = 0; i < numElem; ++i)
  {
    sigxx[i] = sigyy[i] = sigzz[i] = -domain.p(i) - domain.q(i);
  }
}

static inline void IntegrateStressForElems(Domain &domain,
                                           double *sigxx, double *sigyy, double *sigzz,
                                           double *determ, int32_t numElem, int32_t numNode)
{
#if _OPENMP
  int32_t numthreads = omp_get_max_threads();
#else
  int32_t numthreads = 1;
#endif

  int32_t numElem8 = numElem * 8;
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
  for (int32_t k = 0; k < numElem; ++k)
  {
    const int32_t *const elemToNode = domain.nodelist(k);
    double B[3][8]; // shape function derivatives
    double x_local[8];
    double y_local[8];
    double z_local[8];

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
      for (int32_t lnode = 0; lnode < 8; ++lnode)
      {
        int32_t gnode = elemToNode[lnode];
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
    for (int32_t gnode = 0; gnode < numNode; ++gnode)
    {
      int32_t count = domain.nodeElemCount(gnode);
      int32_t *cornerList = domain.nodeElemCornerList(gnode);
      double fx_tmp = double(0.0);
      double fy_tmp = double(0.0);
      double fz_tmp = double(0.0);
      for (int32_t i = 0; i < count; ++i)
      {
        int32_t ielem = cornerList[i];
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

static inline void CalcElemShapeFunctionDerivatives(double const x[],
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

static inline void CalcElemNodeNormals(double pfx[8],
                                       double pfy[8],
                                       double pfz[8],
                                       const double x[8],
                                       const double y[8],
                                       const double z[8])
{
  for (int32_t i = 0; i < 8; ++i)
  {
    pfx[i] = double(0.0);
    pfy[i] = double(0.0);
    pfz[i] = double(0.0);
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

static inline void CalcElemVolumeDerivative(double dvdx[8],
                                            double dvdy[8],
                                            double dvdz[8],
                                            const double x[8],
                                            const double y[8],
                                            const double z[8])
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

static inline void CalcElemFBHourglassForce(double *xd, double *yd, double *zd, double hourgam[][4],
                                            double coefficient,
                                            double *hgfx, double *hgfy, double *hgfz)
{
  double hxx[4];
  for (int32_t i = 0; i < 4; i++)
  {
    hxx[i] = hourgam[0][i] * xd[0] + hourgam[1][i] * xd[1] +
             hourgam[2][i] * xd[2] + hourgam[3][i] * xd[3] +
             hourgam[4][i] * xd[4] + hourgam[5][i] * xd[5] +
             hourgam[6][i] * xd[6] + hourgam[7][i] * xd[7];
  }
  for (int32_t i = 0; i < 8; i++)
  {
    hgfx[i] = coefficient *
              (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
               hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
  }
  for (int32_t i = 0; i < 4; i++)
  {
    hxx[i] = hourgam[0][i] * yd[0] + hourgam[1][i] * yd[1] +
             hourgam[2][i] * yd[2] + hourgam[3][i] * yd[3] +
             hourgam[4][i] * yd[4] + hourgam[5][i] * yd[5] +
             hourgam[6][i] * yd[6] + hourgam[7][i] * yd[7];
  }
  for (int32_t i = 0; i < 8; i++)
  {
    hgfy[i] = coefficient *
              (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
               hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
  }
  for (int32_t i = 0; i < 4; i++)
  {
    hxx[i] = hourgam[0][i] * zd[0] + hourgam[1][i] * zd[1] +
             hourgam[2][i] * zd[2] + hourgam[3][i] * zd[3] +
             hourgam[4][i] * zd[4] + hourgam[5][i] * zd[5] +
             hourgam[6][i] * zd[6] + hourgam[7][i] * zd[7];
  }
  for (int32_t i = 0; i < 8; i++)
  {
    hgfz[i] = coefficient *
              (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
               hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
  }
}

static inline void CalcFBHourglassForceForElems(Domain &domain,
                                                double *determ,
                                                double *x8n, double *y8n, double *z8n,
                                                double *dvdx, double *dvdy, double *dvdz,
                                                double hourg, int32_t numElem,
                                                int32_t numNode)
{

#if _OPENMP
  int32_t numthreads = omp_get_max_threads();
#else
  int32_t numthreads = 1;
#endif
  /*************************************************
   *
   *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
   *               force.
   *
   *************************************************/

  int32_t numElem8 = numElem * 8;

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
  for (int32_t i2 = 0; i2 < numElem; ++i2)
  {
    double *fx_local, *fy_local, *fz_local;
    double hgfx[8], hgfy[8], hgfz[8];

    double coefficient;

    double hourgam[8][4];
    double xd1[8], yd1[8], zd1[8];

    const int32_t *elemToNode = domain.nodelist(i2);
    int32_t i3 = 8 * i2;
    double volinv = double(1.0) / determ[i2];
    double ss1, mass1, volume13;
    for (int32_t i1 = 0; i1 < 4; ++i1)
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

    ss1 = domain.ss(i2);
    mass1 = domain.elemMass(i2);
    volume13 = cbrt(determ[i2]);

    int32_t n0si2 = elemToNode[0];
    int32_t n1si2 = elemToNode[1];
    int32_t n2si2 = elemToNode[2];
    int32_t n3si2 = elemToNode[3];
    int32_t n4si2 = elemToNode[4];
    int32_t n5si2 = elemToNode[5];
    int32_t n6si2 = elemToNode[6];
    int32_t n7si2 = elemToNode[7];

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

    coefficient = -hourg * double(0.01) * ss1 * mass1 / volume13;

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
    for (int32_t gnode = 0; gnode < numNode; ++gnode)
    {
      int32_t count = domain.nodeElemCount(gnode);
      int32_t *cornerList = domain.nodeElemCornerList(gnode);
      double fx_tmp = double(0.0);
      double fy_tmp = double(0.0);
      double fz_tmp = double(0.0);
      for (int32_t i = 0; i < count; ++i)
      {
        int32_t ielem = cornerList[i];
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
                                                double determ[], double hgcoef)
{
  int32_t numElem = domain.numElem();
  int32_t numElem8 = numElem * 8;
  double *dvdx = Allocate<double>(numElem8);
  double *dvdy = Allocate<double>(numElem8);
  double *dvdz = Allocate<double>(numElem8);
  double *x8n = Allocate<double>(numElem8);
  double *y8n = Allocate<double>(numElem8);
  double *z8n = Allocate<double>(numElem8);

  /* start loop over elements */
#pragma omp parallel for firstprivate(numElem)
  for (int32_t i = 0; i < numElem; ++i)
  {
    double x1[8], y1[8], z1[8];
    double pfx[8], pfy[8], pfz[8];

    int32_t *elemToNode = domain.nodelist(i);
    CollectDomainNodesToElemNodes(domain, elemToNode, x1, y1, z1);

    CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

    /* load into temporary storage for FB Hour Glass control */
    for (int32_t ii = 0; ii < 8; ++ii)
    {
      int32_t jj = 8 * i + ii;

      dvdx[jj] = pfx[ii];
      dvdy[jj] = pfy[ii];
      dvdz[jj] = pfz[ii];
      x8n[jj] = x1[ii];
      y8n[jj] = y1[ii];
      z8n[jj] = z1[ii];
    }

    determ[i] = domain.volo(i) * domain.v(i);

    /* Do a check for negative volumes */
    if (domain.v(i) <= double(0.0))
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
  int32_t numElem = domain.numElem();
  if (numElem != 0)
  {
    double hgcoef = domain.hgcoef();
    double *sigxx = Allocate<double>(numElem);
    double *sigyy = Allocate<double>(numElem);
    double *sigzz = Allocate<double>(numElem);
    double *determ = Allocate<double>(numElem);

    /* Sum contributions to total stress tensor */
    InitStressTermsForElems(domain, sigxx, sigyy, sigzz, numElem);

    // call elemlib stress integration loop to produce nodal forces from
    // material stresses.
    IntegrateStressForElems(domain,
                            sigxx, sigyy, sigzz, determ, numElem,
                            domain.numNode());

    // check for negative element volume
#pragma omp parallel for firstprivate(numElem)
    for (int32_t k = 0; k < numElem; ++k)
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

    CalcHourglassControlForElems(domain, determ, hgcoef);

    Release(&determ);
    Release(&sigzz);
    Release(&sigyy);
    Release(&sigxx);
  }
}
