// src/heisenberg.cpp — M28-E: 1D XXZ Heisenberg Spin Chain
//
//   H = J/2 Σ_{⟨i,i+1⟩} (X_i X_{i+1} + Y_i Y_{i+1}) + Jz Σ_{⟨i,i+1⟩} Z_i Z_{i+1}/4
//       − h/2 Σ_i Z_i
//
// Written in the computational (Z) basis |↑↓…⟩.  H conserves total S^z, so we
// block-diagonalise by magnetisation sector m = Σ_i (s_iz), where s_iz = +1 for |↑⟩ = 0-bit
// and −1 for |↓⟩ = 1-bit.  We work in the Sz = 0 sector (half-filling for even N).
//
// CSR sparse Hamiltonian + real Lanczos (full reorthogonalization).
// Dynamics scenario: quench from h_initial (polarised ground state) to h=0.
// Exact diagonal time evolution via exp(-iτH₀)|ψ⟩ is not available here because H is
// off-diagonal; we use Arnoldi-Krylov (m-step) for the time propagation instead.

#include "qx/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef M_PI
#  define M_PI 3.141592653589793238462643383279502884
#endif

namespace qx {
namespace {

using cx   = std::complex<double>;
using MatC = std::vector<cx>;

// ─────────────────────────────────────────────────────────────────────────────
// A.  Basis: Z-basis states in the Sz = m sector
// ─────────────────────────────────────────────────────────────────────────────

// Enumerate all N-bit strings with exactly n_down bits set (n_down = (N - 2*Sz) / 2).
static std::vector<int> spin_states(int N, int n_down) {
  std::vector<int> st;
  if (n_down < 0 || n_down > N) return st;
  if (n_down == 0) { st.push_back(0); return st; }
  // Gosper's hack
  int s = (1 << n_down) - 1;
  int lim = 1 << N;
  while (s < lim) {
    st.push_back(s);
    int c = s & (-s);
    int r = s + c;
    s = (((r ^ s) >> 2) / c) | r;
  }
  return st;
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  CSR Hamiltonian
// ─────────────────────────────────────────────────────────────────────────────

struct HCsr {
  int dim{0};
  std::vector<double> vals;
  std::vector<int>    cols;
  std::vector<int>    row_ptr;
};

static void hcsr_matvec(const HCsr& A, const std::vector<double>& x, std::vector<double>& y) {
  for (int r = 0; r < A.dim; ++r) {
    double s = 0.0;
    for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p)
      s += A.vals[p] * x[A.cols[p]];
    y[r] = s;
  }
}

// y = H * x  (real)
static void hcsr_matvec_cx(const HCsr& A, const std::vector<cx>& x, std::vector<cx>& y) {
  for (int r = 0; r < A.dim; ++r) {
    cx s{0.0, 0.0};
    for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p)
      s += A.vals[p] * x[A.cols[p]];
    y[r] = s;
  }
}

struct CooE { int row, col; double val; };

static HCsr coo_to_csr(std::vector<CooE> coo, int dim) {
  std::sort(coo.begin(), coo.end(), [](const CooE& a, const CooE& b){
    return a.row < b.row || (a.row == b.row && a.col < b.col);
  });
  std::vector<CooE> mg;
  mg.reserve(coo.size());
  for (auto& e : coo) {
    if (!mg.empty() && mg.back().row == e.row && mg.back().col == e.col)
      mg.back().val += e.val;
    else mg.push_back(e);
  }
  HCsr csr; csr.dim = dim;
  csr.row_ptr.resize(static_cast<std::size_t>(dim) + 1, 0);
  for (auto& e : mg) csr.row_ptr[e.row + 1]++;
  for (int r = 0; r < dim; ++r) csr.row_ptr[r + 1] += csr.row_ptr[r];
  csr.vals.resize(mg.size()); csr.cols.resize(mg.size());
  std::vector<int> pos(csr.row_ptr.begin(), csr.row_ptr.end() - 1);
  for (auto& e : mg) {
    int p = pos[e.row]++;
    csr.vals[p] = e.val; csr.cols[p] = e.col;
  }
  return csr;
}

static HCsr build_xxz_csr(int N, double J, double Jz, double h,
                            const std::vector<int>& states,
                            const std::map<int,int>& state_map) {
  int dim = static_cast<int>(states.size());
  std::vector<CooE> coo;
  coo.reserve(static_cast<std::size_t>(dim) * N);

  for (int idx = 0; idx < dim; ++idx) {
    int s = states[idx];
    // Diagonal: Jz/4 Σ Z_i Z_{i+1} − h/2 Σ Z_i
    double diag = 0.0;
    for (int i = 0; i < N - 1; ++i) {
      int zi = ((s >> i) & 1) ? -1 : 1;
      int zj = ((s >> (i + 1)) & 1) ? -1 : 1;
      diag += (Jz * 0.25) * zi * zj;
    }
    for (int i = 0; i < N; ++i) {
      int zi = ((s >> i) & 1) ? -1 : 1;
      diag -= (h * 0.5) * zi;
    }
    if (diag != 0.0) coo.push_back({idx, idx, diag});

    // Off-diagonal: J/2 (S+_i S-_{i+1} + S-_i S+_{i+1})
    //   = J/2 (flip pair where one is up one is down)
    for (int i = 0; i < N - 1; ++i) {
      int bi = 1 << i, bj = 1 << (i + 1);
      bool ui = !((s >> i) & 1);    // site i is up if bit i = 0
      bool uj = !((s >> (i + 1)) & 1);
      if (ui != uj) {  // one up, one down — can flip
        int s2 = s ^ bi ^ bj;
        auto it = state_map.find(s2);
        if (it != state_map.end()) {
          int idx2 = it->second;
          if (idx < idx2) {
            // Emit each pair once; partner visit will add the conjugate direction
            coo.push_back({idx, idx2, J * 0.5});
            coo.push_back({idx2, idx, J * 0.5});
          }
        }
      }
    }
  }
  return coo_to_csr(std::move(coo), dim);
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  Tridiagonal eigensolver (Wilkinson QR)
// ─────────────────────────────────────────────────────────────────────────────

static std::pair<std::vector<double>, std::vector<double>>
tridiag_eig_r(const std::vector<double>& alpha_in, const std::vector<double>& beta_in) {
  int k = static_cast<int>(alpha_in.size());
  std::vector<double> a = alpha_in, b = beta_in;
  std::vector<double> Q(static_cast<std::size_t>(k) * k, 0.0);
  for (int i = 0; i < k; ++i)
    Q[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * k] = 1.0;
  for (int iter = 0; iter < 30 * k; ++iter) {
    int n = k;
    while (n > 1 && std::abs(b[n-2]) < 1e-10*(std::abs(a[n-2])+std::abs(a[n-1]))) --n;
    if (n == 1) break;
    int m = n - 1;
    while (m > 0 && std::abs(b[m-1]) >= 1e-10*(std::abs(a[m-1])+std::abs(a[m]))) --m;
    double dw = (a[n-2]-a[n-1])*0.5, sd = dw>=0.0?1.0:-1.0;
    double mu = a[n-1] - b[n-2]*b[n-2]/(dw+sd*std::sqrt(dw*dw+b[n-2]*b[n-2]));
    double x = a[m]-mu, z = b[m];
    for (int i = m; i < n-1; ++i) {
      double rg = std::sqrt(x*x+z*z); if (rg<1e-300) break;
      double c=x/rg, s=z/rg;
      if (i>m) b[i-1]=rg;
      double ai=a[i],ai1=a[i+1],bi=b[i];
      a[i]=c*c*ai+2*c*s*bi+s*s*ai1; a[i+1]=s*s*ai-2*c*s*bi+c*c*ai1;
      b[i]=c*s*(ai1-ai)+(c*c-s*s)*bi;
      if (i<n-2){double bi1=b[i+1];b[i+1]=c*bi1;z=s*bi1;x=b[i];}
      for (int j=0;j<k;++j){
        double qi=Q[j+i*k],qi1=Q[j+(i+1)*k];
        Q[j+i*k]=c*qi+s*qi1; Q[j+(i+1)*k]=-s*qi+c*qi1;
      }
    }
  }
  std::vector<int> idx(k); std::iota(idx.begin(),idx.end(),0);
  std::sort(idx.begin(),idx.end(),[&](int i,int j){return a[i]<a[j];});
  std::vector<double> ev(k), evec(static_cast<std::size_t>(k)*k);
  for (int j=0;j<k;++j){
    ev[j]=a[idx[j]];
    for (int r=0;r<k;++r)
      evec[static_cast<std::size_t>(r)+static_cast<std::size_t>(j)*k]=
          Q[static_cast<std::size_t>(r)+static_cast<std::size_t>(idx[j])*k];
  }
  return {ev, evec};
}

// ─────────────────────────────────────────────────────────────────────────────
// D.  Lanczos eigensolver (full reorthogonalization)
// ─────────────────────────────────────────────────────────────────────────────

struct EigPair { double eval{0.0}; std::vector<double> evec; };

static std::pair<EigPair, EigPair> heis_lanczos(const HCsr& H, int seed=42) {
  int dim = H.dim; if (dim <= 0) { EigPair b; return {b,b}; }
  int km = std::min({200, dim/4+10, dim}); km = std::max(km, 2);
  std::vector<double> v(dim);
  { std::uint64_t lcg = static_cast<std::uint64_t>(seed)*6364136223846793005ULL+1;
    for (int i=0;i<dim;++i){lcg=lcg*6364136223846793005ULL+1442695040888963407ULL;
      v[i]=static_cast<double>(static_cast<std::int64_t>(lcg>>33))/static_cast<double>(1LL<<30);}
  }
  double nn=0.0; for (auto x:v) nn+=x*x; nn=std::sqrt(nn);
  for (auto& x:v) x/=nn;
  std::vector<double> alpha, beta;
  std::vector<std::vector<double>> Q; Q.reserve(km+1); Q.push_back(v);
  std::vector<double> w(dim); double bp=0.0; int ka=0;
  for (int j=0;j<km;++j) {
    hcsr_matvec(H,Q[j],w);
    double aj=0.0; for (int i=0;i<dim;++i) aj+=Q[j][i]*w[i]; alpha.push_back(aj);
    for (int i=0;i<dim;++i) w[i]-=aj*Q[j][i];
    if (j>0) for (int i=0;i<dim;++i) w[i]-=bp*Q[j-1][i];
    for (int pass=0;pass<2;++pass)
      for (int l=0;l<=j;++l){double d=0.0;
        for (int i=0;i<dim;++i) d+=Q[l][i]*w[i];
        for (int i=0;i<dim;++i) w[i]-=d*Q[l][i];}
    double bj=0.0; for (auto x:w) bj+=x*x; bj=std::sqrt(bj);
    ka=j+1; if (bj<1e-12) break;
    beta.push_back(bj); bp=bj;
    std::vector<double> qn(dim);
    for (int i=0;i<dim;++i) qn[i]=w[i]/bj;
    Q.push_back(qn);
  }
  auto [ev,ec]=tridiag_eig_r(alpha,beta);
  int k=ka;
  auto ritz=[&](int col)->EigPair{
    col=std::min(col,k-1);
    EigPair ep; ep.eval=ev[col]; ep.evec.assign(dim,0.0);
    for (int j=0;j<k;++j){
      double y=ec[static_cast<std::size_t>(j)+static_cast<std::size_t>(col)*k];
      for (int i=0;i<dim;++i) ep.evec[i]+=y*Q[j][i];
    }
    double n2=0.0; for (auto x:ep.evec) n2+=x*x; n2=std::sqrt(n2);
    if (n2>1e-300) for (auto& x:ep.evec) x/=n2;
    return ep;
  };
  return {ritz(0), ritz(1)};
}

// ─────────────────────────────────────────────────────────────────────────────
// E.  Arnoldi-Krylov time propagator  exp(-i dt H)|ψ⟩
// ─────────────────────────────────────────────────────────────────────────────

// Dense complex matrix exponential (Taylor + squaring), same as floquet.cpp
static MatC dense_expm_c(MatC A, int m) {
  double n1=0.0;
  for (int c=0;c<m;++c){double cs=0.0;
    for (int r=0;r<m;++r) cs+=std::abs(A[static_cast<std::size_t>(r)+static_cast<std::size_t>(c)*m]);
    n1=std::max(n1,cs);}
  int s=1; if (n1>1.0) s=static_cast<int>(std::ceil(n1));
  MatC As(static_cast<std::size_t>(m)*m);
  for (std::size_t i=0;i<static_cast<std::size_t>(m)*m;++i) As[i]=A[i]/static_cast<double>(s);
  MatC E(static_cast<std::size_t>(m)*m, cx{0,0});
  for (int i=0;i<m;++i) E[static_cast<std::size_t>(i)+static_cast<std::size_t>(i)*m]={1,0};
  MatC Ak=E; double fact=1.0;
  for (int k=1;k<=14;++k){
    fact/=static_cast<double>(k);
    MatC Ak2(static_cast<std::size_t>(m)*m,cx{0,0});
    for (int r=0;r<m;++r) for (int c=0;c<m;++c){
      cx acc{0,0};
      for (int l=0;l<m;++l)
        acc+=Ak[static_cast<std::size_t>(r)+static_cast<std::size_t>(l)*m]*
             As[static_cast<std::size_t>(l)+static_cast<std::size_t>(c)*m];
      Ak2[static_cast<std::size_t>(r)+static_cast<std::size_t>(c)*m]=acc;}
    Ak=Ak2;
    for (std::size_t i=0;i<static_cast<std::size_t>(m)*m;++i) E[i]+=fact*Ak[i];
  }
  if (s>1){
    MatC Eo=E;
    for (int sq=1;sq<s;++sq){
      MatC E2(static_cast<std::size_t>(m)*m,cx{0,0});
      for (int r=0;r<m;++r) for (int c=0;c<m;++c){
        cx acc{0,0};
        for (int l=0;l<m;++l)
          acc+=E[static_cast<std::size_t>(r)+static_cast<std::size_t>(l)*m]*
               Eo[static_cast<std::size_t>(l)+static_cast<std::size_t>(c)*m];
        E2[static_cast<std::size_t>(r)+static_cast<std::size_t>(c)*m]=acc;}
      E=E2;}}
  return E;
}

static std::vector<cx> arnoldi_step(const HCsr& H, const std::vector<cx>& psi,
                                     double dt, int mv=20) {
  int dim=H.dim; mv=std::min(mv,dim);
  double beta=0.0; for (auto& z:psi) beta+=std::norm(z); beta=std::sqrt(beta);
  if (beta<1e-300) return psi;
  std::vector<std::vector<cx>> V; V.reserve(mv+1);
  {std::vector<cx> v0(dim); for (int i=0;i<dim;++i) v0[i]=psi[i]/beta; V.push_back(v0);}
  MatC Hm(static_cast<std::size_t>(mv)*mv, cx{0,0});
  std::vector<cx> w(dim); int ma=mv;
  for (int j=0;j<mv;++j){
    hcsr_matvec_cx(H,V[j],w);
    for (int l=0;l<=j;++l){
      cx d{0,0}; for (int i=0;i<dim;++i) d+=std::conj(V[l][i])*w[i];
      Hm[static_cast<std::size_t>(l)+static_cast<std::size_t>(j)*mv]=d;
      for (int i=0;i<dim;++i) w[i]-=d*V[l][i];
    }
    double h_next=0.0; for (auto& z:w) h_next+=std::norm(z); h_next=std::sqrt(h_next);
    ma=j+1; if (h_next<1e-12) break;
    Hm[static_cast<std::size_t>(j+1)+static_cast<std::size_t>(j)*mv]={h_next,0};
    std::vector<cx> vnew(dim); for (int i=0;i<dim;++i) vnew[i]=w[i]/h_next;
    V.push_back(vnew);
  }
  // Build e1 and compute exp(-i dt Hm) e1
  MatC Hscaled(static_cast<std::size_t>(ma)*ma);
  for (int r=0;r<ma;++r) for (int c=0;c<ma;++c)
    Hscaled[static_cast<std::size_t>(r)+static_cast<std::size_t>(c)*ma]=
        cx{0,-dt}*Hm[static_cast<std::size_t>(r)+static_cast<std::size_t>(c)*ma];
  MatC eH = dense_expm_c(Hscaled, ma);
  // y = beta * V * eH[:,0]
  std::vector<cx> y(dim,cx{0,0});
  for (int j=0;j<ma;++j){
    cx coeff = beta * eH[static_cast<std::size_t>(j)];
    for (int i=0;i<dim;++i) y[i]+=coeff*V[j][i];
  }
  return y;
}

// ─────────────────────────────────────────────────────────────────────────────
// F.  Bipartite entropy (complex M†M Householder + tridiag)
// ─────────────────────────────────────────────────────────────────────────────

static std::pair<std::vector<double>,std::vector<double>>
hh_herm(std::vector<cx>& A, int n) {
  for (int k=0;k<n-2;++k){
    double sg=0.0;
    for (int i=k+1;i<n;++i) sg+=std::norm(A[static_cast<std::size_t>(i)+static_cast<std::size_t>(k)*n]);
    double ah=std::sqrt(sg);
    cx x1=A[static_cast<std::size_t>(k+1)+static_cast<std::size_t>(k)*n];
    cx ph=(std::abs(x1)<1e-300)?cx{1,0}:x1/std::abs(x1);
    double r2=sg-std::abs(x1)*ah; if (r2<1e-30) continue;
    A[static_cast<std::size_t>(k+1)+static_cast<std::size_t>(k)*n]-=ph*ah;
    std::vector<cx> v(n-k-1);
    for (int i=0;i<n-k-1;++i) v[i]=A[static_cast<std::size_t>(k+1+i)+static_cast<std::size_t>(k)*n];
    std::vector<cx> q(n-k-1,cx{0,0});
    for (int r=0;r<n-k-1;++r) for (int c=0;c<n-k-1;++c)
      q[r]+=A[static_cast<std::size_t>(k+1+r)+static_cast<std::size_t>(k+1+c)*n]*v[c];
    cx vHq{0,0}; for (int i=0;i<n-k-1;++i) vHq+=std::conj(v[i])*q[i];
    std::vector<cx> w(n-k-1);
    for (int i=0;i<n-k-1;++i) w[i]=q[i]/r2-(vHq.real()/(2.0*r2*r2))*v[i];
    for (int r=0;r<n-k-1;++r) for (int c=0;c<n-k-1;++c)
      A[static_cast<std::size_t>(k+1+r)+static_cast<std::size_t>(k+1+c)*n]-=
          v[r]*std::conj(w[c])+w[r]*std::conj(v[c]);
    cx off=ph*ah;
    A[static_cast<std::size_t>(k+1)+static_cast<std::size_t>(k)*n]=off;
    A[static_cast<std::size_t>(k)+static_cast<std::size_t>(k+1)*n]=std::conj(off);
  }
  std::vector<double> dg(n),sb(n-1);
  for (int i=0;i<n;++i) dg[i]=A[static_cast<std::size_t>(i)+static_cast<std::size_t>(i)*n].real();
  for (int i=0;i<n-1;++i) sb[i]=std::abs(A[static_cast<std::size_t>(i+1)+static_cast<std::size_t>(i)*n]);
  return {dg,sb};
}

static std::vector<double> tridiag_evals(std::vector<double> a, std::vector<double> b) {
  int k=static_cast<int>(a.size());
  for (int it=0;it<30*k;++it){
    int n=k;
    while (n>1&&std::abs(b[n-2])<1e-10*(std::abs(a[n-2])+std::abs(a[n-1]))) --n;
    if (n==1) break;
    int m=n-1;
    while (m>0&&std::abs(b[m-1])>=1e-10*(std::abs(a[m-1])+std::abs(a[m]))) --m;
    double dw=(a[n-2]-a[n-1])*0.5,sd=dw>=0?1.0:-1.0;
    double mu=a[n-1]-b[n-2]*b[n-2]/(dw+sd*std::sqrt(dw*dw+b[n-2]*b[n-2]));
    double x=a[m]-mu,z=b[m];
    for (int i=m;i<n-1;++i){
      double rg=std::sqrt(x*x+z*z); if (rg<1e-300) break;
      double c=x/rg,s=z/rg;
      if (i>m) b[i-1]=rg;
      double ai=a[i],ai1=a[i+1],bi=b[i];
      a[i]=c*c*ai+2*c*s*bi+s*s*ai1; a[i+1]=s*s*ai-2*c*s*bi+c*c*ai1;
      b[i]=c*s*(ai1-ai)+(c*c-s*s)*bi;
      if (i<n-2){double bi1=b[i+1];b[i+1]=c*bi1;z=s*bi1;x=b[i];}
    }
  }
  return a;
}

static double heisen_entropy(const std::vector<cx>& psi, int N) {
  // Map full Hilbert space (2^N) psi into dimA x dimB, cut at N/2
  int dimA=1<<(N/2), dimB=1<<(N-N/2);
  std::vector<cx> MtM(static_cast<std::size_t>(dimA)*dimA, cx{0,0});
  for (int a1=0;a1<dimA;++a1)
    for (int a2=a1;a2<dimA;++a2){
      cx v{0,0};
      for (int b=0;b<dimB;++b){
        cx p1=psi[static_cast<std::size_t>(a1)*dimB+b];
        cx p2=psi[static_cast<std::size_t>(a2)*dimB+b];
        v+=std::conj(p1)*p2;
      }
      MtM[static_cast<std::size_t>(a1)+static_cast<std::size_t>(a2)*dimA]=v;
      MtM[static_cast<std::size_t>(a2)+static_cast<std::size_t>(a1)*dimA]=std::conj(v);
    }
  if (dimA==1){double p=MtM[0].real();return p>1e-14?-p*std::log(p):0.0;}
  auto [dg,sb]=hh_herm(MtM,dimA);
  auto ev=tridiag_evals(dg,sb);
  double tr=0.0; for (auto e:ev) if (e>0) tr+=e;
  if (tr<1e-28) return 0.0;
  double S=0.0;
  for (auto e:ev){if (e<=0) continue; double p=e/tr; if (p>1e-14) S-=p*std::log(p);}
  return S;
}

// ─────────────────────────────────────────────────────────────────────────────
// G.  Observable computation
// ─────────────────────────────────────────────────────────────────────────────

// Compute <psi|S_0^alpha S_r^alpha|psi> from ground state (real psi in Sz=0 sector)
// Approximated via the sector-basis psi.
// For spin-spin correlator we embed back into full 2^N space.
static std::vector<double> spin_spin_corr(int N, const std::vector<int>& states,
                                            const std::vector<double>& psi) {
  int dim = static_cast<int>(states.size());
  std::vector<double> corr(N, 0.0);
  std::map<int,int> smap;
  for (int i=0;i<dim;++i) smap[states[i]]=i;

  // ZZ part: (1/4) Σ_s |ψ_s|² Z_0 Z_r
  for (int idx=0;idx<dim;++idx){
    int s=states[idx]; double p=psi[idx]*psi[idx];
    double z0 = ((s>>0)&1)?-1.0:1.0;
    for (int r=0;r<N;++r){
      double zr = ((s>>r)&1)?-1.0:1.0;
      corr[r]+=0.25*p*z0*zr;
    }
  }
  // XX+YY part: (1/4)(S+_0 S-_r + S-_0 S+_r) → flip bits 0 and r
  for (int idx=0;idx<dim;++idx){
    int s=states[idx]; double pv=psi[idx];
    bool u0=!((s>>0)&1);
    for (int r=1;r<N;++r){
      bool ur=!((s>>r)&1);
      if (u0!=ur){ // one up one down at 0 and r
        int s2=s^(1<<0)^(1<<r);
        auto it=smap.find(s2);
        if (it!=smap.end()) corr[r]+=0.5*pv*psi[it->second];
      }
    }
  }
  return corr;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

HeisenbergResult run_heisenberg(const HeisenbergOptions& opts) {
  auto t0=std::chrono::steady_clock::now();
  HeisenbergResult res;
  try {
    int N=static_cast<int>(opts.n_sites);
    if (N<2||N>16) throw std::runtime_error("n_sites must be in [2, 16]");

    // Use Sz=0 sector (n_down = N/2) for even N; else use (N-1)/2 down-spins
    int n_down = N/2;
    auto states = spin_states(N, n_down);
    std::map<int,int> smap;
    for (int i=0;i<static_cast<int>(states.size());++i) smap[states[i]]=i;

    res.n_sites=opts.n_sites; res.J=opts.J; res.Jz=opts.Jz; res.h=opts.h;

    auto solve=[&](double J_v, double Jz_v, double h_v)
        -> std::pair<EigPair,EigPair>{
      auto H=build_xxz_csr(N,J_v,Jz_v,h_v,states,smap);
      return heis_lanczos(H);
    };

    if (opts.scenario=="sweep"){
      int ns=static_cast<int>(opts.n_sweep);
      for (int si=0;si<ns;++si){
        double Jz_v=opts.Jz_min+(opts.Jz_max-opts.Jz_min)*si/std::max(ns-1,1);
        auto [gs,es]=solve(opts.J,Jz_v,opts.h);
        auto corr=spin_spin_corr(N,states,gs.evec);
        double spi=0.0;
        for (int r=0;r<N;++r) spi+=((r%2==0)?1.0:-1.0)*corr[r];
        HeisenbergSweepPoint pt;
        pt.Jz=Jz_v; pt.ground_energy=gs.eval; pt.energy_gap=es.eval-gs.eval;
        pt.structure_pi=spi/N;
        res.sweep.push_back(pt);
      }
    }

    if (opts.scenario=="dynamics"){
      // Prepare polarised ground state under h_initial >> J
      auto [gs_i,_es]=solve(opts.J,opts.Jz,opts.h_initial);
      // Embed ground state in full 2^N space for entropy
      std::vector<cx> psi_full(1<<N, cx{0,0});
      for (int idx=0;idx<static_cast<int>(states.size());++idx)
        psi_full[states[idx]]=gs_i.evec[idx];

      // Post-quench Hamiltonian (h=0)
      auto H_final=build_xxz_csr(N,opts.J,opts.Jz,0.0,states,smap);

      // Sector-space psi for propagation
      std::vector<cx> psi_sec(states.size());
      for (int idx=0;idx<static_cast<int>(states.size());++idx)
        psi_sec[idx]=gs_i.evec[idx];

      auto record=[&](double t){
        HeisenbergFrame f; f.time=t;
        // Sz total = 0 in Sz=0 sector; measure per-site instead
        f.Sz_site.assign(N,0.0);
        for (int idx=0;idx<static_cast<int>(states.size());++idx){
          double p=std::norm(psi_sec[idx]);
          int s=states[idx];
          for (int i=0;i<N;++i)
            f.Sz_site[i]+=p*(((s>>i)&1)?-0.5:0.5);
        }
        f.Sz_total=0.0; for (auto x:f.Sz_site) f.Sz_total+=x;
        double en=0.0;
        {std::vector<cx> Hpsi(states.size(),cx{0,0});
         hcsr_matvec_cx(H_final,psi_sec,Hpsi);
         for (std::size_t i=0;i<psi_sec.size();++i) en+=std::real(std::conj(psi_sec[i])*Hpsi[i]);}
        f.energy=en;
        // entropy in full space
        for (int idx=0;idx<static_cast<int>(states.size());++idx)
          psi_full[states[idx]]=psi_sec[idx];
        f.entropy=heisen_entropy(psi_full,N);
        res.frames.push_back(f);
      };
      record(0.0);
      for (std::uint32_t st=1;st<=opts.n_steps;++st){
        psi_sec=arnoldi_step(H_final,psi_sec,opts.dt);
        record(static_cast<double>(st)*opts.dt);
      }
    }

    // Ground state observables
    {
      auto [gs,es]=solve(opts.J,opts.Jz,opts.h);
      res.ground_energy=gs.eval; res.energy_gap=es.eval-gs.eval;
      res.spin_corr=spin_spin_corr(N,states,gs.evec);
      double spi=0.0;
      for (int r=0;r<N;++r) spi+=((r%2==0)?1.0:-1.0)*res.spin_corr[r];
      res.structure_pi=spi/N;
      res.Sz_site.assign(N,0.0);
      for (int idx=0;idx<static_cast<int>(states.size());++idx){
        double p=gs.evec[idx]*gs.evec[idx]; int s=states[idx];
        for (int i=0;i<N;++i)
          res.Sz_site[i]+=p*(((s>>i)&1)?-0.5:0.5);
      }
    }
    res.ok=true;
  } catch (const std::exception& ex){
    res.ok=false; res.message=ex.what();
  }
  auto t1=std::chrono::steady_clock::now();
  res.elapsed_ms=std::chrono::duration<double,std::milli>(t1-t0).count();
  return res;
}

}  // namespace qx
