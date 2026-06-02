// Validation of block-sparse (Sz-symmetric) TDVP2 vs exact diagonalization:
// real-time Neel quench staggered magnetization, and energy conservation.
#include "dmrg_internal.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <complex>
#include <vector>

using namespace qx_dmrg;
using Cm = Eigen::MatrixXcd;
using Cv = Eigen::VectorXcd;
using Cx = std::complex<double>;

static int failures = 0;
static void check(const char* name, double err, double tol) {
    bool ok = err < tol;
    if (!ok) ++failures;
    printf("  %-44s err=%.3e  %s\n", name, err, ok ? "OK" : "FAIL");
}

static Cm kron(const Cm& A, const Cm& B) {
    Cm C(A.rows()*B.rows(), A.cols()*B.cols());
    for (int i=0;i<A.rows();++i) for (int j=0;j<A.cols();++j)
        C.block(i*B.rows(), j*B.cols(), B.rows(), B.cols()) = A(i,j)*B;
    return C;
}

int main() {
    const int n = 6;
    Cm X(2,2); X<<0,1,1,0;
    Cm Y(2,2); Y<<Cx(0,0),Cx(0,-1),Cx(0,1),Cx(0,0);
    Cm Z(2,2); Z<<1,0,0,-1;
    Cm I2 = Cm::Identity(2,2);
    auto emb=[&](const Cm& P,int i){ Cm m(1,1); m<<1; for(int k=0;k<n;++k) m=kron(m,(k==i)?P:I2); return m; };
    int d=1<<n;
    Cm H=Cm::Zero(d,d);
    for(int i=0;i+1<n;++i) for(const Cm* P:{&X,&Y,&Z}) H+=emb(*P,i)*emb(*P,i+1);
    Eigen::SelfAdjointEigenSolver<Cm> es(H);
    Cv psi0=Cv::Zero(d); int idx=0; for(int i=0;i<n;++i) idx=idx*2+(i%2==0?0:1); psi0(idx)=1;
    std::vector<Cm> Zs; for(int i=0;i<n;++i) Zs.push_back(emb(Z,i));
    auto stag_exact=[&](double t){ Cv c=es.eigenvectors().adjoint()*psi0;
        for(int k=0;k<d;++k) c(k)*=std::exp(Cx(0,-t)*es.eigenvalues()(k));
        Cv ps=es.eigenvectors()*c;
        double m=0; for(int i=0;i<n;++i) m+=((i%2==0)?1.0:-1.0)*std::real(ps.dot(Zs[i]*ps)); return m/n; };

    auto mpo = build_heisenberg_mpo(n, 1.0);
    double dt=0.02; int steps=25;
    auto r = run_tdvp2_blocks(mpo, n, dt, steps, /*chi_max*/16, 1e-12, /*imag*/false);

    printf("Block-sparse TDVP2 Neel quench (n=%d):\n", n);
    double maxerr=0;
    for (size_t k=0;k<r.times.size();++k) maxerr=std::max(maxerr, std::abs(r.stag[k]-stag_exact(r.times[k])));
    check("staggered magnetization vs exact", maxerr, 1e-3);
    // energy conservation: Neel energy under sigma.sigma = -(n-1)
    check("energy conserved (= -(n-1))", std::abs(r.energy - (-(n-1.0))), 1e-6);
    printf("  max bond reached: %d\n", r.max_bond);

    // imaginary-time -> ground state
    auto g = run_tdvp2_blocks(mpo, n, 0.1, 200, 16, 1e-12, /*imag*/true);
    check("imag-time -> ground vs exact", std::abs(g.energy - es.eigenvalues()(0)), 1e-5);

    printf("\n%s\n", failures==0 ? "ALL PASSED" : "SOME FAILED");
    return failures==0 ? 0 : 1;
}
