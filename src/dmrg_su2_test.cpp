// Validation of the SU(2) Heisenberg MPO (stage 2): expand the reduced
// spherical-tensor MPO to full operators and verify the two-site contraction
// reproduces the Heisenberg bond  J * sigma.sigma.
#include "dmrg_su2.hpp"
#include <array>
#include <cstdio>
#include <cmath>

using namespace qx_su2;

static int failures = 0;
static void check(const char* name, double got, double want, double tol = 1e-12) {
    bool ok = std::abs(got - want) < tol;
    if (!ok) ++failures;
    printf("  %-40s got=%+.10f want=%+.10f  %s\n", name, got, want, ok ? "OK" : "FAIL");
}

// Physical basis: index 0 = up (tm=+1), 1 = down (tm=-1).
static int tm_of(int i) { return i == 0 ? +1 : -1; }

int main() {
    const double J = 1.0;
    auto trans = SU2Heisenberg::transitions(J);
    // Find the two operator-carrying transitions.
    const SU2Heisenberg::Trans *emit = nullptr, *absorb = nullptr;
    for (auto& t : trans) {
        if (t.a == SU2Heisenberg::BEGIN && t.b == SU2Heisenberg::OP) emit = &t;
        if (t.a == SU2Heisenberg::OP && t.b == SU2Heisenberg::DONE) absorb = &t;
    }

    // Scalar-product closure factor for the transported operator multiplet of
    // spin k = S_op:  A^(k).B^(k) = (-1)^k sqrt(2k+1) (A (x) B)^(0). The bare
    // bond sum below produces the coupled (A(x)B)^0; this factor turns it into
    // the physical scalar product. In the full SU(2) DMRG this emerges from the
    // 6j/9j recoupling of the environment / H_eff contractions.
    const int S_op = SU2Heisenberg::bond_tS(SU2Heisenberg::OP) / 2;  // = 1
    const double closure = std::pow(-1.0, S_op) * std::sqrt(2.0 * S_op + 1.0);  // -sqrt(3)

    // Two-site Hamiltonian H[(m0',m1')][(m0,m1)], 4x4. Left bond fixed to begin,
    // right bond fixed to done; the shared bond runs over the op multiplet (tM in {-2,0,2}).
    std::array<std::array<double, 4>, 4> H{};
    for (int m0p = 0; m0p < 2; ++m0p)
    for (int m1p = 0; m1p < 2; ++m1p)
    for (int m0 = 0; m0 < 2; ++m0)
    for (int m1 = 0; m1 < 2; ++m1) {
        double v = 0.0;
        for (int tM = -2; tM <= 2; tM += 2) {
            double w0 = SU2Heisenberg::elem(*emit,   /*tMA*/0, /*tMB*/tM, tm_of(m0p), tm_of(m0));
            double w1 = SU2Heisenberg::elem(*absorb, /*tMA*/tM, /*tMB*/0, tm_of(m1p), tm_of(m1));
            v += w0 * w1;
        }
        H[m0p * 2 + m1p][m0 * 2 + m1] = closure * v;
    }

    // Reference J * sigma.sigma in basis {uu, ud, du, dd}.
    std::array<std::array<double, 4>, 4> Ref{};
    Ref[0][0] = J;  Ref[3][3] = J;            // uu, dd : +J
    Ref[1][1] = -J; Ref[2][2] = -J;           // ud, du : -J (sigma_z sigma_z)
    Ref[1][2] = 2 * J; Ref[2][1] = 2 * J;     // spin flip 2J

    printf("SU(2) Heisenberg MPO -> two-site bond == J*sigma.sigma:\n");
    const char* lbl[4] = {"uu", "ud", "du", "dd"};
    double maxerr = 0;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            maxerr = std::max(maxerr, std::abs(H[i][j] - Ref[i][j]));
    check("two-site H max element error", maxerr, 0.0, 1e-12);
    // Show the matrix for transparency.
    for (int i = 0; i < 4; ++i) {
        printf("    %s : ", lbl[i]);
        for (int j = 0; j < 4; ++j) printf("%+.4f ", H[i][j]);
        printf("\n");
    }

    // Total-spin check: H eigenvalues should be -3J (singlet) and +J (triplet, x3).
    // Trace = sum of eigenvalues = -3J + 3J = 0; and Tr(H^2) = 9J^2 + 3J^2 = 12J^2.
    double tr = 0, tr2 = 0;
    for (int i = 0; i < 4; ++i) {
        tr += H[i][i];
        for (int j = 0; j < 4; ++j) tr2 += H[i][j] * H[j][i];
    }
    check("Tr(H)", tr, 0.0, 1e-12);
    check("Tr(H^2)", tr2, 12.0 * J * J, 1e-12);

    printf("\n%s\n", failures == 0 ? "ALL PASSED" : "SOME FAILED");
    return failures == 0 ? 0 : 1;
}
