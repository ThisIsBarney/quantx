#pragma once
// SU(2)-symmetric DMRG: reduced-tensor data structures (stage 1) and the SU(2)
// Heisenberg MPO as a spherical-tensor operator (stage 2).
//
// Conventions (doubled-integer "two-j" throughout, see su2.hpp):
//   - Spins/multiplets are labelled by tS = 2S >= 0.
//   - A bond stores (tS, deg): the spin label and its *reduced* multiplicity
//     (the number of times the multiplet S appears). The full U(1) dimension it
//     stands for is sum_S deg_S * (2S+1).
//   - The physical spin-1/2 site has tS = 1.
//
// Operator convention: the on-site operator is the Pauli vector sigma (so that
// sigma.sigma reproduces the Heisenberg bond used by the U(1) code). Reduced
// matrix elements use the Wigner-Eckart form
//     <s' m'| T^k_q |s m> = <s m; k q | s' m'> * <s'||T^k||s>,
// giving  <1/2||I||1/2> = 1  and  <1/2||sigma||1/2> = -sqrt(3).

#include "su2.hpp"
#include <vector>

namespace qx_su2 {

// -- Stage 1: reduced-tensor bond -------------------------------------------
struct SpinBond {
    std::vector<int> tS;    // doubled spin labels, sorted ascending
    std::vector<int> deg;   // reduced multiplicity of each multiplet
    int n() const { return (int)tS.size(); }
    int deg_of(int s) const {
        for (int i = 0; i < n(); ++i) if (tS[i] == s) return deg[i];
        return 0;
    }
};

// -- Stage 2: SU(2) Heisenberg MPO ------------------------------------------
//
// MPO bond states (multiplets), in order:
//   0 "begin" (tS=0)  no operator emitted yet      (left boundary)
//   1 "op"    (tS=2)  one spin operator in transit (the rank-1 sigma)
//   2 "done"  (tS=0)  the sigma.sigma term finished (right boundary)
struct SU2Heisenberg {
    static constexpr double R_I     = 1.0;                 // <1/2||I||1/2>
    static double R_SIGMA() { return -std::sqrt(3.0); }    // <1/2||sigma||1/2>

    // MPO bond states: spin of each state.
    static constexpr int N_BOND = 3;
    static constexpr int BEGIN = 0, OP = 1, DONE = 2;
    static int bond_tS(int b) { return (b == OP) ? 2 : 0; }

    // A transition contributes operator T^k (k from the bond spins) with reduced r.
    struct Trans { int a, b, tk; double r; };
    static std::vector<Trans> transitions(double J) {
        // The operator bond (spin S_op = 1) closes a scalar product:
        //   A^(k).B^(k) = (-1)^k sqrt(2k+1) (A (x) B)^(0).
        // A bare bond contraction yields the coupled (A(x)B)^0; folding this
        // closure factor into the ABSORBING (op->done) operator makes the raw
        // MPO contraction reproduce J*sigma.sigma directly -- so the SU(2)
        // environment/H_eff contractions (which replay the raw MPO) get the
        // physical energy with no extra bookkeeping.
        const int S_op = 1;
        const double closure = ((S_op & 1) ? -1.0 : 1.0) * std::sqrt(2.0 * S_op + 1.0);
        return {
            {BEGIN, BEGIN, 0, R_I},                       // I
            {BEGIN, OP,    2, J * R_SIGMA()},              // emit J*sigma
            {OP,    DONE,  2, R_SIGMA() * closure},        // absorb sigma (+ closure)
            {DONE,  DONE,  0, R_I},                        // I
        };
    }

    // Full matrix element of one transition: bond a (comp tMA) -> bond b (comp tMB),
    // physical bra tm_bra, ket tm_ket. q = tMB - tMA = tm_bra - tm_ket.
    static double elem(const Trans& t, int tMA, int tMB, int tm_bra, int tm_ket) {
        int tq = tMB - tMA;
        if (tq != tm_bra - tm_ket) return 0.0;
        double cg_bond = clebsch_gordan(bond_tS(t.a), tMA, t.tk, tq, bond_tS(t.b), tMB);
        if (cg_bond == 0.0) return 0.0;
        double cg_phys = clebsch_gordan(1, tm_ket, t.tk, tq, 1, tm_bra);
        return cg_bond * cg_phys * t.r;
    }
};

} // namespace qx_su2
