#include "dmrg_internal.hpp"
#include "tdvp.hpp"            // qx_tdvp::expmv (Krylov matrix exponential)

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_thread_num() { return 0; }
static inline int omp_get_max_threads() { return 1; }
#endif

namespace qx_dmrg {

// Scalar-templated dense matrix/vector aliases. The block-sparse DMRG core below
// is templated on the scalar type S: it runs in real (double) arithmetic for real
// Hamiltonians (e.g. Heisenberg) -- ~4x faster than complex -- and in complex
// (std::complex<double>) arithmetic otherwise. run_dmrg_blocks dispatches on whether
// the MPO is real.
template<typename S> using MatX = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
template<typename S> using VecX = Eigen::Matrix<S, Eigen::Dynamic, 1>;

// Matvec profiling counter (shared across instantiations).
static long long g_block_matvecs = 0;

// TEBD phase profiling (set env QX_TEBD_PROFILE=1). Seconds per phase.
static double g_tebd_theta = 0, g_tebd_gate = 0, g_tebd_svd = 0, g_tebd_recomb = 0;
static long long g_tebd_maxblk = 0, g_tebd_nsvd = 0;

// =============================================================================
// Sz-conserving block-sparse DMRG  (M179)
// =============================================================================
//
// For any Hamiltonian that conserves total Sz (e.g. Heisenberg XXX), the MPS
// and environment tensors are block-diagonal in the Sz quantum number.
// Exploiting this reduces the dominant O(chi^3) DGEMM to blocks of size O((chi/k)^3)
// where k = sqrt(number of reachable sectors), giving a 2-4x wall-time speedup.
//
// Key conventions
// ---------------
// QN = accumulated Sz in sigma-units (up=+1, dn=-1) of the LEFT part of the chain.
// At bond i (right side of site i): QN = sum_{j<=i} s_j.
//
// MPS block layout
//   block_A[i][q_L] = Mat(dL, dR_up + dR_dn)
//     q_L    = left-bond QN for site i
//     rows   = dL = bond_dim[i][q_L]
//     cols   = [up block: bond_dim[i+1][q_L+1] | dn block: bond_dim[i+1][q_L-1]]
//
// Two-site theta block layout (indexed by q_mid = QN of bond between sites i, i+1)
//   theta[q_mid] = Mat(n_row, n_col)
//     rows = [s1=up: dL(q_mid-1) rows from q_L=q_mid-1 | s1=dn: dL(q_mid+1) from q_L=q_mid+1]
//     cols = [s2=up: dR2(q_mid+1) | s2=dn: dR2(q_mid-1)]
//
// Environment block layout
//   BlockEnv.blocks[b][q_ket] = Mat(dim_bra, dim_ket)
//     bra QN = q_ket + DELTA_QN[b],  DELTA_QN = {0,+2,-2,0,0} for Heisenberg
// =============================================================================

// -- BondSpace ----------------------------------------------------------------

struct BondSpace {
    std::vector<int> qns;      // sorted unique QN values
    std::vector<int> dims;     // dim of each sector
    std::vector<int> offsets;  // cumulative offset in a dense vector
    int total_dim = 0;

    int n_sectors() const { return (int)qns.size(); }

    int find_idx(int q) const {
        auto it = std::lower_bound(qns.begin(), qns.end(), q);
        if (it == qns.end() || *it != q) return -1;
        return (int)(it - qns.begin());
    }
    int dim(int q) const { int i = find_idx(q); return i >= 0 ? dims[i] : 0; }

    // Build from (sorted) list of (qn, dim) pairs
    static BondSpace from_pairs(const std::vector<std::pair<int,int>>& pairs) {
        BondSpace bs;
        int acc = 0;
        for (auto& [q,d] : pairs) {
            if (d <= 0) continue;
            bs.qns.push_back(q);
            bs.dims.push_back(d);
            bs.offsets.push_back(acc);
            acc += d;
        }
        bs.total_dim = acc;
        return bs;
    }
};

// Build the BondSpace at bond i for a length-n chain, total Sz=0, max_bond chi.
// Distributes chi states proportionally to the number of paths through each sector.
static BondSpace make_heisenberg_bond_space(int bond_i, int n, int max_bond) {
    int sites_L = bond_i + 1;   // sites on the left of this bond
    int sites_R = n - sites_L;  // sites on the right

    // Enumerate reachable QN sectors. Path counts are only used as relative weights
    // for the proportional allocation, so use double to avoid integer overflow
    // (binom(100,50)^2 ~ 1e29 overflows long long and corrupts the bond dims).
    auto binom = [](int nn, int kk) -> double {
        if (kk < 0 || kk > nn) return 0.0;
        double r = 1.0;
        for (int j = 0; j < kk; ++j) r = r * (nn - j) / (j + 1);
        return r;
    };

    std::vector<std::pair<int,double>> sectors;
    int qn_max = std::min(sites_L, sites_R);
    for (int q = -qn_max; q <= qn_max; q += 2) {
        int ups_L = (sites_L + q) / 2;
        int ups_R = (sites_R - q) / 2;
        if ((sites_L + q) % 2 != 0) continue;
        double w = binom(sites_L, ups_L) * binom(sites_R, ups_R);
        if (w > 0) sectors.emplace_back(q, w);
    }
    if (sectors.empty()) {
        BondSpace bs;
        bs.qns = {0}; bs.dims = {max_bond}; bs.offsets = {0};
        bs.total_dim = max_bond;
        return bs;
    }

    double total_w = 0;
    for (auto& [q,w] : sectors) total_w += w;

    // Proportional allocation, at least 1 per sector
    std::vector<std::pair<int,int>> raw;
    int allocated = 0;
    for (auto& [q,w] : sectors) {
        int d = std::max(1, (int)std::round((double)w / total_w * max_bond));
        raw.emplace_back(q, d);
        allocated += d;
    }
    // Trim or pad to exactly max_bond
    while (allocated > max_bond) {
        auto it = std::max_element(raw.begin(), raw.end(),
            [](auto& a, auto& b){ return a.second < b.second; });
        if (it->second <= 1) break;
        it->second--; allocated--;
    }
    while (allocated < max_bond) {
        auto it = std::max_element(raw.begin(), raw.end(),
            [](auto& a, auto& b){ return a.second < b.second; });
        it->second++; allocated++;
    }
    std::sort(raw.begin(), raw.end());
    return BondSpace::from_pairs(raw);
}

// -- Block-sparse environment -------------------------------------------------
//
// Heisenberg MPO bond -> QN shift of bra vs ket:
//   b=0 (I)    DeltaQN=0    b=1 (S+/X) DeltaQN=+2   b=2 (S-/X) DeltaQN=-2
//   b=3 (Sz/Z) DeltaQN=0    b=4 (full) DeltaQN=0
static constexpr int HB_CHI_W = 5;
static constexpr int HB_DELTA[5] = {0, +2, -2, 0, 0};

// Map DELTA value to the set of MPO bond indices with that delta
static std::vector<int> bonds_with_delta(int delta) {
    std::vector<int> res;
    for (int b = 0; b < HB_CHI_W; ++b)
        if (HB_DELTA[b] == delta) res.push_back(b);
    return res;
}

// =============================================================================
// Scalar-templated core. Everything below uses the member aliases Mat/Vec, so
// the same code runs in real or complex arithmetic; S=double for real
// Hamiltonians (~4x faster), S=complex<double> otherwise.
// =============================================================================
template<typename S>
struct BlockDMRG {
    using Mat = MatX<S>;
    using Vec = VecX<S>;
    // Cast a (possibly complex) MPO scalar to S; for S=double take the real part.
    static S mscal(const Cx& c) {
        if constexpr (std::is_same_v<S, Cx>) return c; else return c.real();
    }
    // Real part of a block scalar (identity for S=double).
    static double rp(const S& x) {
        if constexpr (std::is_same_v<S, Cx>) return x.real(); else return x;
    }

struct BlockEnv {
    // blocks[b][q_ket] = Mat(dim_bra, dim_ket),  bra_qn = q_ket + HB_DELTA[b]
    std::array<std::map<int,Mat>, HB_CHI_W> blocks;
    BondSpace bs;  // ket bond space
};

// Pre-allocate zero blocks for a BlockEnv with a given bond space.
static void init_block_env(BlockEnv& env, const BondSpace& bs) {
    env.bs = bs;
    for (int b = 0; b < HB_CHI_W; ++b) {
        env.blocks[b].clear();
        int delta = HB_DELTA[b];
        for (int qi = 0; qi < bs.n_sectors(); ++qi) {
            int q_ket = bs.qns[qi], d_ket = bs.dims[qi];
            int d_bra = bs.dim(q_ket + delta);
            if (d_bra > 0)
                env.blocks[b][q_ket] = Mat::Zero(d_bra, d_ket);
        }
    }
}

// -- Block-sparse left environment update -------------------------------------
//
// L_new at bond (i+1) from L at bond i, site tensor block_A[i], and MPO W[i].
//
// Physics: L_new[b_out][q_ket_R](m_bra_R, m_ket_R)
//   = sum_{b_in, s, sp} W[b_in][b_out](s, sp)
//     * A_bra[q_bra_L]^dag_{s} * L[b_in][q_ket_L] * A_ket[q_ket_L]_{sp}
//
// where q_bra_L = q_ket_L + HB_DELTA[b_in],
//       q_ket_R = q_ket_L + spz,  q_bra_R = q_bra_L + sz.
//
// Dimension chain: A_bra_{s}^H: (dR_s_bra, dL_bra)
//                 L[b_in]: (dL_bra, dL_ket)
//                 A_ket_{sp}: (dL_ket, dR_sp_ket)
//                 result: (dR_s_bra, dR_sp_ket) = shape of L_new[b_out][q_ket_R]
static BlockEnv block_update_left(const BlockEnv& L,
                                  const std::map<int,Mat>& block_A,
                                  const BondSpace& bs_L,
                                  const BondSpace& bs_R,
                                  const MPOSite& W) {
    BlockEnv L_new;
    init_block_env(L_new, bs_R);

    for (int qi = 0; qi < bs_L.n_sectors(); ++qi) {
        int q_ket_L = bs_L.qns[qi];

        // Ket A tensor (row sector = q_ket_L)
        auto it_A_ket = block_A.find(q_ket_L);
        if (it_A_ket == block_A.end()) continue;
        const Mat& Ablk_ket = it_A_ket->second;  // (dL_ket, dR_up_ket + dR_dn_ket)
        int dR_up_ket = bs_R.dim(q_ket_L + 1);
        int dR_dn_ket = bs_R.dim(q_ket_L - 1);

        for (int b_in = 0; b_in < W.chi_l; ++b_in) {
            auto it_L = L.blocks[b_in].find(q_ket_L);
            if (it_L == L.blocks[b_in].end()) continue;
            const Mat& Lblk = it_L->second;  // (dL_bra, dL_ket), bra_qn = q_ket_L + HB_DELTA[b_in]
            if (Lblk.size() == 0) continue;

            // Bra A tensor: bra left QN = q_ket_L + HB_DELTA[b_in]
            int q_bra_L = q_ket_L + HB_DELTA[b_in];
            auto it_A_bra = block_A.find(q_bra_L);
            if (it_A_bra == block_A.end()) continue;
            const Mat& Ablk_bra = it_A_bra->second;  // (dL_bra, dR_up_bra + dR_dn_bra)
            int dR_up_bra = bs_R.dim(q_bra_L + 1);
            int dR_dn_bra = bs_R.dim(q_bra_L - 1);

            for (int b_out = 0; b_out < W.chi_r; ++b_out) {
                const Mat2& Wab = W.W(b_in, b_out);

                for (int s = 0; s < 2; ++s) {   // bra spin (conjugate A)
                    int sz   = (s == 0) ? +1 : -1;
                    int dR_s = (sz == +1) ? dR_up_bra : dR_dn_bra;
                    if (dR_s == 0) continue;
                    Eigen::Ref<const Mat> A_s_bra = (sz == +1)
                        ? Ablk_bra.leftCols(dR_up_bra) : Ablk_bra.rightCols(dR_dn_bra);

                    for (int sp = 0; sp < 2; ++sp) {  // ket spin
                        S w = mscal(Wab(s, sp));
                        if (std::abs(w) < 1e-15) continue;
                        int spz   = (sp == 0) ? +1 : -1;
                        int dR_sp = (spz == +1) ? dR_up_ket : dR_dn_ket;
                        if (dR_sp == 0) continue;

                        int q_ket_R = q_ket_L + spz;  // new ket right bond QN
                        // QN consistency: HB_DELTA[b_out] = HB_DELTA[b_in] + sz - spz
                        int req_delta = HB_DELTA[b_in] + sz - spz;
                        if (HB_DELTA[b_out] != req_delta) continue;

                        auto jt = L_new.blocks[b_out].find(q_ket_R);
                        if (jt == L_new.blocks[b_out].end()) continue;

                        Eigen::Ref<const Mat> A_sp_ket = (spz == +1)
                            ? Ablk_ket.leftCols(dR_up_ket) : Ablk_ket.rightCols(dR_dn_ket);

                        // L_new(dR_s_bra, dR_sp_ket) += w * A_s_bra^H * Lblk * A_sp_ket
                        jt->second.noalias() += w * (A_s_bra.adjoint() * (Lblk * A_sp_ket));
                    }
                }
            }
        }
    }
    return L_new;
}

// -- Block-sparse right environment update ------------------------------------
//
// R_new at bond i from R at bond (i+1), site tensor block_A[i], MPO W[i].
//
// Physics: R_new[b_out][q_ket_L](m_bra_L, m_ket_L)
//   = sum_{b_in, s, sp} W[b_out][b_in](sp, s)
//     * conj(A_bra[q_bra_L]_{sp}) * R[b_in][q_ket_R] * A_ket[q_ket_L]_{s}^T
//
// where q_bra_L = q_ket_L + HB_DELTA[b_out],
//       q_ket_R = q_ket_L + sz,  q_bra_R = q_bra_L + spz,
//       R[b_in][q_ket_R] has shape (d_bra_R, d_ket_R) with bra_qn = q_ket_R + HB_DELTA[b_in].
//
// Dimension chain: conj(A_bra_{sp}): (dL_bra, dR_sp_bra)
//                 R[b_in]: (dR_sp_bra, dR_s_ket)
//                 A_ket_{s}^T: (dR_s_ket, dL_ket)
//                 result: (dL_bra, dL_ket) = shape of R_new[b_out][q_ket_L]
static BlockEnv block_update_right(const BlockEnv& R,
                                   const std::map<int,Mat>& block_A,
                                   const BondSpace& bs_L,
                                   const BondSpace& bs_R,
                                   const MPOSite& W) {
    BlockEnv R_new;
    init_block_env(R_new, bs_L);

    for (int qi = 0; qi < bs_L.n_sectors(); ++qi) {
        int q_ket_L = bs_L.qns[qi];

        // Ket A tensor (row sector = q_ket_L)
        auto it_A_ket = block_A.find(q_ket_L);
        if (it_A_ket == block_A.end()) continue;
        const Mat& Ablk_ket = it_A_ket->second;  // (dL_ket, dR_up_ket + dR_dn_ket)
        int dR_up_ket = bs_R.dim(q_ket_L + 1);
        int dR_dn_ket = bs_R.dim(q_ket_L - 1);

        for (int b_out = 0; b_out < W.chi_l; ++b_out) {
            auto jt = R_new.blocks[b_out].find(q_ket_L);
            if (jt == R_new.blocks[b_out].end()) continue;

            // Bra A tensor: bra left QN = q_ket_L + HB_DELTA[b_out]
            int q_bra_L = q_ket_L + HB_DELTA[b_out];
            auto it_A_bra = block_A.find(q_bra_L);
            if (it_A_bra == block_A.end()) continue;
            const Mat& Ablk_bra = it_A_bra->second;  // (dL_bra, dR_up_bra + dR_dn_bra)
            int dR_up_bra = bs_R.dim(q_bra_L + 1);
            int dR_dn_bra = bs_R.dim(q_bra_L - 1);

            for (int b_in = 0; b_in < W.chi_r; ++b_in) {
                const Mat2& Wab = W.W(b_out, b_in);

                for (int s = 0; s < 2; ++s) {   // ket spin
                    int sz   = (s == 0) ? +1 : -1;
                    int dR_s = (sz == +1) ? dR_up_ket : dR_dn_ket;
                    if (dR_s == 0) continue;
                    int q_ket_R = q_ket_L + sz;
                    Eigen::Ref<const Mat> A_s_ket = (sz == +1)
                        ? Ablk_ket.leftCols(dR_up_ket) : Ablk_ket.rightCols(dR_dn_ket);

                    for (int sp = 0; sp < 2; ++sp) {  // bra spin
                        S w = mscal(Wab(sp, s));
                        if (std::abs(w) < 1e-15) continue;
                        int spz   = (sp == 0) ? +1 : -1;
                        int dR_sp = (spz == +1) ? dR_up_bra : dR_dn_bra;
                        if (dR_sp == 0) continue;

                        // QN consistency: HB_DELTA[b_out] = HB_DELTA[b_in] + sz - spz
                        // (here s is the KET spin, sp is the BRA spin).
                        int req_delta = HB_DELTA[b_in] + sz - spz;
                        if (HB_DELTA[b_out] != req_delta) continue;

                        auto it_R = R.blocks[b_in].find(q_ket_R);
                        if (it_R == R.blocks[b_in].end()) continue;
                        const Mat& Rblk = it_R->second;  // (dR_sp_bra, dR_s_ket)
                        if (Rblk.rows() != dR_sp || Rblk.cols() != dR_s) continue;

                        Eigen::Ref<const Mat> A_sp_bra = (spz == +1)
                            ? Ablk_bra.leftCols(dR_up_bra) : Ablk_bra.rightCols(dR_dn_bra);

                        // R_new(dL_bra, dL_ket) += w * conj(A_sp_bra) * Rblk * A_s_ket^T
                        jt->second.noalias() += w * (A_sp_bra.conjugate() * (Rblk * A_s_ket.transpose()));
                    }
                }
            }
        }
    }
    return R_new;
}

// -- Block-sparse effective Hamiltonian application ---------------------------
//
// Applies H_eff to the two-site block tensor theta[q_mid].
//
// The effective Hamiltonian is:
//   out[q_mid_bra] += sum_{bL,bR,s1k,s2k,s1b,s2b,q_mid_ket}
//       L[bL][q_L_ket] * Wc[bL][bR][s1b,s2b,s1k,s2k] * R[bR][q_R_ket]^T
//       * theta[q_mid_ket][s1k_rows, s2k_cols]
//
// The QN relations:
//   q_L_ket  = q_mid_ket - s1k_z
//   q_L_bra  = q_L_ket  + HB_DELTA[bL]
//   q_mid_bra = q_L_bra + s1b_z  (may differ from q_mid_ket when XY coupling flips site-i spin)
//   q_R_ket  = q_mid_ket + s2k_z
//   q_R_bra  = q_mid_bra + s2b_z = q_R_ket + HB_DELTA[bR]  (QN conservation at right end)
//
// Crucially, q_mid_bra can differ from q_mid_ket (by +-2) when s1b != s1k.
// This cross-sector coupling is what mediates the XY term in the Heisenberg Hamiltonian.

// Combined two-site MPO weights, fixed per bond:
//   Wc[bL][bR][s1b*2+s2b][s1k*2+s2k] = sum_bm Wi(bL,bm)(s1b,s1k) * Wj(bm,bR)(s2b,s2k)
struct WComb {
    std::array<std::array<std::array<std::array<S,4>,4>,HB_CHI_W>,HB_CHI_W> w{};
};

static WComb build_wcomb(const MPOSite& Wi, const MPOSite& Wj) {
    WComb wc;
    for (int bL = 0; bL < Wi.chi_l; ++bL)
        for (int bm = 0; bm < Wi.chi_r; ++bm)
            for (int bR = 0; bR < Wj.chi_r; ++bR) {
                const Mat2& wi = Wi.W(bL, bm);
                const Mat2& wj = Wj.W(bm, bR);
                for (int s1b = 0; s1b < 2; ++s1b)
                    for (int s2b = 0; s2b < 2; ++s2b)
                        for (int s1k = 0; s1k < 2; ++s1k)
                            for (int s2k = 0; s2k < 2; ++s2k)
                                wc.w[bL][bR][s1b*2+s2b][s1k*2+s2k] += mscal(wi(s1b,s1k) * wj(s2b,s2k));
            }
    return wc;
}

// -- Precomputed H_eff contraction plan ---------------------------------------
//
// The environments L, R, the MPO and the bond spaces are fixed for the whole
// local eigensolve, so all the QN bookkeeping (sector matching, std::map block
// lookups, dimension/offset arithmetic) can be done ONCE and replayed for every
// matvec.  This removes the per-matvec map lookups that dominate when there are
// many small sectors (long chains).  The plan is grouped to keep the two-GEMM
// reuse of block_apply_heff: T1 = theta_sub * R^T once per (ket-block, R-block),
// and LT1 = L * T1 once per L-block.
struct HeffPlan {
    std::vector<int>                 out_q;       // out index -> q_mid_bra
    std::vector<std::pair<int,int>>  out_shape;   // out index -> (rows, cols)
    std::vector<int>                 ket_sectors; // ket index -> theta sector key
    double flops = 0;                             // ~GEMM work per matvec (par gate)
    int    max_t1 = 1;                            // max elements of any T1 = theta*R^T
    int    max_lt1 = 1;                           // max elements of any LT1 = L*T1

    struct Write  { S w; int out_idx, row_b, col_b, dL_b, dR_b; };
    struct LGroup { const Mat* Lblk; std::vector<Write> writes; };
    struct T1Group {
        int ket_idx;                 // index into ket_sectors (theta block resolved once)
        int row_k, col_k, dL_k, dR_k;
        const Mat* Rblk;
        std::vector<LGroup> lgroups;
    };
    std::vector<T1Group> groups;

    // Row-chunked reduction tasks: summing the per-thread partial outputs is
    // parallelised over (out sector, row range) so the dominant sector does not
    // serialise the merge. Built once per bond from out_shape.
    struct Chunk { int out_idx, r0, rc; };
    std::vector<Chunk> reduce_chunks;
};

// Build the plan. `ket_keys` are the sectors the input theta will carry (the
// expanded two-site tensor sectors). L and R must outlive every use of the plan.
static HeffPlan build_heff_plan(
    const BlockEnv& L, const MPOSite& Wi, const MPOSite& Wj, const BlockEnv& R,
    const WComb& Wc, const std::vector<int>& ket_keys,
    const BondSpace& bs_L, const BondSpace& bs_R2)
{
    HeffPlan plan;
    std::map<int,int> out_index;  // q_mid_bra -> out idx
    auto get_out_idx = [&](int q, int nr, int nc) -> int {
        auto it = out_index.find(q);
        if (it != out_index.end()) return it->second;
        int idx = (int)plan.out_q.size();
        out_index[q] = idx;
        plan.out_q.push_back(q);
        plan.out_shape.push_back({nr, nc});
        return idx;
    };

    for (int q_mid_ket : ket_keys) {
        int ket_idx = (int)plan.ket_sectors.size();
        plan.ket_sectors.push_back(q_mid_ket);
        int dL_up_ket = bs_L.dim(q_mid_ket - 1);
        int dL_dn_ket = bs_L.dim(q_mid_ket + 1);
        int dR_up_ket = bs_R2.dim(q_mid_ket + 1);
        int dR_dn_ket = bs_R2.dim(q_mid_ket - 1);

        for (int s1k = 0; s1k < 2; ++s1k) {
            int s1kz = (s1k == 0) ? +1 : -1;
            int dL_k = (s1k == 0) ? dL_up_ket : dL_dn_ket;
            if (dL_k == 0) continue;
            int row_k = (s1k == 0) ? 0 : dL_up_ket;
            int q_L_ket = q_mid_ket - s1kz;

            for (int s2k = 0; s2k < 2; ++s2k) {
                int s2kz = (s2k == 0) ? +1 : -1;
                int dR_k = (s2k == 0) ? dR_up_ket : dR_dn_ket;
                if (dR_k == 0) continue;
                int col_k = (s2k == 0) ? 0 : dR_up_ket;
                int q_R_ket = q_mid_ket + s2kz;

                for (int bR = 0; bR < Wj.chi_r; ++bR) {
                    auto it_R = R.blocks[bR].find(q_R_ket);
                    if (it_R == R.blocks[bR].end()) continue;
                    const Mat& Rblk = it_R->second;
                    if (Rblk.cols() != dR_k) continue;
                    int dR_bra = (int)Rblk.rows();
                    int q_R_bra = q_R_ket + HB_DELTA[bR];

                    typename HeffPlan::T1Group g;
                    g.ket_idx = ket_idx;
                    g.row_k = row_k; g.col_k = col_k; g.dL_k = dL_k; g.dR_k = dR_k;
                    g.Rblk = &Rblk;
                    plan.flops += 2.0 * dL_k * dR_k * dR_bra;  // T1 = theta_sub * R^T
                    plan.max_t1 = std::max(plan.max_t1, dL_k * dR_bra);

                    for (int bL = 0; bL < Wi.chi_l; ++bL) {
                        int q_L_bra = q_L_ket + HB_DELTA[bL];
                        int dL_bra  = bs_L.dim(q_L_bra);
                        if (dL_bra == 0) continue;
                        auto it_L = L.blocks[bL].find(q_L_ket);
                        if (it_L == L.blocks[bL].end()) continue;
                        const Mat& Lblk = it_L->second;
                        if (Lblk.rows() != dL_bra || Lblk.cols() != dL_k) continue;

                        typename HeffPlan::LGroup lg;
                        lg.Lblk = &Lblk;
                        plan.flops += 2.0 * dL_bra * dL_k * dR_bra;  // LT1 = L * T1
                        plan.max_lt1 = std::max(plan.max_lt1, dL_bra * dR_bra);

                        for (int s1b = 0; s1b < 2; ++s1b) {
                            int s1bz = (s1b == 0) ? +1 : -1;
                            int q_mid_bra = q_L_bra + s1bz;
                            int dL_up_bra = bs_L.dim(q_mid_bra - 1);
                            int dL_dn_bra = bs_L.dim(q_mid_bra + 1);
                            int dR_up_bra = bs_R2.dim(q_mid_bra + 1);
                            int dR_dn_bra = bs_R2.dim(q_mid_bra - 1);
                            int n_row = dL_up_bra + dL_dn_bra;
                            int n_col = dR_up_bra + dR_dn_bra;
                            if (n_row == 0 || n_col == 0) continue;
                            int dL_b = (s1b == 0) ? dL_up_bra : dL_dn_bra;
                            if (dL_b != dL_bra) continue;
                            int row_b = (s1b == 0) ? 0 : dL_up_bra;

                            for (int s2b = 0; s2b < 2; ++s2b) {
                                S w = Wc.w[bL][bR][s1b*2+s2b][s1k*2+s2k];
                                if (std::abs(w) < 1e-15) continue;
                                int s2bz = (s2b == 0) ? +1 : -1;
                                int dR_b = (s2b == 0) ? dR_up_bra : dR_dn_bra;
                                if (dR_b == 0) continue;
                                if (q_mid_bra + s2bz != q_R_bra) continue;
                                if (dR_bra != dR_b) continue;
                                int col_b = (s2b == 0) ? 0 : dR_up_bra;

                                int idx = get_out_idx(q_mid_bra, n_row, n_col);
                                lg.writes.push_back({w, idx, row_b, col_b, dL_b, dR_b});
                            }
                        }
                        if (!lg.writes.empty()) g.lgroups.push_back(std::move(lg));
                    }
                    if (!g.lgroups.empty()) plan.groups.push_back(std::move(g));
                }
            }
        }
    }
    // Row-chunked reduction tasks (split big sectors so the merge parallelises).
    constexpr int CH = 48;
    for (int s = 0; s < (int)plan.out_q.size(); ++s) {
        int rows = plan.out_shape[s].first;
        for (int r0 = 0; r0 < rows; r0 += CH)
            plan.reduce_chunks.push_back({s, r0, std::min(CH, rows - r0)});
    }
    return plan;
}

// Execute the plan on the current theta, returning out[q_mid_bra].
static std::map<int,Mat> apply_heff_plan(const HeffPlan& plan,
                                         const std::map<int,Mat>& theta) {
    ++g_block_matvecs;
    const int n_out = (int)plan.out_q.size();
    const int n_grp = (int)plan.groups.size();
    // Thread only when the GEMM work outweighs the fork-join + merge overhead.
    const bool par = (plan.flops >= 4.0e6) && (n_grp > 1);

    // Resolve each distinct ket sector's theta block once (not per group).
    std::vector<const Mat*> theta_blocks(plan.ket_sectors.size(), nullptr);
    for (size_t k = 0; k < plan.ket_sectors.size(); ++k) {
        auto it = theta.find(plan.ket_sectors[k]);
        if (it != theta.end()) theta_blocks[k] = &it->second;
    }

    std::vector<Mat> out(n_out);

    if (!par) {
        // Serial: accumulate straight into `out`, using reusable scratch for the two
        // GEMM temporaries (no per-group heap allocation).
        std::vector<S> t1buf(plan.max_t1), lt1buf(plan.max_lt1);
        for (int gi = 0; gi < n_grp; ++gi) {
            const typename HeffPlan::T1Group& g = plan.groups[gi];
            const Mat* theta_q = theta_blocks[g.ket_idx];
            if (!theta_q) continue;
            int dRb = (int)g.Rblk->rows();
            Eigen::Map<Mat> T1(t1buf.data(), g.dL_k, dRb);
            if (g.dL_k <= 24 && dRb <= 24)
                T1.noalias() = theta_q->block(g.row_k, g.col_k, g.dL_k, g.dR_k).lazyProduct(g.Rblk->transpose());
            else
                T1.noalias() = theta_q->block(g.row_k, g.col_k, g.dL_k, g.dR_k) * g.Rblk->transpose();
            for (const auto& lg : g.lgroups) {
                int dLb = (int)lg.Lblk->rows();
                Eigen::Map<Mat> LT1(lt1buf.data(), dLb, dRb);
                if (dLb <= 24 && dRb <= 24) LT1.noalias() = lg.Lblk->lazyProduct(T1);
                else                        LT1.noalias() = (*lg.Lblk) * T1;
                for (const auto& wr : lg.writes) {
                    Mat& o = out[wr.out_idx];
                    if (o.size() == 0)
                        o = Mat::Zero(plan.out_shape[wr.out_idx].first,
                                      plan.out_shape[wr.out_idx].second);
                    o.block(wr.row_b, wr.col_b, wr.dL_b, wr.dR_b).noalias() += wr.w * LT1;
                }
            }
        }
    } else {
        // Parallel: each thread builds a private partial output; the partials are
        // then summed by a row-chunked parallel reduction (no serial critical merge).
        const int nthreads = omp_get_max_threads();
        std::vector<std::vector<Mat>> tls(nthreads);

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::vector<Mat>& dst = tls[tid];
            dst.assign(n_out, Mat());
            std::vector<S> t1buf(plan.max_t1), lt1buf(plan.max_lt1);  // per-thread scratch
            #pragma omp for schedule(dynamic) nowait
            for (int gi = 0; gi < n_grp; ++gi) {
                const typename HeffPlan::T1Group& g = plan.groups[gi];
                const Mat* theta_q = theta_blocks[g.ket_idx];
                if (!theta_q) continue;
                int dRb = (int)g.Rblk->rows();
                Eigen::Map<Mat> T1(t1buf.data(), g.dL_k, dRb);
                if (g.dL_k <= 24 && dRb <= 24)
                    T1.noalias() = theta_q->block(g.row_k, g.col_k, g.dL_k, g.dR_k).lazyProduct(g.Rblk->transpose());
                else
                    T1.noalias() = theta_q->block(g.row_k, g.col_k, g.dL_k, g.dR_k) * g.Rblk->transpose();
                for (const auto& lg : g.lgroups) {
                    int dLb = (int)lg.Lblk->rows();
                    Eigen::Map<Mat> LT1(lt1buf.data(), dLb, dRb);
                    if (dLb <= 24 && dRb <= 24) LT1.noalias() = lg.Lblk->lazyProduct(T1);
                    else                        LT1.noalias() = (*lg.Lblk) * T1;
                    for (const auto& wr : lg.writes) {
                        Mat& o = dst[wr.out_idx];
                        if (o.size() == 0)
                            o = Mat::Zero(plan.out_shape[wr.out_idx].first,
                                          plan.out_shape[wr.out_idx].second);
                        o.block(wr.row_b, wr.col_b, wr.dL_b, wr.dR_b).noalias() += wr.w * LT1;
                    }
                }
            }
        }  // implicit barrier

        // Allocate output for touched sectors (every row is rewritten below).
        for (int s = 0; s < n_out; ++s)
            for (int t = 0; t < nthreads; ++t)
                if (!tls[t].empty() && tls[t][s].size() != 0) {
                    out[s].resize(plan.out_shape[s].first, plan.out_shape[s].second);
                    break;
                }

        const int n_chunk = (int)plan.reduce_chunks.size();
        #pragma omp parallel for schedule(dynamic)
        for (int ci = 0; ci < n_chunk; ++ci) {
            const auto& ch = plan.reduce_chunks[ci];
            if (out[ch.out_idx].size() == 0) continue;  // untouched sector
            bool first = true;
            for (int t = 0; t < nthreads; ++t) {
                if (tls[t].empty() || tls[t][ch.out_idx].size() == 0) continue;
                auto src = tls[t][ch.out_idx].middleRows(ch.r0, ch.rc);
                if (first) { out[ch.out_idx].middleRows(ch.r0, ch.rc) = src; first = false; }
                else       { out[ch.out_idx].middleRows(ch.r0, ch.rc) += src; }
            }
        }
    }

    std::map<int,Mat> res;
    for (int i = 0; i < n_out; ++i)
        if (out[i].size() != 0) res[plan.out_q[i]] = std::move(out[i]);
    return res;
}

// -- Single-site effective Hamiltonian ----------------------------------------
//
// Applies the one-site H_eff to the site tensor M (same block layout as block_A[i]:
//   M[q_L] = Mat(dL, dR_up + dR_dn),  up cols -> q_R = q_L+1, dn cols -> q_R = q_L-1).
// L is the environment at the LEFT bond of the site, R at the RIGHT bond, W the MPO
// at the site. QN relations (bra vs ket):
//   q_L_bra = q_L_ket + HB_DELTA[b_in]
//   q_R_ket = q_L_ket + s_ket_z,  q_R_bra = q_L_bra + s_bra_z = q_R_ket + HB_DELTA[b_out]
static std::map<int,Mat> apply_heff_1site(
    const BlockEnv& L, const MPOSite& W, const BlockEnv& R,
    const std::map<int,Mat>& M,
    const BondSpace& bs_L, const BondSpace& bs_R)
{
    ++g_block_matvecs;
    std::map<int,Mat> out;

    for (auto& [q_L_ket, Mblk] : M) {
        int dL_ket   = bs_L.dim(q_L_ket);
        int dR_up_ket = bs_R.dim(q_L_ket + 1);
        int dR_dn_ket = bs_R.dim(q_L_ket - 1);

        for (int s_ket = 0; s_ket < 2; ++s_ket) {
            int skz   = (s_ket == 0) ? +1 : -1;
            int dR_s_ket = (s_ket == 0) ? dR_up_ket : dR_dn_ket;
            if (dR_s_ket == 0) continue;
            int col_k = (s_ket == 0) ? 0 : dR_up_ket;
            int q_R_ket = q_L_ket + skz;
            Eigen::Ref<const Mat> M_sub = Mblk.block(0, col_k, dL_ket, dR_s_ket);

            for (int b_in = 0; b_in < W.chi_l; ++b_in) {
                int q_L_bra = q_L_ket + HB_DELTA[b_in];
                int dL_bra  = bs_L.dim(q_L_bra);
                if (dL_bra == 0) continue;
                auto it_L = L.blocks[b_in].find(q_L_ket);
                if (it_L == L.blocks[b_in].end()) continue;
                const Mat& Lblk = it_L->second;   // (dL_bra, dL_ket)
                if (Lblk.rows() != dL_bra || Lblk.cols() != dL_ket) continue;

                Mat LM = Lblk * M_sub;            // (dL_bra, dR_s_ket)

                for (int b_out = 0; b_out < W.chi_r; ++b_out) {
                    auto it_R = R.blocks[b_out].find(q_R_ket);
                    if (it_R == R.blocks[b_out].end()) continue;
                    const Mat& Rblk = it_R->second;   // (dR_bra, dR_s_ket)
                    if (Rblk.cols() != dR_s_ket) continue;
                    int dR_bra = (int)Rblk.rows();
                    int q_R_bra = q_R_ket + HB_DELTA[b_out];

                    // s_bra fixed by q_R_bra = q_L_bra + s_bra_z
                    int s_bra = (q_R_bra == q_L_bra + 1) ? 0
                              : (q_R_bra == q_L_bra - 1) ? 1 : -1;
                    if (s_bra < 0) continue;
                    S w = mscal(W.W(b_in, b_out)(s_bra, s_ket));
                    if (std::abs(w) < 1e-15) continue;
                    int dR_up_bra = bs_R.dim(q_L_bra + 1);
                    int dR_dn_bra = bs_R.dim(q_L_bra - 1);
                    int dR_s_bra  = (s_bra == 0) ? dR_up_bra : dR_dn_bra;
                    if (dR_s_bra != dR_bra) continue;
                    int col_b = (s_bra == 0) ? 0 : dR_up_bra;

                    Mat T = LM * Rblk.transpose();   // (dL_bra, dR_bra)

                    auto it_out = out.find(q_L_bra);
                    if (it_out == out.end()) {
                        out[q_L_bra] = Mat::Zero(dL_bra, dR_up_bra + dR_dn_bra);
                        it_out = out.find(q_L_bra);
                    }
                    it_out->second.block(0, col_b, dL_bra, dR_bra).noalias() += w * T;
                }
            }
        }
    }
    // Ensure all input sectors are present (Galerkin: H_eff maps the space to itself).
    for (auto& [q, blk] : M)
        if (!out.count(q)) out[q] = Mat::Zero(blk.rows(), blk.cols());
    return out;
}

// -- Single-site canonicalisation (center move) -------------------------------
//
// L->R: split the one-site centre tensor M (block_A[i] layout over bs_L x bs_mid)
// into a left-canonical A[i] and centre matrices C[q_R] (to be absorbed into the
// next site).  Grouping by the right-bond QN q_R, the rows that map to q_R are the
// up-block of M[q_R-1] and the dn-block of M[q_R+1]; an SVD of that stacked matrix
// gives U (-> A) and S V^dag (-> C).  No truncation (keep = min(rows, dim)).
struct SiteMove {
    std::map<int,Mat> A;       // q_L -> left-canonical site tensor
    std::map<int,Mat> C;       // q_R -> centre matrix (keep x old dim)
    BondSpace          new_mid;
};

static SiteMove singlesite_split_LR(const std::map<int,Mat>& M,
                                    const BondSpace& bs_L, const BondSpace& bs_mid) {
    SiteMove mv;
    std::map<int,Mat> Umap;     // q_R -> U (rows x keep)
    std::map<int,int> keepmap;  // q_R -> keep
    std::vector<std::pair<int,int>> new_sectors;

    std::set<int> qRs;
    for (auto& [q_L, _] : M) { qRs.insert(q_L + 1); qRs.insert(q_L - 1); }

    for (int q_R : qRs) {
        int dRmid = bs_mid.dim(q_R);
        if (dRmid == 0) continue;
        int dL_up = bs_L.dim(q_R - 1);   // s=up rows come from M[q_R-1]
        int dL_dn = bs_L.dim(q_R + 1);   // s=dn rows come from M[q_R+1]
        int rows = dL_up + dL_dn;
        if (rows == 0) continue;
        Mat X = Mat::Zero(rows, dRmid);
        if (dL_up > 0 && M.count(q_R - 1))
            X.topRows(dL_up) = M.at(q_R - 1).leftCols(dRmid);            // up-block
        if (dL_dn > 0 && M.count(q_R + 1)) {
            int dRup_p = bs_mid.dim(q_R + 2);                            // up-width of M[q_R+1]
            X.bottomRows(dL_dn) = M.at(q_R + 1).block(0, dRup_p, dL_dn, dRmid);  // dn-block
        }
        Eigen::BDCSVD<Mat> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
        int keep = (int)svd.singularValues().size();
        Umap[q_R] = svd.matrixU().leftCols(keep);
        Mat Vt = svd.matrixV().adjoint().topRows(keep);                 // keep x dRmid
        for (int k = 0; k < keep; ++k) Vt.row(k) *= svd.singularValues()[k];
        mv.C[q_R] = Vt;                                                 // S V^dag
        keepmap[q_R] = keep;
        new_sectors.emplace_back(q_R, keep);
    }
    std::sort(new_sectors.begin(), new_sectors.end());
    mv.new_mid = BondSpace::from_pairs(new_sectors);

    // Reassemble A[q_L]: up-cols from split(q_L+1) top rows, dn-cols from split(q_L-1) bottom rows.
    for (int qi = 0; qi < bs_L.n_sectors(); ++qi) {
        int q_L = bs_L.qns[qi], dL = bs_L.dims[qi];
        int keep_up = keepmap.count(q_L + 1) ? keepmap[q_L + 1] : 0;
        int keep_dn = keepmap.count(q_L - 1) ? keepmap[q_L - 1] : 0;
        Mat A = Mat::Zero(dL, keep_up + keep_dn);
        if (keep_up > 0) A.leftCols(keep_up)  = Umap[q_L + 1].topRows(dL);
        if (keep_dn > 0) A.rightCols(keep_dn) = Umap[q_L - 1].bottomRows(dL);
        mv.A[q_L] = A;
    }
    return mv;
}

// R->L: split the one-site centre tensor M into a RIGHT-canonical B[i] (stored in
// mv.A) and centre matrices C[q_L] = U*S (stored in mv.C, to be absorbed into the
// previous site). Per left-bond sector -- the physical+right index stays together.
// mv.new_mid is the new LEFT bond space (bond[i]).
static SiteMove singlesite_split_RL(const std::map<int,Mat>& M) {
    SiteMove mv;
    std::vector<std::pair<int,int>> new_sectors;
    for (auto& [q_L, Mblk] : M) {
        if (Mblk.size() == 0) continue;
        Eigen::BDCSVD<Mat> svd(Mblk, Eigen::ComputeThinU | Eigen::ComputeThinV);
        int keep = (int)svd.singularValues().size();
        mv.A[q_L] = svd.matrixV().adjoint().topRows(keep);   // keep x cols (right-canonical)
        Mat Cp = svd.matrixU().leftCols(keep);               // dL x keep
        for (int k = 0; k < keep; ++k) Cp.col(k) *= svd.singularValues()[k];
        mv.C[q_L] = Cp;                                      // U S
        new_sectors.emplace_back(q_L, keep);
    }
    std::sort(new_sectors.begin(), new_sectors.end());
    mv.new_mid = BondSpace::from_pairs(new_sectors);
    return mv;
}

// -- Block Lanczos (flat packing) ---------------------------------------------

static int theta_total_dim(const std::map<int,Mat>& theta) {
    int d = 0;
    for (auto& [q,blk] : theta) d += (int)(blk.rows() * blk.cols());
    return d;
}

static Vec theta_to_vec(const std::map<int,Mat>& theta) {
    Vec v(theta_total_dim(theta));
    int off = 0;
    for (auto& [q, blk] : theta) {
        int sz = (int)(blk.rows() * blk.cols());
        std::copy(blk.data(), blk.data() + sz, v.data() + off);
        off += sz;
    }
    return v;
}

static void vec_to_theta(const Vec& v, const std::map<int,Mat>& ref,
                         std::map<int,Mat>& out) {
    int off = 0;
    for (auto& [q, blk_ref] : ref) {
        int sz = (int)(blk_ref.rows() * blk_ref.cols());
        out[q] = Eigen::Map<const Mat>(v.data() + off, blk_ref.rows(), blk_ref.cols());
        off += sz;
    }
}

static std::pair<double, std::map<int,Mat>> block_lanczos(
    const std::function<std::map<int,Mat>(const std::map<int,Mat>&)>& matvec,
    const std::map<int,Mat>& init,
    int max_iter = 100, double tol = 1e-10)
{
    int dim = theta_total_dim(init);
    if (dim == 0) return {0.0, init};

    // Normalised matvec: output has EXACTLY the same sector structure as init.
    // block_apply_heff may return a strict subset of init sectors (zero coupling), or
    // extra sectors outside init (Galerkin projection drops those). We project the
    // result onto init: missing sectors -> zero, extra sectors -> dropped. This keeps
    // theta_to_vec / the dense H_eff matrix dimensionally consistent with init.
    auto matvec_normed = [&](const std::map<int,Mat>& v) -> std::map<int,Mat> {
        auto raw = matvec(v);
        std::map<int,Mat> out;
        for (auto& [q, ref] : init) {
            auto it = raw.find(q);
            if (it != raw.end() && it->second.rows() == ref.rows()
                                && it->second.cols() == ref.cols())
                out[q] = it->second;
            else
                out[q] = Mat::Zero(ref.rows(), ref.cols());
        }
        return out;
    };

    // Small dim: full diagonalisation. Building H costs `dim` matvecs, so this only
    // pays off when dim is below a typical Lanczos iteration count; keep the threshold
    // small (larger problems use the iterative path, which exploits the good initial guess).
    if (dim <= 24) {
        Mat H(dim, dim);
        std::map<int,Mat> e_blk;
        for (auto& [q, ref] : init) e_blk[q] = Mat::Zero(ref.rows(), ref.cols());
        int col = 0;
        // Iterate basis elements in COLUMN-MAJOR order (col outer, row inner) to match
        // theta_to_vec / vec_to_theta, which pack each block column-major (Eigen default).
        // Iterating row-major here would scramble H's column ordering relative to its rows.
        for (auto& [q, blk_ref] : init) {
            for (int c = 0; c < blk_ref.cols(); ++c)
                for (int r = 0; r < blk_ref.rows(); ++r) {
                    // Zero ALL sectors before setting the single basis element
                    for (auto& [qq, blk] : e_blk) blk.setZero();
                    e_blk[q](r, c) = 1.0;
                    H.col(col++) = theta_to_vec(matvec_normed(e_blk));
                }
        }
        H = (H + H.adjoint()) * 0.5;
        Eigen::SelfAdjointEigenSolver<Mat> eig(H);
        std::map<int,Mat> gs;
        vec_to_theta(eig.eigenvectors().col(0), init, gs);
        return {eig.eigenvalues()[0], gs};
    }

    // Lanczos with Ritz-residual stopping. The initial vector is the current two-site
    // state, which in converged sweeps is already (close to) the eigenvector, so check
    // the residual every iteration and allow a very early exit.
    constexpr int MIN_STEPS  = 2;
    constexpr int CHECK_FREQ = 1;

    auto matvec_flat = [&](const Vec& vf) -> Vec {
        std::map<int,Mat> tb; vec_to_theta(vf, init, tb);
        return theta_to_vec(matvec_normed(tb));
    };

    Vec v0 = theta_to_vec(init);
    double nrm = v0.norm();
    if (nrm < 1e-14) { v0.setRandom(); nrm = v0.norm(); }
    v0 /= nrm;

    int m = std::min(max_iter, dim);
    std::vector<Vec> q_vecs; q_vecs.reserve(m + 1);
    q_vecs.push_back(v0);
    Eigen::VectorXd alpha(m), beta(m);
    int actual_m = m;

    for (int j = 0; j < m; ++j) {
        Vec w = matvec_flat(q_vecs[j]);
        alpha[j] = rp(w.dot(q_vecs[j]));
        w -= alpha[j] * q_vecs[j];
        if (j > 0) w -= beta[j-1] * q_vecs[j-1];
        double b = w.norm(); beta[j] = b;
        if (b < 1e-14) { actual_m = j+1; break; }
        if (j < m-1) q_vecs.push_back(w / b);
        int cur = j+1;
        if (cur >= MIN_STEPS && (cur % CHECK_FREQ == 0 || j == m-1)) {
            Eigen::MatrixXd T = Eigen::MatrixXd::Zero(cur, cur);
            for (int k = 0; k < cur;   ++k) T(k,k)   = alpha[k];
            for (int k = 0; k < cur-1; ++k) { T(k,k+1) = beta[k]; T(k+1,k) = beta[k]; }
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(T);
            double ritz = b * std::abs(eig.eigenvectors().col(0)[cur-1]);
            // Stop on an absolute OR a relative residual: the local eigensolve only
            // needs to be accurate to well below the SVD truncation, so converging it
            // to ~1e-12 relative (as a fixed 1e-10 absolute does for large |E|) just
            // wastes matvecs. The overall accuracy is recovered over the sweeps.
            double e0 = std::abs(eig.eigenvalues()[0]);
            if (ritz < tol || ritz < 1e-7 * e0) { actual_m = cur; break; }
        }
    }
    m = actual_m;
    Eigen::MatrixXd T(m,m); T.setZero();
    for (int j = 0; j < m;   ++j) T(j,j)   = alpha[j];
    for (int j = 0; j < m-1; ++j) { T(j,j+1) = beta[j]; T(j+1,j) = beta[j]; }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(T);
    Vec psi = Vec::Zero(dim);
    for (int j = 0; j < std::min(m,(int)q_vecs.size()); ++j)
        psi += eig.eigenvectors()(j,0) * q_vecs[j];
    psi /= psi.norm();
    std::map<int,Mat> gs; vec_to_theta(psi, init, gs);
    return {eig.eigenvalues()[0], gs};
}

// -- Block-sparse SVD and MPS update ------------------------------------------

struct BlockSVDResult {
    std::map<int,Mat>  U_up;   // q_mid -> (dL_up(q_mid), n_keep) -- up rows of U
    std::map<int,Mat>  U_dn;   // q_mid -> (dL_dn(q_mid), n_keep) -- dn rows of U
    std::map<int,Mat>  Vt;     // q_mid -> (n_keep, n_cols)
    std::map<int,RVec> sv;
    BondSpace          new_mid_bs;
    double             trunc_err;
};

// Thin SVD M = U diag(sv) V^dag via the Gram matrix + Hermitian eigensolver --
// faster than a general SVD (solves the simpler m x m or n x n eigenproblem) and
// the standard route in fast TEBD codes. Squares the condition number, so the
// tiny tail singular values (which TEBD truncates anyway) lose precision; the
// kept Schmidt values are accurate. U:(m x k), V:(n x k), sv:(k) descending.
static void thin_svd_gram(const Mat& M, Mat& U, Mat& V, RVec& sv) {
    const int m = (int)M.rows(), nc = (int)M.cols();
    if (m <= nc) {
        Eigen::SelfAdjointEigenSolver<Mat> es(M * M.adjoint());   // m x m
        const RVec& ev = es.eigenvalues();                        // ascending
        const Mat&  Q  = es.eigenvectors();
        U.resize(m, m); sv.resize(m);
        for (int j = 0; j < m; ++j) {
            double lam = ev(m-1-j); sv(j) = std::sqrt(lam > 0 ? lam : 0.0);
            U.col(j) = Q.col(m-1-j);
        }
        Mat MhU = M.adjoint() * U;                                // n x m
        V.resize(nc, m);
        double smax = (sv.size() && sv(0) > 0) ? sv(0) : 1.0;
        for (int j = 0; j < m; ++j) {
            if (sv(j) > 1e-14 * smax) V.col(j) = MhU.col(j) / sv(j);
            else V.col(j).setZero();
        }
    } else {
        Eigen::SelfAdjointEigenSolver<Mat> es(M.adjoint() * M);   // n x n
        const RVec& ev = es.eigenvalues();
        const Mat&  Q  = es.eigenvectors();
        V.resize(nc, nc); sv.resize(nc);
        for (int j = 0; j < nc; ++j) {
            double lam = ev(nc-1-j); sv(j) = std::sqrt(lam > 0 ? lam : 0.0);
            V.col(j) = Q.col(nc-1-j);
        }
        Mat MV = M * V;                                           // m x n
        U.resize(m, nc);
        double smax = (sv.size() && sv(0) > 0) ? sv(0) : 1.0;
        for (int j = 0; j < nc; ++j) {
            if (sv(j) > 1e-14 * smax) U.col(j) = MV.col(j) / sv(j);
            else U.col(j).setZero();
        }
    }
}

static BlockSVDResult block_svd_truncate(
    const std::map<int,Mat>& theta,
    const BondSpace& bs_L,
    const BondSpace& bs_R2,
    int max_bond, double cutoff)
{
    // SVD per q_mid sector (independent across sectors -> parallelise). Use
    // JacobiSVD for small blocks (much lower fixed overhead than BDCSVD, which
    // dominates the TEBD step when each bond carries many small Sz sectors), and
    // BDCSVD only for the few large blocks. Results stored directly (no map of
    // heavy SVD objects).
    struct SVDBlk { Mat U, V; RVec s; };
    std::vector<int> svq;
    std::vector<const Mat*> svblk;
    for (auto& [q_mid, blk] : theta)
        if (blk.size() != 0) { svq.push_back(q_mid); svblk.push_back(&blk); }
    g_tebd_nsvd += (long long)svq.size();
    for (auto* p : svblk) g_tebd_maxblk = std::max(g_tebd_maxblk, (long long)std::min(p->rows(), p->cols()));
    std::vector<SVDBlk> sb(svq.size());
    #pragma omp parallel for schedule(dynamic) if(svq.size() > 1)
    for (int i = 0; i < (int)svq.size(); ++i)
        thin_svd_gram(*svblk[i], sb[i].U, sb[i].V, sb[i].s);
    std::map<int, SVDBlk*> svds;
    for (int i = 0; i < (int)svq.size(); ++i) svds[svq[i]] = &sb[i];

    // Collect all singular values for global truncation
    struct SVE { double sv; int q; int k; };
    std::vector<SVE> all_sv;
    for (auto& [q, s] : svds)
        for (int k = 0; k < s->s.size(); ++k)
            all_sv.push_back({s->s[k], q, k});
    std::sort(all_sv.begin(), all_sv.end(), [](auto& a, auto& b){ return a.sv > b.sv; });

    // Apply cutoff
    double sv2_total = 0;
    for (auto& e : all_sv) sv2_total += e.sv * e.sv;
    int n_keep = (int)all_sv.size();
    if (sv2_total > 0 && cutoff > 0) {
        double discard = 0;
        for (int k = (int)all_sv.size()-1; k >= 0; --k) {
            discard += all_sv[k].sv * all_sv[k].sv;
            if (discard / sv2_total > cutoff * cutoff) { n_keep = k+1; break; }
        }
    }
    n_keep = std::max(1, std::min(n_keep, max_bond));
    // Per-value floor (matches e.g. tenpy's svd_min): drop Schmidt values below
    // cutoff*||theta||.  Without this, the cumulative criterion above keeps a long
    // tail of negligible states, inflating the bond dimension and doing wasted
    // O(chi^3) work when the physical entanglement is far below max_bond.
    if (cutoff > 0 && !all_sv.empty()) {
        double floor = cutoff * std::sqrt(sv2_total);
        while (n_keep > 1 && all_sv[n_keep - 1].sv < floor) --n_keep;
    }

    std::map<int,int> keep_per_q;
    for (int k = 0; k < n_keep; ++k) keep_per_q[all_sv[k].q]++;

    BlockSVDResult res;
    res.trunc_err = 0;
    for (int k = n_keep; k < (int)all_sv.size(); ++k)
        res.trunc_err += all_sv[k].sv * all_sv[k].sv;

    // Build new bond space and extract U/Vt blocks
    std::vector<std::pair<int,int>> new_sectors;
    for (auto& [q_mid, svd] : svds) {
        int keep = keep_per_q.count(q_mid) ? keep_per_q[q_mid] : 0;
        if (keep == 0) continue;
        int dL_up = bs_L.dim(q_mid - 1);
        int dL_dn = bs_L.dim(q_mid + 1);
        const Mat& U = svd->U;
        res.U_up[q_mid] = U.topRows(dL_up).leftCols(keep);
        res.U_dn[q_mid] = U.bottomRows(dL_dn).leftCols(keep);
        res.Vt[q_mid]   = svd->V.adjoint().topRows(keep);
        res.sv[q_mid]   = svd->s.head(keep);
        new_sectors.emplace_back(q_mid, keep);
    }
    std::sort(new_sectors.begin(), new_sectors.end());
    res.new_mid_bs = BondSpace::from_pairs(new_sectors);
    return res;
}

// -- Build theta from block MPS -----------------------------------------------

static std::map<int,Mat> form_theta(
    const std::map<int,Mat>& block_Ai,
    const std::map<int,Mat>& block_Ai1,
    const std::map<int,RVec>& sv_mid,
    const BondSpace& bs_L,
    const BondSpace& bs_mid,
    const BondSpace& bs_R2,
    bool absorb_sv_into_Ai1)
{
    std::map<int,Mat> theta;

    // Find all q_mid values
    std::set<int> q_mids;
    for (auto& [q_L, _] : block_Ai) {
        int q_up = q_L + 1, q_dn = q_L - 1;
        if (bs_mid.dim(q_up) > 0) q_mids.insert(q_up);
        if (bs_mid.dim(q_dn) > 0) q_mids.insert(q_dn);
    }

    for (int q_mid : q_mids) {
        int dL_up = bs_L.dim(q_mid - 1);
        int dL_dn = bs_L.dim(q_mid + 1);
        int dR_up = bs_R2.dim(q_mid + 1);
        int dR_dn = bs_R2.dim(q_mid - 1);
        int n_row = dL_up + dL_dn;
        int n_col = dR_up + dR_dn;
        if (n_row == 0 || n_col == 0) continue;

        Mat th = Mat::Zero(n_row, n_col);

        // Get A[i+1][q_mid] with optional sv absorption
        auto it1 = block_Ai1.find(q_mid);
        if (it1 == block_Ai1.end()) continue;
        Mat Ai1 = it1->second;  // (n_keep_mid, dR_up+dR_dn)
        if (absorb_sv_into_Ai1) {
            auto it_sv = sv_mid.find(q_mid);
            if (it_sv != sv_mid.end()) {
                const RVec& sv = it_sv->second;
                int sv_len = std::min((int)sv.size(), (int)Ai1.rows());
                for (int k = 0; k < sv_len; ++k) Ai1.row(k) *= sv[k];
            }
        }

        // For each (s1, q_L): A[i][q_L] contributes rows to th
        for (int s1 = 0; s1 < 2; ++s1) {
            int s1z = (s1 == 0) ? +1 : -1;
            int q_L = q_mid - s1z;
            int dL  = bs_L.dim(q_L);
            if (dL == 0) continue;
            auto it0 = block_Ai.find(q_L);
            if (it0 == block_Ai.end()) continue;
            const Mat& Ai = it0->second;   // (dL, dR_up_mid + dR_dn_mid)
            int dR_up_mid = bs_mid.dim(q_L + 1);
            int dR_dn_mid = bs_mid.dim(q_L - 1);
            int col_off_Ai = (s1 == 0) ? 0 : dR_up_mid;
            int n_keep_mid = bs_mid.dim(q_mid);
            if (col_off_Ai + n_keep_mid > Ai.cols()) continue;
            Mat Ai_s1 = Ai.block(0, col_off_Ai, dL, n_keep_mid); // (dL, n_keep_mid)
            if (!absorb_sv_into_Ai1) {
                // Right sweep: absorb sv into Ai_s1
                auto it_sv = sv_mid.find(q_mid);
                if (it_sv != sv_mid.end()) {
                    const RVec& sv = it_sv->second;
                    int sv_len = std::min((int)sv.size(), n_keep_mid);
                    for (int k = 0; k < sv_len; ++k) Ai_s1.col(k) *= sv[k];
                }
            }
            int row_off = (s1 == 0) ? 0 : dL_up;
            if (row_off + dL > n_row) continue;
            th.block(row_off, 0, dL, n_col).noalias() += Ai_s1 * Ai1;
        }
        if (th.norm() > 1e-14) theta[q_mid] = th;
    }
    return theta;
}

// ===== Block-sparse TDVP2 (two-site, bond-adaptive) time evolution ==========
// Reuses the block kernels (form_theta merge, build/apply_heff_plan two-site,
// apply_heff_1site one-site, block_svd_truncate, env updates) with qx_tdvp::expmv
// as the local time-step. Only the S=Cx instantiation is used (time evolution).
struct TDVP2Out {
    double energy = 0;
    std::vector<double> times, stag;   // staggered magnetization (1/N) sum (-1)^i <sigma^z_i>
    int max_bond = 0;
    double wall_s = 0;
};

// <sigma^z_i> for a center tensor M (block_A layout): up cols (+1) vs dn cols (-1).
static double center_sz(const std::map<int,Mat>& M, const BondSpace& bsR) {
    double up = 0, dn = 0;
    for (auto& [q_L, blk] : M) {
        int dR_up = bsR.dim(q_L + 1);
        int dR_dn = bsR.dim(q_L - 1);
        if (dR_up > 0) up += blk.leftCols(dR_up).squaredNorm();
        if (dR_dn > 0) dn += blk.rightCols(dR_dn).squaredNorm();
    }
    double z = up + dn;
    return z > 0 ? (up - dn) / z : 0.0;
}

// staggered magnetization on a COPY (center sweep L->R, non-destructive).
static double measure_stag(std::vector<std::map<int,Mat>> A,
                           std::vector<BondSpace> bond, int n) {
    double m = 0;
    for (int i = 0; i < n; ++i) {
        m += ((i % 2 == 0) ? 1.0 : -1.0) * center_sz(A[i], bond[i+1]);
        if (i < n - 1) {
            SiteMove mv = singlesite_split_LR(A[i], bond[i], bond[i+1]);
            A[i] = mv.A; bond[i+1] = mv.new_mid;
            // absorb centre matrices C[q_R] into A[i+1] (left bond)
            std::map<int,Mat> next;
            for (auto& [q_R, Cblk] : mv.C) {
                auto it = A[i+1].find(q_R);
                if (it != A[i+1].end()) next[q_R] = Cblk * it->second;
            }
            A[i+1] = next;
        }
    }
    return m / n;
}

// ----- reusable block-TDVP2 helpers (shared by run_tdvp2 and METTS) ----------
// Full block-MPS state: site tensors, bond spaces, and L/R environments.
struct BMPS {
    std::vector<std::map<int,Mat>> A;
    std::vector<BondSpace> bond;
    std::vector<BlockEnv> L, R;
    int n = 0;
};

static void bt_expand_theta(std::map<int,Mat>& theta, const BondSpace& bsL, const BondSpace& bsR2) {
    for (int qi = 0; qi < bsL.n_sectors(); ++qi) {
        int q_L = bsL.qns[qi];
        for (int s = 0; s < 2; ++s) {
            int q_mid = q_L + ((s==0)?+1:-1);
            if (theta.count(q_mid)) continue;
            int nr = bsL.dim(q_mid-1) + bsL.dim(q_mid+1);
            int nc = bsR2.dim(q_mid+1) + bsR2.dim(q_mid-1);
            if (nr > 0 && nc > 0) theta[q_mid] = Mat::Zero(nr, nc);
        }
    }
}
static void bt_evolve_theta(std::map<int,Mat>& theta, const HeffPlan& plan, Cx tau) {
    auto mv = [&](const Vec& vin) -> Vec {
        std::map<int,Mat> tb; vec_to_theta(vin, theta, tb);
        auto raw = apply_heff_plan(plan, tb);
        std::map<int,Mat> outn;
        for (auto& [q, ref] : theta) { auto it = raw.find(q);
            outn[q] = (it!=raw.end()&&it->second.rows()==ref.rows()&&it->second.cols()==ref.cols())
                      ? it->second : Mat::Zero(ref.rows(), ref.cols()); }
        return theta_to_vec(outn);
    };
    Vec ev = qx_tdvp::expmv(tau, mv, theta_to_vec(theta));
    std::map<int,Mat> out; vec_to_theta(ev, theta, out); theta = out;
}
static void bt_evolve_site(std::map<int,Mat>& M, const BlockEnv& Le, const MPOSite& W,
                           const BlockEnv& Re, const BondSpace& bsL, const BondSpace& bsR, Cx tau) {
    auto mv = [&](const Vec& vin) -> Vec {
        std::map<int,Mat> tb; vec_to_theta(vin, M, tb);
        auto raw = apply_heff_1site(Le, W, Re, tb, bsL, bsR);
        std::map<int,Mat> outn;
        for (auto& [q, ref] : M) { auto it = raw.find(q);
            outn[q] = (it!=raw.end()&&it->second.rows()==ref.rows()&&it->second.cols()==ref.cols())
                      ? it->second : Mat::Zero(ref.rows(), ref.cols()); }
        return theta_to_vec(outn);
    };
    Vec ev = qx_tdvp::expmv(tau, mv, theta_to_vec(M));
    std::map<int,Mat> out; vec_to_theta(ev, M, out); M = out;
}
static std::map<int,Mat> bt_A_from_U(const BlockSVDResult& svd, const BondSpace& bsL, const BondSpace& bsMid) {
    std::map<int,Mat> A;
    for (int qi = 0; qi < bsL.n_sectors(); ++qi) {
        int q_L = bsL.qns[qi], dL = bsL.dims[qi];
        int n_up = bsMid.dim(q_L+1), n_dn = bsMid.dim(q_L-1);
        Mat blk = Mat::Zero(dL, n_up + n_dn);
        if (n_up > 0 && svd.U_up.count(q_L+1)) blk.leftCols(n_up)  = svd.U_up.at(q_L+1);
        if (n_dn > 0 && svd.U_dn.count(q_L-1)) blk.rightCols(n_dn) = svd.U_dn.at(q_L-1);
        A[q_L] = blk;
    }
    return A;
}
static std::map<int,Mat> bt_center_LR(const BlockSVDResult& svd) {     // M (at i+1) = sv * Vt
    std::map<int,Mat> M;
    for (auto& [q, Vt] : svd.Vt) {
        Mat m = Vt;
        auto it = svd.sv.find(q);
        if (it != svd.sv.end()) { const RVec& s = it->second;
            int L2 = std::min((int)s.size(),(int)m.rows()); for (int k=0;k<L2;++k) m.row(k)*=s[k]; }
        M[q] = m;
    }
    return M;
}
static std::map<int,Mat> bt_center_RL(const BlockSVDResult& svd, const BondSpace& bsL, const BondSpace& bsMid) {
    std::map<int,Mat> M;
    for (int qi = 0; qi < bsL.n_sectors(); ++qi) {
        int q_L = bsL.qns[qi], dL = bsL.dims[qi];
        int n_up = bsMid.dim(q_L+1), n_dn = bsMid.dim(q_L-1);
        Mat blk = Mat::Zero(dL, n_up + n_dn);
        auto scaled = [&](Mat u, int q) {
            auto it = svd.sv.find(q);
            if (it != svd.sv.end()) { const RVec& s=it->second;
                int L2=std::min((int)s.size(),(int)u.cols()); for(int k=0;k<L2;++k) u.col(k)*=s[k]; }
            return u;
        };
        if (n_up > 0 && svd.U_up.count(q_L+1)) blk.leftCols(n_up)  = scaled(svd.U_up.at(q_L+1), q_L+1);
        if (n_dn > 0 && svd.U_dn.count(q_L-1)) blk.rightCols(n_dn) = scaled(svd.U_dn.at(q_L-1), q_L-1);
        M[q_L] = blk;
    }
    return M;
}

// Build a block-MPS for an arbitrary Sz product state (sz[i] = +1 up, -1 dn).
static BMPS cps_init(const std::vector<MPOSite>& mpo, int n, const std::vector<int>& sz) {
    BMPS s; s.n = n;
    std::vector<int> qb(n + 1); qb[0] = 0;
    for (int i = 0; i < n; ++i) qb[i+1] = qb[i] + sz[i];
    s.bond.resize(n + 1);
    for (int i = 0; i <= n; ++i) s.bond[i] = BondSpace::from_pairs({{qb[i], 1}});
    s.A.resize(n);
    for (int i = 0; i < n; ++i) {
        int q_L = qb[i];
        int dR_up = s.bond[i+1].dim(q_L + 1), dR_dn = s.bond[i+1].dim(q_L - 1);
        Mat blk = Mat::Zero(1, dR_up + dR_dn);
        blk(0, (sz[i] == +1) ? 0 : dR_up) = S(1);
        s.A[i][q_L] = blk;
    }
    s.L.assign(n + 1, BlockEnv()); s.R.assign(n + 1, BlockEnv());
    { BlockEnv L0; L0.bs = s.bond[0]; L0.blocks[0][qb[0]] = Mat::Constant(1,1,S(1)); s.L[0] = L0; }
    { BlockEnv Rn; Rn.bs = s.bond[n]; Rn.blocks[0][qb[n]] = Mat::Constant(1,1,S(1)); s.R[n] = Rn; }
    for (int i = n - 1; i >= 0; --i)
        s.R[i] = block_update_right(s.R[i+1], s.A[i], s.bond[i], s.bond[i+1], mpo[i]);
    return s;
}

// One symmetric second-order block TDVP2 step (mutates s; centre returns to 0).
static void tdvp2_step_blocks(BMPS& s, const std::vector<MPOSite>& mpo,
                              double dt, bool imag, int chi_max, double cutoff) {
    const int n = s.n;
    const Cx fwd = imag ? Cx(-dt/2,0) : Cx(0,-dt/2);
    const Cx bwd = imag ? Cx(+dt/2,0) : Cx(0,+dt/2);
    const std::map<int,RVec> no_sv;
    auto& A = s.A; auto& bond = s.bond; auto& L = s.L; auto& R = s.R;
    for (int i = 0; i < n - 1; ++i) {                      // left -> right
        auto theta = form_theta(A[i], A[i+1], no_sv, bond[i], bond[i+1], bond[i+2], true);
        bt_expand_theta(theta, bond[i], bond[i+2]);
        WComb wc = build_wcomb(mpo[i], mpo[i+1]);
        std::vector<int> kk; for (auto& [q,_] : theta) kk.push_back(q);
        HeffPlan plan = build_heff_plan(L[i], mpo[i], mpo[i+1], R[i+2], wc, kk, bond[i], bond[i+2]);
        bt_evolve_theta(theta, plan, fwd);
        auto svd = block_svd_truncate(theta, bond[i], bond[i+2], chi_max, cutoff);
        bond[i+1] = svd.new_mid_bs;
        A[i] = bt_A_from_U(svd, bond[i], bond[i+1]);
        L[i+1] = block_update_left(L[i], A[i], bond[i], bond[i+1], mpo[i]);
        auto M = bt_center_LR(svd);
        if (i < n - 2) bt_evolve_site(M, L[i+1], mpo[i+1], R[i+2], bond[i+1], bond[i+2], bwd);
        A[i+1] = M;
    }
    for (int i = n - 2; i >= 0; --i) {                    // right -> left
        auto theta = form_theta(A[i], A[i+1], no_sv, bond[i], bond[i+1], bond[i+2], true);
        bt_expand_theta(theta, bond[i], bond[i+2]);
        WComb wc = build_wcomb(mpo[i], mpo[i+1]);
        std::vector<int> kk; for (auto& [q,_] : theta) kk.push_back(q);
        HeffPlan plan = build_heff_plan(L[i], mpo[i], mpo[i+1], R[i+2], wc, kk, bond[i], bond[i+2]);
        bt_evolve_theta(theta, plan, fwd);
        auto svd = block_svd_truncate(theta, bond[i], bond[i+2], chi_max, cutoff);
        bond[i+1] = svd.new_mid_bs;
        A[i+1] = svd.Vt;
        R[i+1] = block_update_right(R[i+2], A[i+1], bond[i+1], bond[i+2], mpo[i+1]);
        auto M = bt_center_RL(svd, bond[i], bond[i+1]);
        if (i > 0) bt_evolve_site(M, L[i], mpo[i], R[i+1], bond[i], bond[i+1], bwd);
        A[i] = M;
    }
    if (imag) {   // renormalize (centre at site 0)
        double nz = 0; for (auto& [q,b] : A[0]) nz += b.squaredNorm();
        nz = std::sqrt(nz); if (nz > 0) for (auto& [q,b] : A[0]) b /= nz;
    }
}

// ===== Block-sparse TEBD (Trotter gates -- the fast nearest-neighbour route) ==
// A two-site Sz-conserving gate G = exp(tau * h_bond) for the XXZ bond
//   h = J (sigma^x sigma^x + sigma^y sigma^y) + J*delta sigma^z sigma^z
// (delta=1 is the isotropic Heisenberg point). |uu>,|dd> -> eigenvalue J*delta;
// the Sz=0 block {|ud>,|du>} is [[-J*delta, 2J],[2J, -J*delta]] with eigenvalues
// -J*delta +- 2J. tau = -i dt (real time) or -dt (imag).
struct Gate { S pdiag; S g00, g01, g10, g11; };
static Gate build_xxz_gate(double J, double delta, Cx tau) {
    const S Jd = S(J * delta), J2 = S(2.0 * J), t = S(tau);
    S pdiag = std::exp(t * Jd);                 // |uu>,|dd> phase
    S pre   = std::exp(t * (-Jd));              // -J*delta diagonal of the Sz=0 block
    S c = std::cosh(t * J2), s = std::sinh(t * J2);
    return Gate{ pdiag, pre * c, pre * s, pre * s, pre * c };
}

// Apply a homogeneous nearest-neighbour gate to the two-site block tensor theta
// (expanded). For each left charge a: the |ud> block (in theta[a+1]) and the |du>
// block (in theta[a-1]) -- both (dim a)x(dim a), charge (q_L=a,q_R=a) -- are mixed
// by the Sz=0 2x2 gate; the |uu> and |dd> diagonal blocks get the phase pdiag.
static void apply_gate_theta(std::map<int,Mat>& theta, const BondSpace& bsL,
                             const BondSpace& bsR, const Gate& g) {
    for (int ai = 0; ai < bsL.n_sectors(); ++ai) {
        int a = bsL.qns[ai];
        int dLa = bsL.dim(a);
        if (dLa == 0) continue;
        auto itP = theta.find(a + 1);   // holds |uu> and |ud> rows for q_L=a
        auto itM = theta.find(a - 1);   // holds |du> and |dd> rows for q_L=a
        int dRa = bsR.dim(a);           // q_R=a dimension (shared by |ud>,|du>)
        if (itP != theta.end()) {       // |uu> phase: rows [0,dLa), cols [0, dim(a+2))
            int dRup = bsR.dim(a + 2);
            if (dRup > 0) itP->second.block(0, 0, dLa, dRup) *= g.pdiag;
        }
        if (itM != theta.end()) {       // |dd> phase
            int dLupM = bsL.dim(a - 2);
            int dRdnM = bsR.dim(a - 2);
            if (dRdnM > 0) itM->second.block(dLupM, dRa, dLa, dRdnM) *= g.pdiag;
        }
        if (dRa > 0 && itP != theta.end() && itM != theta.end()) {   // Sz=0 mix
            int dRup = bsR.dim(a + 2);  // |ud> cols start after the s2=up block
            int dLupM = bsL.dim(a - 2); // |du> rows start after the s1=up block
            Mat Tud = itP->second.block(0, dRup, dLa, dRa);
            Mat Tdu = itM->second.block(dLupM, 0, dLa, dRa);
            itP->second.block(0, dRup, dLa, dRa)  = g.g00 * Tud + g.g01 * Tdu;
            itM->second.block(dLupM, 0, dLa, dRa) = g.g10 * Tud + g.g11 * Tdu;
        }
    }
}

// One symmetric second-order block TEBD step (L->R then R->L with half gates, so
// every bond is acted on with the full dt and the sweep is time-symmetric).
// No environments and no Krylov -- just a gate + SVD per bond, the reason TEBD is
// cheaper than TDVP for nearest-neighbour models.
static void tebd2_step_blocks(BMPS& s, double dt, bool imag, int chi_max,
                              double cutoff, double J = 1.0, double delta = 1.0) {
    const int n = s.n;
    const Cx tau = imag ? Cx(-dt/2, 0) : Cx(0, -dt/2);
    const Gate g = build_xxz_gate(J, delta, tau);
    const std::map<int,RVec> no_sv;
    auto& A = s.A; auto& bond = s.bond;
    const bool prof = (std::getenv("QX_TEBD_PROFILE") != nullptr);
    using pclk = std::chrono::steady_clock;
    auto ptick = [&](double& acc, pclk::time_point t0){ if (prof) acc += std::chrono::duration<double>(pclk::now()-t0).count(); };
    for (int i = 0; i < n - 1; ++i) {                      // left -> right
        auto tt = pclk::now();
        auto theta = form_theta(A[i], A[i+1], no_sv, bond[i], bond[i+1], bond[i+2], true);
        bt_expand_theta(theta, bond[i], bond[i+2]);
        ptick(g_tebd_theta, tt); tt = pclk::now();
        apply_gate_theta(theta, bond[i], bond[i+2], g);
        ptick(g_tebd_gate, tt); tt = pclk::now();
        auto svd = block_svd_truncate(theta, bond[i], bond[i+2], chi_max, cutoff);
        ptick(g_tebd_svd, tt); tt = pclk::now();
        bond[i+1] = svd.new_mid_bs;
        A[i]   = bt_A_from_U(svd, bond[i], bond[i+1]);
        A[i+1] = bt_center_LR(svd);
        ptick(g_tebd_recomb, tt);
    }
    for (int i = n - 2; i >= 0; --i) {                    // right -> left
        auto tt = pclk::now();
        auto theta = form_theta(A[i], A[i+1], no_sv, bond[i], bond[i+1], bond[i+2], true);
        bt_expand_theta(theta, bond[i], bond[i+2]);
        ptick(g_tebd_theta, tt); tt = pclk::now();
        apply_gate_theta(theta, bond[i], bond[i+2], g);
        ptick(g_tebd_gate, tt); tt = pclk::now();
        auto svd = block_svd_truncate(theta, bond[i], bond[i+2], chi_max, cutoff);
        ptick(g_tebd_svd, tt); tt = pclk::now();
        bond[i+1] = svd.new_mid_bs;
        A[i+1] = svd.Vt;
        A[i]   = bt_center_RL(svd, bond[i], bond[i+1]);
        ptick(g_tebd_recomb, tt);
    }
    if (imag) {   // renormalize (centre at site 0)
        double nz = 0; for (auto& [q,b] : A[0]) nz += b.squaredNorm();
        nz = std::sqrt(nz); if (nz > 0) for (auto& [q,b] : A[0]) b /= nz;
    }
}

// Energy <H> via the two-site H_eff at the centre (assumes centre at site 0).
static double bt_energy(const BMPS& s, const std::vector<MPOSite>& mpo) {
    const std::map<int,RVec> no_sv;
    auto theta = form_theta(s.A[0], s.A[1], no_sv, s.bond[0], s.bond[1], s.bond[2], true);
    WComb wc = build_wcomb(mpo[0], mpo[1]);
    std::vector<int> kk; for (auto& [q,_] : theta) kk.push_back(q);
    HeffPlan plan = build_heff_plan(s.L[0], mpo[0], mpo[1], s.R[2], wc, kk, s.bond[0], s.bond[2]);
    auto Hth = apply_heff_plan(plan, theta);
    Vec tv = theta_to_vec(theta);
    std::map<int,Mat> Hn;
    for (auto& [q,ref] : theta) { auto it=Hth.find(q);
        Hn[q] = (it!=Hth.end()&&it->second.rows()==ref.rows()&&it->second.cols()==ref.cols())? it->second : Mat::Zero(ref.rows(),ref.cols()); }
    return rp(tv.dot(theta_to_vec(Hn))) / rp(tv.dot(tv));
}

static TDVP2Out run_tdvp2(const std::vector<MPOSite>& mpo, int n,
                          double dt, int n_steps, int chi_max, double cutoff,
                          bool imag) {
    static_assert(std::is_same_v<S, Cx>, "TDVP2 requires complex scalar");
    using clk = std::chrono::steady_clock;
    std::vector<int> neel(n); for (int i=0;i<n;++i) neel[i] = (i%2==0)?+1:-1;
    BMPS s = cps_init(mpo, n, neel);

    TDVP2Out out;
    out.times.push_back(0.0);
    out.stag.push_back(measure_stag(s.A, s.bond, n));
    auto t0 = clk::now();
    for (int step = 0; step < n_steps; ++step) {
        tdvp2_step_blocks(s, mpo, dt, imag, chi_max, cutoff);
        out.times.push_back((step+1)*dt);
        out.stag.push_back(measure_stag(s.A, s.bond, n));
    }
    out.wall_s = std::chrono::duration<double>(clk::now() - t0).count();
    out.energy = bt_energy(s, mpo);
    out.max_bond = 0;
    for (auto& b : s.bond) { int t=0; for (int d : b.dims) t+=d; out.max_bond = std::max(out.max_bond, t); }
    return out;
}

// Block-sparse TEBD2 Neel quench (same interface as run_tdvp2; mpo only used for
// the initial state and the final energy -- the evolution is gate-based).
// measure_stride: record the staggered magnetization every `measure_stride` steps
// (1 = every step; <=0 = endpoints only -- isolates the bare evolution cost).
static TDVP2Out run_tebd2(const std::vector<MPOSite>& mpo, int n,
                          double dt, int n_steps, int chi_max, double cutoff,
                          bool imag, int measure_stride = 1,
                          double J = 1.0, double delta = 1.0) {
    static_assert(std::is_same_v<S, Cx>, "TEBD2 requires complex scalar");
    using clk = std::chrono::steady_clock;
    std::vector<int> neel(n); for (int i=0;i<n;++i) neel[i] = (i%2==0)?+1:-1;
    BMPS s = cps_init(mpo, n, neel);
    const bool prof = (std::getenv("QX_TEBD_PROFILE") != nullptr);
    if (prof) { g_tebd_theta=g_tebd_gate=g_tebd_svd=g_tebd_recomb=0; g_tebd_maxblk=g_tebd_nsvd=0; }

    TDVP2Out out;
    auto t0 = clk::now();
    for (int step = 0; step < n_steps; ++step) {
        tebd2_step_blocks(s, dt, imag, chi_max, cutoff, J, delta);
        if (measure_stride > 0 && (step % measure_stride == 0 || step == n_steps - 1)) {
            out.times.push_back((step+1)*dt);
            out.stag.push_back(measure_stag(s.A, s.bond, n));
        }
    }
    out.wall_s = std::chrono::duration<double>(clk::now() - t0).count();
    if (prof) {
        double tot = g_tebd_theta+g_tebd_gate+g_tebd_svd+g_tebd_recomb;
        std::fprintf(stderr, "[QX_TEBD_PROFILE n=%d chi=%d] evo=%.3fs theta=%.3f(%.0f%%) gate=%.3f(%.0f%%) svd=%.3f(%.0f%%) recomb=%.3f(%.0f%%) maxblk=%d nsvd=%lld\n",
            n, chi_max, out.wall_s, g_tebd_theta, 100*g_tebd_theta/tot, g_tebd_gate, 100*g_tebd_gate/tot,
            g_tebd_svd, 100*g_tebd_svd/tot, g_tebd_recomb, 100*g_tebd_recomb/tot,
            (int)g_tebd_maxblk, g_tebd_nsvd);
    }
    rebuild_mpo_envs(s, mpo);   // TEBD does not maintain MPO envs; bt_energy needs them
    out.energy = bt_energy(s, mpo);
    out.max_bond = 0;
    for (auto& b : s.bond) { int t=0; for (int d : b.dims) t+=d; out.max_bond = std::max(out.max_bond, t); }
    return out;
}

// ===== Dynamical structure factor S(k,omega) ================================
// Apply sigma^z on one site of a block-MPS in place (diagonal: up -> +1, dn -> -1).
// sigma^z is its own inverse and unitary, so the right-canonical property and the
// norm are preserved; only the MPO environments of the modified state go stale.
static void apply_sz_site(BMPS& s, int site) {
    for (auto& [q_L, blk] : s.A[site]) {
        int dR_dn = s.bond[site+1].dim(q_L - 1);
        int dR_up = s.bond[site+1].dim(q_L + 1);
        if (dR_dn > 0) blk.rightCols(dR_dn) *= S(-1);   // dn columns -> -1
        (void)dR_up;
    }
}

// Rebuild the MPO environments of a block-MPS (centre on site 0) after an in-place
// edit, so it can be time-evolved by tdvp2_step_blocks.
static void rebuild_mpo_envs(BMPS& s, const std::vector<MPOSite>& mpo) {
    int n = s.n;
    s.L.assign(n + 1, BlockEnv()); s.R.assign(n + 1, BlockEnv());
    { BlockEnv L0; L0.bs = s.bond[0]; L0.blocks[0][0] = Mat::Constant(1,1,S(1)); s.L[0] = L0; }
    { BlockEnv Rn; Rn.bs = s.bond[n];
      for (int qi = 0; qi < s.bond[n].n_sectors(); ++qi)
          Rn.blocks[0][s.bond[n].qns[qi]] = Mat::Constant(1,1,S(1));
      s.R[n] = Rn; }
    for (int i = n - 1; i >= 0; --i)
        s.R[i] = block_update_right(s.R[i+1], s.A[i], s.bond[i], s.bond[i+1], mpo[i]);
}

// Two-MPS overlap left environments OL[i][q] = <bra[0..i-1] | ket[0..i-1]> as a
// (d_bra_i(q) x d_ket_i(q)) block per shared Sz sector q. bra and ket share the
// total Sz, so the overlap is block-diagonal in Sz.
static std::vector<std::map<int,Mat>> overlap_left_envs(const BMPS& bra, const BMPS& ket) {
    int n = bra.n;
    std::vector<std::map<int,Mat>> OL(n + 1);
    OL[0][0] = Mat::Constant(1, 1, S(1));
    for (int i = 0; i < n; ++i) {
        for (auto& [q_L, Olk] : OL[i]) {
            auto ib = bra.A[i].find(q_L); auto ik = ket.A[i].find(q_L);
            if (ib == bra.A[i].end() || ik == ket.A[i].end()) continue;
            const Mat& Ab = ib->second; const Mat& Ak = ik->second;
            for (int s = 0; s < 2; ++s) {
                int sz = (s == 0) ? +1 : -1, q_R = q_L + sz;
                int ub = bra.bond[i+1].dim(q_L+1), db = bra.bond[i+1].dim(q_L-1);
                int uk = ket.bond[i+1].dim(q_L+1), dk = ket.bond[i+1].dim(q_L-1);
                int wb = (sz==+1)?ub:db, wk = (sz==+1)?uk:dk;
                if (wb == 0 || wk == 0) continue;
                Eigen::Ref<const Mat> Abs = (sz==+1)? Ab.leftCols(ub) : Ab.rightCols(db);
                Eigen::Ref<const Mat> Aks = (sz==+1)? Ak.leftCols(uk) : Ak.rightCols(dk);
                Mat contrib = Abs.adjoint() * (Olk * Aks);   // (wb x wk)
                auto& dst = OL[i+1][q_R];
                if (dst.size()==0) dst = contrib; else dst += contrib;
            }
        }
    }
    return OL;
}

// Two-MPS overlap right environments OR[i][q] = <bra[i..n-1] | ket[i..n-1]>.
static std::vector<std::map<int,Mat>> overlap_right_envs(const BMPS& bra, const BMPS& ket) {
    int n = bra.n;
    std::vector<std::map<int,Mat>> OR(n + 1);
    for (int qi = 0; qi < bra.bond[n].n_sectors(); ++qi)
        OR[n][bra.bond[n].qns[qi]] = Mat::Constant(1, 1, S(1));
    for (int i = n - 1; i >= 0; --i) {
        for (auto& [q_L, Ab] : bra.A[i]) {
            auto ik = ket.A[i].find(q_L);
            if (ik == ket.A[i].end()) continue;
            const Mat& Ak = ik->second;
            for (int s = 0; s < 2; ++s) {
                int sz = (s == 0) ? +1 : -1, q_R = q_L + sz;
                auto it = OR[i+1].find(q_R);
                if (it == OR[i+1].end()) continue;
                int ub = bra.bond[i+1].dim(q_L+1), db = bra.bond[i+1].dim(q_L-1);
                int uk = ket.bond[i+1].dim(q_L+1), dk = ket.bond[i+1].dim(q_L-1);
                int wb = (sz==+1)?ub:db, wk = (sz==+1)?uk:dk;
                if (wb == 0 || wk == 0) continue;
                Eigen::Ref<const Mat> Abs = (sz==+1)? Ab.leftCols(ub) : Ab.rightCols(db);
                Eigen::Ref<const Mat> Aks = (sz==+1)? Ak.leftCols(uk) : Ak.rightCols(dk);
                // conj on the bra (consistent with OL): OR = sum_s conj(Abs) OR' Aks^T
                Mat contrib = Abs.conjugate() * (it->second * Aks.transpose());  // (d_bra_L x d_ket_L)
                auto& dst = OR[i][q_L];
                if (dst.size()==0) dst = contrib; else dst += contrib;
            }
        }
    }
    return OR;
}

// <bra| sigma^z_j |ket> for every site j, given precomputed overlap envs.
static std::vector<S> measure_sz_all(const BMPS& bra, const BMPS& ket,
                                     const std::vector<std::map<int,Mat>>& OL,
                                     const std::vector<std::map<int,Mat>>& OR) {
    int n = bra.n;
    std::vector<S> out(n, S(0));
    for (int j = 0; j < n; ++j) {
        S val = 0;
        for (auto& [q_L, Olk] : OL[j]) {
            auto ib = bra.A[j].find(q_L); auto ik = ket.A[j].find(q_L);
            if (ib == bra.A[j].end() || ik == ket.A[j].end()) continue;
            const Mat& Ab = ib->second; const Mat& Ak = ik->second;
            for (int s = 0; s < 2; ++s) {
                int sz = (s == 0) ? +1 : -1, q_R = q_L + sz;
                auto it = OR[j+1].find(q_R);
                if (it == OR[j+1].end()) continue;
                int ub = bra.bond[j+1].dim(q_L+1), db = bra.bond[j+1].dim(q_L-1);
                int uk = ket.bond[j+1].dim(q_L+1), dk = ket.bond[j+1].dim(q_L-1);
                int wb = (sz==+1)?ub:db, wk = (sz==+1)?uk:dk;
                if (wb == 0 || wk == 0) continue;
                Eigen::Ref<const Mat> Abs = (sz==+1)? Ab.leftCols(ub) : Ab.rightCols(db);
                Eigen::Ref<const Mat> Aks = (sz==+1)? Ak.leftCols(uk) : Ak.rightCols(dk);
                Mat left = Abs.adjoint() * (Olk * Aks);          // (wb x wk)
                val += S(sz) * (left.array() * it->second.array()).sum();
            }
        }
        out[j] = val;
    }
    return out;
}

struct SpectralOut {
    std::vector<double> times;             // length n_steps+1
    int l0 = 0, n = 0, max_bond = 0;
    double E0 = 0, wall_s = 0;
    // C[t][j] = <psi0| sigma^z_j e^{-i(H-E0)t} sigma^z_{l0} |psi0>  (real, imag parts)
    std::vector<std::vector<double>> C_re, C_im;
};

// Dynamical spin structure factor data: C_{j,l0}(t) for a central source l0.
static SpectralOut run_spectral(const std::vector<MPOSite>& mpo, int n,
                                double dt, int n_steps, int chi_max, double cutoff,
                                int n_sweeps, double J = 1.0, double delta = 1.0) {
    static_assert(std::is_same_v<S, Cx>, "spectral requires complex scalar");
    using clk = std::chrono::steady_clock;
    SpectralOut out; out.n = n; out.l0 = n / 2;

    // 1. ground state (complex BMPS, centre on site 0)
    BMPS psi0;
    DMRGResult gs = run(mpo, n, chi_max, cutoff, n_sweeps, 1e-12, &psi0);
    out.E0 = gs.energy;

    // Normalisation Z = <psi0|psi0> in the overlap convention (the same machinery
    // measures the correlations, so C is normalised exactly by this ratio).
    double Z;
    { auto OLn = overlap_left_envs(psi0, psi0);
      Z = OLn[n].count(0) ? std::real(OLn[n].at(0)(0,0)) : 1.0; }
    if (std::abs(Z) < 1e-300) Z = 1.0;

    // 2. phi = sigma^z_{l0} |psi0>  (still right-canonical; TEBD needs no MPO envs)
    BMPS phi = psi0;
    apply_sz_site(phi, out.l0);

    // 3. real-time evolve, measuring C_{j,l0}(t) at each step
    auto t0 = clk::now();
    for (int step = 0; step <= n_steps; ++step) {
        double t = step * dt;
        auto OL = overlap_left_envs(psi0, phi);
        auto OR = overlap_right_envs(psi0, phi);
        auto ov = measure_sz_all(psi0, phi, OL, OR);   // <psi0| sz_j |phi(t)>
        S phase = std::exp(Cx(0.0, out.E0 * t));
        std::vector<double> re(n), im(n);
        for (int j = 0; j < n; ++j) { S c = phase * ov[j] / S(Z); re[j] = c.real(); im[j] = c.imag(); }
        out.C_re.push_back(re); out.C_im.push_back(im); out.times.push_back(t);
        for (auto& b : phi.bond) { int s=0; for (int d : b.dims) s+=d; out.max_bond = std::max(out.max_bond, s); }
        if (step < n_steps)
            tebd2_step_blocks(phi, dt, /*imag*/false, chi_max, cutoff, J, delta);  // fast NN route
    }
    out.wall_s = std::chrono::duration<double>(clk::now() - t0).count();
    return out;
}

// ===== Loschmidt echo / dynamical quantum phase transitions =================
struct LoschmidtOut {
    std::vector<double> times, L, rate;   // L(t)=|<psi0|psi(t)>|^2; rate=-ln L / n
    int n = 0, max_bond = 0;
    double E0 = 0, wall_s = 0;
};

// Quench: ground state of H_i (mpo_i), evolved under the XXZ(Jf, deltaf) gate by
// block TEBD; the return probability L(t) and rate function lambda(t) = -ln L / n
// (whose non-analyticities are dynamical quantum phase transitions). The overlaps
// are normalised by <psi0|psi0> and <psi(t)|psi(t)> so truncation norm-drift and
// the overlap-convention constant both cancel.
static LoschmidtOut run_loschmidt(const std::vector<MPOSite>& mpo_i, int n,
                                  double dt, int n_steps, int chi_max, double cutoff,
                                  int n_sweeps, double Jf, double deltaf) {
    static_assert(std::is_same_v<S, Cx>, "loschmidt requires complex scalar");
    using clk = std::chrono::steady_clock;
    LoschmidtOut out; out.n = n;

    BMPS psi0;
    DMRGResult gs = run(mpo_i, n, chi_max, cutoff, n_sweeps, 1e-12, &psi0);
    out.E0 = gs.energy;
    double Z0;
    { auto OL = overlap_left_envs(psi0, psi0);
      Z0 = OL[n].count(0) ? std::real(OL[n].at(0)(0,0)) : 1.0; }
    if (std::abs(Z0) < 1e-300) Z0 = 1.0;

    BMPS psi = psi0;
    auto t0 = clk::now();
    for (int step = 0; step <= n_steps; ++step) {
        auto OLc = overlap_left_envs(psi0, psi);
        auto OLn = overlap_left_envs(psi,  psi);
        S ov = OLc[n].count(0) ? OLc[n].at(0)(0,0) : S(0);
        double Zt = OLn[n].count(0) ? std::real(OLn[n].at(0)(0,0)) : 1.0;
        double L = (Zt > 1e-300) ? std::norm(ov) / (Z0 * Zt) : 0.0;
        out.times.push_back(step * dt);
        out.L.push_back(L);
        out.rate.push_back(L > 1e-300 ? -std::log(L) / n : 1e30);
        for (auto& b : psi.bond) { int s=0; for (int d : b.dims) s+=d; out.max_bond = std::max(out.max_bond, s); }
        if (step < n_steps)
            tebd2_step_blocks(psi, dt, /*imag*/false, chi_max, cutoff, Jf, deltaf);
    }
    out.wall_s = std::chrono::duration<double>(clk::now() - t0).count();
    return out;
}

// ===== Apply a block MPO to a block MPS (for operator-applied observables) ====
// Compress a block MPS to chi_max by symmetric SVD sweeps (centre ends at 0).
static void compress_bmps(BMPS& p, int chi_max, double cutoff) {
    const int n = p.n; const std::map<int,RVec> no_sv;
    for (int i = 0; i < n - 1; ++i) {
        auto theta = form_theta(p.A[i], p.A[i+1], no_sv, p.bond[i], p.bond[i+1], p.bond[i+2], true);
        auto svd = block_svd_truncate(theta, p.bond[i], p.bond[i+2], chi_max, cutoff);
        p.bond[i+1] = svd.new_mid_bs;
        p.A[i]   = bt_A_from_U(svd, p.bond[i], p.bond[i+1]);
        p.A[i+1] = bt_center_LR(svd);
    }
    for (int i = n - 2; i >= 0; --i) {
        auto theta = form_theta(p.A[i], p.A[i+1], no_sv, p.bond[i], p.bond[i+1], p.bond[i+2], true);
        auto svd = block_svd_truncate(theta, p.bond[i], p.bond[i+2], chi_max, cutoff);
        p.bond[i+1] = svd.new_mid_bs;
        p.A[i+1] = svd.Vt;
        p.A[i]   = bt_center_RL(svd, p.bond[i], p.bond[i+1]);
    }
}

// phi = W|psi> for a block MPO W with the Heisenberg Sz bond structure. The
// combined bond carries q_c = q_mps + HB_DELTA[mpo_bond]; the product is built
// exactly (combined bonds up to 5x larger), then SVD-compressed to chi_max.
static BMPS apply_mpo_to_bmps(const BMPS& s, const std::vector<MPOSite>& W,
                              int chi_max, double cutoff) {
    const int n = s.n;
    BMPS p; p.n = n; p.A.resize(n); p.bond.resize(n + 1);
    p.L.assign(n + 1, BlockEnv()); p.R.assign(n + 1, BlockEnv());
    auto mpodim = [&](int ib) { return (ib == 0) ? W[0].chi_l : W[ib-1].chi_r; };
    auto hbd    = [&](int ib, int b) { return (mpodim(ib) == 1) ? 0 : HB_DELTA[b]; };

    // Combined bond spaces + layout: each q_c is composed of (q_mps, b) sub-blocks.
    struct Comp { int q_mps, b, off, dim; };
    std::vector<std::map<int, std::vector<Comp>>> lay(n + 1);
    for (int i = 0; i <= n; ++i) {
        int md = mpodim(i);
        std::map<int,int> acc;
        for (int qi = 0; qi < s.bond[i].n_sectors(); ++qi) {
            int q_mps = s.bond[i].qns[qi], dq = s.bond[i].dims[qi];
            for (int b = 0; b < md; ++b) {
                int q_c = q_mps + hbd(i, b);
                lay[i][q_c].push_back({q_mps, b, acc[q_c], dq});
                acc[q_c] += dq;
            }
        }
        std::vector<std::pair<int,int>> secs(acc.begin(), acc.end());
        std::sort(secs.begin(), secs.end());
        p.bond[i] = BondSpace::from_pairs(secs);
    }
    auto find_off = [&](int ib, int q_c, int q_mps, int b) -> std::pair<int,int> {
        auto it = lay[ib].find(q_c);
        if (it != lay[ib].end())
            for (auto& c : it->second) if (c.q_mps == q_mps && c.b == b) return {c.off, c.dim};
        return {-1, 0};
    };

    for (int i = 0; i < n; ++i) {
        int mdR = mpodim(i + 1);
        for (auto& [q_cL, compsL] : lay[i]) {
            int dLc = p.bond[i].dim(q_cL);
            int dR_up = p.bond[i+1].dim(q_cL + 1), dR_dn = p.bond[i+1].dim(q_cL - 1);
            Mat blk = Mat::Zero(dLc, dR_up + dR_dn);
            for (auto& cL : compsL) {
                auto itA = s.A[i].find(cL.q_mps);
                if (itA == s.A[i].end()) continue;
                const Mat& A = itA->second;
                int mps_up = s.bond[i+1].dim(cL.q_mps + 1), mps_dn = s.bond[i+1].dim(cL.q_mps - 1);
                for (int sp = 0; sp < 2; ++sp) {            // mps physical (in)
                    int spz = (sp == 0) ? +1 : -1, q_mps_R = cL.q_mps + spz;
                    int dRs = (spz == +1) ? mps_up : mps_dn;
                    if (dRs == 0) continue;
                    Eigen::Ref<const Mat> Asp = (spz == +1) ? A.leftCols(mps_up) : A.rightCols(mps_dn);
                    for (int so = 0; so < 2; ++so) {        // output physical
                        int sz = (so == 0) ? +1 : -1, q_cR = q_cL + sz;
                        int col_base = (sz == +1) ? 0 : dR_up;
                        for (int b_R = 0; b_R < mdR; ++b_R) {
                            if (q_cR != q_mps_R + hbd(i + 1, b_R)) continue;
                            S w = mscal(W[i].W(cL.b, b_R)(so, sp));
                            if (std::abs(w) < 1e-15) continue;
                            auto [offR, dimR] = find_off(i + 1, q_cR, q_mps_R, b_R);
                            if (offR < 0) continue;
                            blk.block(cL.off, col_base + offR, cL.dim, dimR).noalias() += w * Asp;
                        }
                    }
                }
            }
            p.A[i][q_cL] = blk;
        }
    }
    compress_bmps(p, chi_max, cutoff);
    return p;
}

// ===== Spin conductivity: current-current correlator C(t) = <J(t)J(0)> ========
struct ConductivityOut {
    std::vector<double> times, C_re, C_im;   // C(t) = <psi0|J e^{-i(H-E0)t} J|psi0>
    int n = 0, max_bond = 0;
    double E0 = 0, wall_s = 0;
};
static ConductivityOut run_conductivity(const std::vector<MPOSite>& mpo_H, int n,
                                        double dt, int n_steps, int chi_max,
                                        double cutoff, int n_sweeps,
                                        double J, double delta) {
    static_assert(std::is_same_v<S, Cx>, "conductivity requires complex scalar");
    using clk = std::chrono::steady_clock;
    ConductivityOut out; out.n = n;
    BMPS psi0;
    DMRGResult gs = run(mpo_H, n, chi_max, cutoff, n_sweeps, 1e-12, &psi0);
    out.E0 = gs.energy;
    double Zpsi;   // <psi0|psi0> in the overlap convention -> normalises C
    { auto OL = overlap_left_envs(psi0, psi0);
      Zpsi = OL[n].count(0) ? std::real(OL[n].at(0)(0,0)) : 1.0; }
    if (std::abs(Zpsi) < 1e-300) Zpsi = 1.0;

    auto Wj  = qx_dmrg::build_current_mpo(n, J);
    BMPS phi0 = apply_mpo_to_bmps(psi0, Wj, chi_max, cutoff);   // phi = J|psi0>
    BMPS phi  = phi0;

    auto t0 = clk::now();
    for (int step = 0; step <= n_steps; ++step) {
        auto OL = overlap_left_envs(phi0, phi);
        S ov = OL[n].count(0) ? OL[n].at(0)(0,0) : S(0);
        S c = std::exp(Cx(0.0, out.E0 * step * dt)) * ov / S(Zpsi);
        out.times.push_back(step * dt);
        out.C_re.push_back(c.real()); out.C_im.push_back(c.imag());
        for (auto& b : phi.bond) { int s=0; for (int dm : b.dims) s+=dm; out.max_bond = std::max(out.max_bond, s); }
        if (step < n_steps) tebd2_step_blocks(phi, dt, /*imag*/false, chi_max, cutoff, J, delta);
    }
    out.wall_s = std::chrono::duration<double>(clk::now() - t0).count();
    return out;
}

// -- Block-sparse DMRG sweep --------------------------------------------------

static DMRGResult run(const std::vector<MPOSite>& mpo, int n,
                      int max_bond, double cutoff,
                      int n_sweeps, double tol,
                      BMPS* out_state = nullptr) {
    auto t_start = std::chrono::steady_clock::now();

    // Build bond spaces
    std::vector<BondSpace> bond(n + 1);
    bond[0] = BondSpace::from_pairs({{0, 1}});   // left boundary: qn=0, dim=1
    bond[n] = BondSpace::from_pairs({{0, 1}});   // right boundary: qn=0, dim=1
    for (int i = 0; i < n - 1; ++i)
        bond[i + 1] = make_heisenberg_bond_space(i, n, max_bond);

    // Initialize block MPS to the Neel state |up dn up dn ...>.
    // This matches the strategy used by the dense DMRG (init_neel_state in dmrg_dense.cpp).
    // Random initialization leads to self-consistent local minima: the random right
    // environments R[i] bias the first sweep toward an excited state, and the resulting
    // wrong-subspace MPS is self-consistent under all subsequent sweeps.  Starting from
    // the Neel state gives physically motivated environments that allow the first L->R sweep
    // to find the correct subspace.
    std::vector<std::map<int,Mat>> block_A(n);
    std::vector<std::map<int,RVec>> block_sv(n - 1);

    {
        // q_neel: accumulated Sz of the Neel state before site i
        // Neel: up at even sites, dn at odd sites -> q_neel follows 0, +1, 0, +1, ...
        int q_neel = 0;
        for (int i = 0; i < n; ++i) {
            const BondSpace& bsL = bond[i], &bsR = bond[i + 1];
            // Zero-initialise all sectors at this site
            for (int qi = 0; qi < bsL.n_sectors(); ++qi) {
                int q_L = bsL.qns[qi], dL = bsL.dims[qi];
                int dR = bsR.dim(q_L + 1) + bsR.dim(q_L - 1);
                block_A[i][q_L] = Mat::Zero(dL, dR);
            }
            // Place the Neel amplitude in the correct sector and column
            int neel_sz = (i % 2 == 0) ? +1 : -1;   // up=+1 at even, dn=-1 at odd
            if (bsL.dim(q_neel) > 0) {
                auto it = block_A[i].find(q_neel);
                if (it != block_A[i].end() && it->second.rows() > 0) {
                    int dR_up = bsR.dim(q_neel + 1);
                    int col_off = (neel_sz == +1) ? 0 : dR_up;  // up->leftCols, dn->rightCols
                    if (col_off < it->second.cols())
                        it->second(0, col_off) = S(1.0);
                }
            }
            q_neel += neel_sz;
        }
    }
    // Initialize singular values to 1 (will be overwritten after first sweep step)
    for (int i = 0; i < n - 1; ++i) {
        for (int qi = 0; qi < bond[i+1].n_sectors(); ++qi) {
            int q = bond[i+1].qns[qi], d = bond[i+1].dims[qi];
            block_sv[i][q] = RVec::Ones(d);
        }
    }

    // Initial right environments
    std::vector<BlockEnv> R(n + 1), L(n + 1);
    {
        BlockEnv Rn;
        Rn.bs = bond[n];
        // Right boundary: chi_r=1 at site n-1, so only MPO index b=0 is active.
        // R[n].blocks[0][q=0] = 1 is the right vacuum (identity scalar environment).
        Rn.blocks[0][0] = Mat::Constant(1, 1, S(1));
        R[n] = Rn;
    }
    for (int i = n - 1; i >= 0; --i)
        R[i] = block_update_right(R[i+1], block_A[i], bond[i], bond[i+1], mpo[i]);

    // Initial left environments
    {
        BlockEnv L0;
        L0.bs = bond[0];
        // Left boundary: chi_l=1 at site 0, only MPO index b=0 is active.
        L0.blocks[0][0] = Mat::Constant(1, 1, S(1));
        L[0] = L0;
    }

    // Pre-allocate all valid theta sectors (including those with zero amplitude).
    // Without this, sectors absent from form_theta output (e.g. because the previous
    // step's ground state has zero amplitude there) would never enter the Krylov space,
    // causing the optimization to be stuck in a reduced subspace.
    // Note: block_apply_heff with input {-2:0, 0:nonzero, +2:0} WILL produce nonzero
    // output in the +-2 sectors (via XY cross-coupling), so the Lanczos naturally
    // populates them from the first matvec application.
    auto expand_theta_sectors = [](std::map<int,Mat>& theta,
                                   const BondSpace& bs_L, const BondSpace& bs_R2) {
        for (int qi = 0; qi < bs_L.n_sectors(); ++qi) {
            int q_L = bs_L.qns[qi];
            for (int s = 0; s < 2; ++s) {
                int sz = (s == 0) ? +1 : -1;
                int q_mid = q_L + sz;
                if (theta.count(q_mid)) continue;
                int n_row = bs_L.dim(q_mid - 1) + bs_L.dim(q_mid + 1);
                int n_col = bs_R2.dim(q_mid + 1) + bs_R2.dim(q_mid - 1);
                if (n_row > 0 && n_col > 0)
                    theta[q_mid] = Mat::Zero(n_row, n_col);
            }
        }
    };

    DMRGResult result;
    result.energy = 0; result.truncation_error = 0; result.n_sweeps_done = 0;
    double prev_energy = 0;

    // Note: no perturbative noise. The Neel initial state plus the two-site update
    // (which lets the bond dimension grow naturally) converges to the dense result
    // to ~1e-14 for the Heisenberg chain. A crude noise term was tried earlier but
    // only inflated bond dimensions (and thus wall time) without improving accuracy;
    // if a harder Hamiltonian ever needs it, add a proper density-matrix perturbation
    // (White's correction) that is truncated together with the kept states.
    // Optional phase profiling (set env QX_DMRG_PROFILE=1).
    const bool prof = (std::getenv("QX_DMRG_PROFILE") != nullptr);
    if (prof) g_block_matvecs = 0;
    double t_theta = 0, t_lanczos = 0, t_svd = 0, t_env = 0;
    using clk = std::chrono::steady_clock;
    auto tick = [&](double& acc, clk::time_point t0) {
        if (prof) acc += std::chrono::duration<double>(clk::now() - t0).count();
    };

    // Hybrid mode: a few two-site warm-up sweeps (to set the bond dimensions), then
    // cheaper single-site refinement sweeps. Gated by QX_DMRG_SINGLESITE while it is
    // being validated; the default remains pure two-site.
    // Hybrid: two-site warm-up until the bond dimensions stabilise (the bond
    // distribution across QN sectors is what single-site can't adapt), then switch
    // to cheaper single-site refinement. Gated by QX_DMRG_SINGLESITE while validating.
    const bool use_ss = (std::getenv("QX_DMRG_NO_SINGLESITE") == nullptr);
    int min_warmup = 2, max_warmup = n_sweeps;
    if (const char* w = std::getenv("QX_DMRG_WARMUP")) min_warmup = std::atoi(w);
    bool do_ss = false;
    int  ss_start = n_sweeps;

    for (int sweep = 0; sweep < max_warmup; ++sweep) {
        double sweep_trunc = 0, E0 = 0;

        // Left-to-right half sweep
        for (int i = 0; i < n - 1; ++i) {
            auto tt = clk::now();
            auto theta = form_theta(block_A[i], block_A[i+1], block_sv[i],
                                    bond[i], bond[i+1], bond[i+2], true);
            expand_theta_sectors(theta, bond[i], bond[i+2]);
            WComb wc = build_wcomb(mpo[i], mpo[i+1]);  // fixed per bond
            std::vector<int> ket_keys; ket_keys.reserve(theta.size());
            for (auto& [q, _] : theta) ket_keys.push_back(q);
            HeffPlan plan = build_heff_plan(L[i], mpo[i], mpo[i+1], R[i+2],
                                            wc, ket_keys, bond[i], bond[i+2]);
            tick(t_theta, tt);
            auto mv = [&](const std::map<int,Mat>& v) {
                return apply_heff_plan(plan, v);
            };
            tt = clk::now();
            auto [E, psi_blk] = block_lanczos(mv, theta);
            tick(t_lanczos, tt);
            E0 = E;

            tt = clk::now();
            auto svd = block_svd_truncate(psi_blk, bond[i], bond[i+2], max_bond, cutoff);
            tick(t_svd, tt);
            sweep_trunc += svd.trunc_err;
            bond[i+1] = svd.new_mid_bs;

            // Update A[i]: for each q_mid, U_up/U_dn go into the up/dn blocks of A[i][q_L]
            for (int qi = 0; qi < bond[i].n_sectors(); ++qi) {
                int q_L = bond[i].qns[qi], dL = bond[i].dims[qi];
                int q_up = q_L + 1, q_dn = q_L - 1;
                int n_up = bond[i+1].dim(q_up), n_dn = bond[i+1].dim(q_dn);
                Mat blk = Mat::Zero(dL, n_up + n_dn);
                if (n_up > 0 && svd.U_up.count(q_up))
                    blk.leftCols(n_up) = svd.U_up.at(q_up);
                if (n_dn > 0 && svd.U_dn.count(q_dn))
                    blk.rightCols(n_dn) = svd.U_dn.at(q_dn);
                block_A[i][q_L] = blk;
            }
            block_sv[i] = svd.sv;
            block_A[i+1] = svd.Vt;

            tt = clk::now();
            L[i+1] = block_update_left(L[i], block_A[i], bond[i], bond[i+1], mpo[i]);
            tick(t_env, tt);
        }

        // Right-to-left half sweep
        for (int i = n - 2; i >= 0; --i) {
            auto tt = clk::now();
            auto theta = form_theta(block_A[i], block_A[i+1], block_sv[i],
                                    bond[i], bond[i+1], bond[i+2], false);
            expand_theta_sectors(theta, bond[i], bond[i+2]);
            WComb wc = build_wcomb(mpo[i], mpo[i+1]);  // fixed per bond
            std::vector<int> ket_keys; ket_keys.reserve(theta.size());
            for (auto& [q, _] : theta) ket_keys.push_back(q);
            HeffPlan plan = build_heff_plan(L[i], mpo[i], mpo[i+1], R[i+2],
                                            wc, ket_keys, bond[i], bond[i+2]);
            tick(t_theta, tt);
            auto mv = [&](const std::map<int,Mat>& v) {
                return apply_heff_plan(plan, v);
            };
            tt = clk::now();
            auto [E, psi_blk] = block_lanczos(mv, theta);
            tick(t_lanczos, tt);
            E0 = E;

            tt = clk::now();
            auto svd = block_svd_truncate(psi_blk, bond[i], bond[i+2], max_bond, cutoff);
            tick(t_svd, tt);
            sweep_trunc += svd.trunc_err;
            bond[i+1] = svd.new_mid_bs;

            for (int qi = 0; qi < bond[i].n_sectors(); ++qi) {
                int q_L = bond[i].qns[qi], dL = bond[i].dims[qi];
                int q_up = q_L+1, q_dn = q_L-1;
                int n_up = bond[i+1].dim(q_up), n_dn = bond[i+1].dim(q_dn);
                Mat blk = Mat::Zero(dL, n_up + n_dn);
                if (n_up > 0 && svd.U_up.count(q_up)) blk.leftCols(n_up)  = svd.U_up.at(q_up);
                if (n_dn > 0 && svd.U_dn.count(q_dn)) blk.rightCols(n_dn) = svd.U_dn.at(q_dn);
                block_A[i][q_L] = blk;
            }
            block_sv[i] = svd.sv;
            block_A[i+1] = svd.Vt;

            auto te = clk::now();
            R[i+1] = block_update_right(R[i+2], block_A[i+1], bond[i+1], bond[i+2], mpo[i+1]);
            tick(t_env, te);
        }

        result.sweep_energies.push_back(E0);
        result.truncation_error = sweep_trunc;
        double dE = (sweep > 0) ? std::abs(E0 - prev_energy) : 1e30;
        result.n_sweeps_done = sweep + 1;
        result.energy = E0;
        if (dE < tol) break;                       // fully converged (no single-site needed)
        // Switch to single-site once the two-site energy has nearly plateaued: the
        // bond distribution is then essentially set, and the cheaper single-site
        // sweeps polish the energy the rest of the way.
        if (use_ss && sweep + 1 >= min_warmup && dE < 1e-4 * std::abs(E0)) {
            do_ss = true; ss_start = sweep + 1; prev_energy = E0; break;
        }
        prev_energy = E0;
    }

    // ---- Single-site refinement phase (hybrid) ------------------------------
    if (do_ss && ss_start < n_sweeps) {
        // Centre on site 0 (after the warm-up's final R->L it sits at bond 1).
        std::map<int,Mat> M = block_A[0];
        for (auto& [q_L, blk] : M) {
            int dR_up = bond[1].dim(q_L + 1), dR_dn = bond[1].dim(q_L - 1);
            if (dR_up > 0 && block_sv[0].count(q_L + 1)) {
                const RVec& s = block_sv[0].at(q_L + 1);
                for (int c = 0; c < dR_up && c < s.size(); ++c) blk.col(c) *= s[c];
            }
            if (dR_dn > 0 && block_sv[0].count(q_L - 1)) {
                const RVec& s = block_sv[0].at(q_L - 1);
                for (int c = 0; c < dR_dn && c < s.size(); ++c) blk.col(dR_up + c) *= s[c];
            }
        }

        for (int sweep = ss_start; sweep < n_sweeps; ++sweep) {
            double E0 = 0;
            // L->R: optimise each site, move the centre rightward.
            for (int i = 0; i < n; ++i) {
                auto mv = [&](const std::map<int,Mat>& v) {
                    return apply_heff_1site(L[i], mpo[i], R[i+1], v, bond[i], bond[i+1]);
                };
                auto tt = clk::now();
                auto [E, Mopt] = block_lanczos(mv, M);
                tick(t_lanczos, tt);
                E0 = E;
                if (i < n - 1) {
                    tt = clk::now();
                    SiteMove sp = singlesite_split_LR(Mopt, bond[i], bond[i+1]);
                    tick(t_svd, tt);
                    block_A[i] = sp.A;
                    bond[i+1] = sp.new_mid;
                    std::map<int,Mat> Mn;   // absorb centre into the next site
                    for (auto& [q_R, Cb] : sp.C) {
                        auto it = block_A[i+1].find(q_R);
                        if (it != block_A[i+1].end()) Mn[q_R] = Cb * it->second;
                    }
                    M = std::move(Mn);
                    tt = clk::now();
                    L[i+1] = block_update_left(L[i], block_A[i], bond[i], bond[i+1], mpo[i]);
                    tick(t_env, tt);
                } else {
                    M = std::move(Mopt);
                }
            }
            // R->L: optimise each site, move the centre leftward.
            for (int i = n - 1; i >= 0; --i) {
                auto mv = [&](const std::map<int,Mat>& v) {
                    return apply_heff_1site(L[i], mpo[i], R[i+1], v, bond[i], bond[i+1]);
                };
                auto tt = clk::now();
                auto [E, Mopt] = block_lanczos(mv, M);
                tick(t_lanczos, tt);
                E0 = E;
                if (i > 0) {
                    BondSpace old_bond_i = bond[i];
                    tt = clk::now();
                    SiteMove sp = singlesite_split_RL(Mopt);
                    tick(t_svd, tt);
                    block_A[i] = sp.A;          // right-canonical
                    bond[i] = sp.new_mid;       // new LEFT bond of site i
                    std::map<int,Mat> Mp;       // absorb centre into the previous site
                    for (auto& [q_Lp, Ab] : block_A[i-1]) {
                        int dLp = bond[i-1].dim(q_Lp);
                        int oldRup = old_bond_i.dim(q_Lp + 1);  // up-block width of A[i-1]
                        int ku = sp.new_mid.dim(q_Lp + 1), kd = sp.new_mid.dim(q_Lp - 1);
                        Mat Mb = Mat::Zero(dLp, ku + kd);
                        if (ku > 0 && sp.C.count(q_Lp + 1))
                            Mb.leftCols(ku)  = Ab.leftCols(oldRup) * sp.C[q_Lp + 1];
                        if (kd > 0 && sp.C.count(q_Lp - 1))
                            Mb.rightCols(kd) = Ab.middleCols(oldRup, old_bond_i.dim(q_Lp - 1)) * sp.C[q_Lp - 1];
                        Mp[q_Lp] = Mb;
                    }
                    M = std::move(Mp);
                    tt = clk::now();
                    R[i] = block_update_right(R[i+1], block_A[i], bond[i], bond[i+1], mpo[i]);
                    tick(t_env, tt);
                } else {
                    M = std::move(Mopt);
                }
            }
            // Leave the centre on site 0 so the state is consistent.
            block_A[0] = M;
            for (auto& [q, blk] : block_sv[0]) blk = RVec::Ones(blk.size());

            result.sweep_energies.push_back(E0);
            if (std::abs(E0 - prev_energy) < tol) {
                result.energy = E0; result.n_sweeps_done = sweep + 1; break;
            }
            prev_energy = E0;
            result.energy = E0;
            result.n_sweeps_done = sweep + 1;
        }
    }

    // Validation of the single-site H_eff: after the final R->L half-sweep the
    // orthogonality centre is at bond 1 (block_sv[0]); putting that centre on site 0
    // gives M0 whose single-site Rayleigh quotient must equal the ground energy.
    if (std::getenv("QX_DMRG_1SITE_CHECK")) {
        std::map<int,Mat> M0 = block_A[0];
        for (auto& [q_L, blk] : M0) {
            int dR_up = bond[1].dim(q_L + 1), dR_dn = bond[1].dim(q_L - 1);
            if (dR_up > 0 && block_sv[0].count(q_L + 1)) {
                const RVec& s = block_sv[0].at(q_L + 1);
                for (int c = 0; c < dR_up && c < s.size(); ++c) blk.col(c) *= s[c];
            }
            if (dR_dn > 0 && block_sv[0].count(q_L - 1)) {
                const RVec& s = block_sv[0].at(q_L - 1);
                for (int c = 0; c < dR_dn && c < s.size(); ++c) blk.col(dR_up + c) *= s[c];
            }
        }
        auto HM = apply_heff_1site(L[0], mpo[0], R[1], M0, bond[0], bond[1]);
        double num = 0, den = 0;
        for (auto& [q, blk] : M0) {
            den += blk.squaredNorm();
            if (HM.count(q)) num += rp((blk.array().conjugate() * HM[q].array()).sum());
        }
        double e1 = (den > 1e-30) ? num / den : 0;
        printf("[QX_1SITE_CHECK n=%d] E(2site)=%.8f  E(1site H_eff)=%.8f  diff=%.2e\n",
               n, result.energy, e1, std::abs(result.energy - e1));

        // Isolation test of the L->R single-site split on a middle site tensor.
        if (n >= 4) {
            int si = n / 2;
            SiteMove mvsp = singlesite_split_LR(block_A[si], bond[si], bond[si+1]);
            // (1) reconstruction A.C == M
            double recon = 0;
            for (auto& [q_L, Mblk] : block_A[si]) {
                int dL = bond[si].dim(q_L);
                int dR_up = bond[si+1].dim(q_L+1), dR_dn = bond[si+1].dim(q_L-1);
                Mat R = Mat::Zero(dL, dR_up + dR_dn);
                int ku = mvsp.new_mid.dim(q_L+1), kd = mvsp.new_mid.dim(q_L-1);
                if (dR_up > 0 && ku > 0 && mvsp.C.count(q_L+1))
                    R.leftCols(dR_up) = mvsp.A[q_L].leftCols(ku) * mvsp.C[q_L+1];
                if (dR_dn > 0 && kd > 0 && mvsp.C.count(q_L-1))
                    R.rightCols(dR_dn) = mvsp.A[q_L].rightCols(kd) * mvsp.C[q_L-1];
                recon = std::max(recon, (R - Mblk).cwiseAbs().maxCoeff());
            }
            // (2) A left-canonical: sum_{q_L,s} A_s^dag A_s == I per q_R
            double canon = 0;
            std::map<int,Mat> G;
            for (auto& [q_L, Ablk] : mvsp.A) {
                int ku = mvsp.new_mid.dim(q_L+1), kd = mvsp.new_mid.dim(q_L-1);
                if (ku > 0) { Mat a = Ablk.leftCols(ku);
                    G[q_L+1] = (G.count(q_L+1)?G[q_L+1]:Mat::Zero(ku,ku)) + a.adjoint()*a; }
                if (kd > 0) { Mat a = Ablk.rightCols(kd);
                    G[q_L-1] = (G.count(q_L-1)?G[q_L-1]:Mat::Zero(kd,kd)) + a.adjoint()*a; }
            }
            for (auto& [q,g] : G)
                canon = std::max(canon, (g - Mat::Identity(g.rows(),g.cols())).cwiseAbs().maxCoeff());
            printf("[QX_1SITE_CHECK n=%d] split(site %d): ||A*C - M||=%.2e  ||A^dag A - I||=%.2e\n",
                   n, si, recon, canon);
        }
        fflush(stdout);
    }

    // Export the converged ground state as a right-canonical BMPS (centre on
    // site 0), with L[0]/R[i] environments rebuilt -- ready for TDVP2 / overlaps.
    if (out_state) {
        out_state->n = n;
        out_state->A = block_A;
        out_state->bond = bond;
        out_state->L.assign(n + 1, BlockEnv());
        out_state->R.assign(n + 1, BlockEnv());
        { BlockEnv L0; L0.bs = bond[0]; L0.blocks[0][0] = Mat::Constant(1,1,S(1)); out_state->L[0] = L0; }
        { BlockEnv Rn; Rn.bs = bond[n];
          for (int qi = 0; qi < bond[n].n_sectors(); ++qi)
              Rn.blocks[0][bond[n].qns[qi]] = Mat::Constant(1,1,S(1));
          out_state->R[n] = Rn; }
        for (int i = n - 1; i >= 0; --i)
            out_state->R[i] = block_update_right(out_state->R[i+1], block_A[i], bond[i], bond[i+1], mpo[i]);
    }

    auto t_end = std::chrono::steady_clock::now();
    result.wall_time_s = std::chrono::duration<double>(t_end - t_start).count();
    if (prof) {
        double tot = t_theta + t_lanczos + t_svd + t_env;
        printf("[QX_DMRG_PROFILE n=%d chi=%d] total=%.3fs  theta=%.3f(%.0f%%)  "
               "lanczos=%.3f(%.0f%%)  svd=%.3f(%.0f%%)  env=%.3f(%.0f%%)  matvecs=%lld\n",
               n, max_bond, tot,
               t_theta, 100*t_theta/tot, t_lanczos, 100*t_lanczos/tot,
               t_svd, 100*t_svd/tot, t_env, 100*t_env/tot, g_block_matvecs);
        fflush(stdout);
    }
    return result;
}

};  // template<typename S> struct BlockDMRG

// True if every MPO matrix element is real (within tolerance).
static bool mpo_is_real(const std::vector<MPOSite>& mpo) {
    for (const auto& site : mpo)
        for (const auto& op : site.ops)
            for (int a = 0; a < 2; ++a)
                for (int b = 0; b < 2; ++b)
                    if (std::abs(op(a, b).imag()) > 1e-13) return false;
    return true;
}

// Entry point: dispatch to the real or complex block-sparse core.
DMRGResult run_dmrg_blocks(const std::vector<MPOSite>& mpo, int n,
                           int max_bond, double cutoff,
                           int n_sweeps, double tol) {
    if (mpo_is_real(mpo))
        return BlockDMRG<double>::run(mpo, n, max_bond, cutoff, n_sweeps, tol);
    return BlockDMRG<Cx>::run(mpo, n, max_bond, cutoff, n_sweeps, tol);
}

// Entry point: block-sparse TDVP2 time evolution (always complex).
TDVPBlockResult run_tdvp2_blocks(const std::vector<MPOSite>& mpo, int n,
                                 double dt, int n_steps, int chi_max,
                                 double cutoff, bool imag) {
    auto o = BlockDMRG<Cx>::run_tdvp2(mpo, n, dt, n_steps, chi_max, cutoff, imag);
    TDVPBlockResult r;
    r.energy = o.energy; r.times = o.times; r.stag = o.stag;
    r.max_bond = o.max_bond; r.wall_time_s = o.wall_s;
    return r;
}

// Entry point: block-sparse TEBD2 time evolution (always complex).
TDVPBlockResult run_tebd2_blocks(const std::vector<MPOSite>& mpo, int n,
                                 double dt, int n_steps, int chi_max,
                                 double cutoff, bool imag, int measure_stride,
                                 double J, double delta) {
    auto o = BlockDMRG<Cx>::run_tebd2(mpo, n, dt, n_steps, chi_max, cutoff, imag, measure_stride, J, delta);
    TDVPBlockResult r;
    r.energy = o.energy; r.times = o.times; r.stag = o.stag;
    r.max_bond = o.max_bond; r.wall_time_s = o.wall_s;
    return r;
}

// Entry point: dynamical structure factor S(k,omega) data (always complex).
SpectralResult run_spectral_blocks(const std::vector<MPOSite>& mpo, int n,
                                   double dt, int n_steps, int chi_max,
                                   double cutoff, int n_sweeps, double J, double delta) {
    auto o = BlockDMRG<Cx>::run_spectral(mpo, n, dt, n_steps, chi_max, cutoff, n_sweeps, J, delta);
    SpectralResult r;
    r.times = o.times; r.l0 = o.l0; r.n = o.n; r.max_bond = o.max_bond;
    r.E0 = o.E0; r.wall_time_s = o.wall_s; r.C_re = o.C_re; r.C_im = o.C_im;
    return r;
}

// Entry point: Loschmidt echo / DQPT (always complex).
LoschmidtResult run_loschmidt_blocks(const std::vector<MPOSite>& mpo_i, int n,
                                     double dt, int n_steps, int chi_max,
                                     double cutoff, int n_sweeps,
                                     double Jf, double deltaf) {
    auto o = BlockDMRG<Cx>::run_loschmidt(mpo_i, n, dt, n_steps, chi_max, cutoff,
                                          n_sweeps, Jf, deltaf);
    LoschmidtResult r;
    r.times = o.times; r.L = o.L; r.rate = o.rate;
    r.n = o.n; r.max_bond = o.max_bond; r.E0 = o.E0; r.wall_time_s = o.wall_s;
    return r;
}

// Entry point: spin conductivity current-current correlator (always complex).
ConductivityResult run_conductivity_blocks(const std::vector<MPOSite>& mpo_H, int n,
                                           double dt, int n_steps, int chi_max,
                                           double cutoff, int n_sweeps,
                                           double J, double delta) {
    auto o = BlockDMRG<Cx>::run_conductivity(mpo_H, n, dt, n_steps, chi_max, cutoff,
                                             n_sweeps, J, delta);
    ConductivityResult r;
    r.times = o.times; r.C_re = o.C_re; r.C_im = o.C_im;
    r.n = o.n; r.max_bond = o.max_bond; r.E0 = o.E0; r.wall_time_s = o.wall_s;
    return r;
}

} // namespace qx_dmrg

