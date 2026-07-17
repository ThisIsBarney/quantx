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
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

#ifdef QX_HAS_LAPACKE
// Use std::complex as the LAPACKE complex type (bit-compatible, MSVC-friendly).
#define LAPACK_COMPLEX_CUSTOM
#define lapack_complex_float  std::complex<float>
#define lapack_complex_double std::complex<double>
#include <lapacke.h>
// The engine parallelises over Sz blocks with its own OpenMP; a multi-threaded BLAS
// *inside* that loop oversubscribes and access-violates (and Phase 0 found it
// catastrophic for performance too). Pin OpenBLAS to one thread only for the duration
// of that loop, then RESTORE the host's setting -- so loading QuantX into a process
// that also uses OpenBLAS (numpy/scipy/...) does not silently single-thread it (Codex).
extern "C" { int openblas_get_num_threads(void); void openblas_set_num_threads(int); }
namespace {
// OpenBLAS's thread count is process-global (no thread-local setter), so concurrent
// guards would race -- one could restore the host's value while another's loop is
// still running, re-enabling the nested threading that crashes (Codex). Serialise the
// guarded regions with a mutex: it's uncontended in QuantX's normal single-driver use
// and correct if a host ever calls QuantX from multiple threads at once.
//
// INHERENT LIMITATION: because the setting is process-global, while a QuantX block-SVD
// loop runs the process OpenBLAS is momentarily 1 thread, so a *concurrent* external
// OpenBLAS call (numpy/scipy on another host thread, sharing this OpenBLAS) can be
// transiently single-threaded -- the mutex only serialises QuantX's own guards, it
// cannot fence non-QuantX BLAS users. No global knob can avoid this; restoring the
// host's value (above) handles the common sequential case (QuantX call, then numpy).
// When embedding QuantX-with-OpenBLAS in a multithreaded BLAS host, run the process
// with OPENBLAS_NUM_THREADS=1 (standard for OpenMP-parallel hosts) or build without
// QX_ENABLE_BLAS.
std::mutex g_openblas_mtx;
struct OpenBlasSingleThreadGuard {
    std::lock_guard<std::mutex> lk;          // held for the whole block-SVD loop
    int prev;
    OpenBlasSingleThreadGuard()
        : lk(g_openblas_mtx), prev(openblas_get_num_threads()) { openblas_set_num_threads(1); }
    ~OpenBlasSingleThreadGuard() { openblas_set_num_threads(prev); }  // restore under lock, then unlock
};
}
#endif

#ifdef QX_USE_CUDA
// GPU (cuSOLVER/cuBLAS) Gram-trick SVD for the large blocks (Phase A,
// docs/gpu_acceleration_plan.md). Plain-type interface; impl in gpu_svd.cu.
#include "gpu_svd.hpp"
#endif

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
static long long g_tebd_maxblk = 0, g_tebd_nsvd = 0, g_tebd_nlapacke = 0, g_tebd_ngpu = 0;
// fine SVD split (QX_SVD_PROFILE=1, run single-thread): Gram-GEMM / eigensolve / back.
static double g_svd_gemm = 0, g_svd_eig = 0, g_svd_back = 0;
static bool g_svd_prof = false;

#ifdef QX_USE_CUDA
// Minimum min(rows,cols) for a block to be routed to the GPU SVD (Phase A). Below
// this, PCIe transfer + kernel-launch overhead dominate the O(k^3) compute, so keep
// it on the CPU. The measured CPU(Eigen-Gram)/GPU(cuSOLVER) crossover is ~d=128
// (qx_gpu_svd_bench: d=64 0.28x, d=128 1.15x, d=256 5.6x, d=1024 45x); against an
// OpenBLAS CPU baseline the crossover is a bit higher, so 128 is a safe default.
// Override with env QX_GPU_MIN_BLK.
static int gpu_min_blk(int d = 2) {
    static int env = []() {
        const char* e = std::getenv("QX_GPU_MIN_BLK");
        if (e) { int x = std::atoi(e); if (x > 0) return x; }
        return 0;   // 0 = unset
    }();
    if (env > 0) return env;   // explicit override applies to all local dims
    // d>2 fragments chi across more charge sectors -> more, smaller per-sector blocks.
    // A low crossover then floods the single GPU with many MEDIUM blocks that serialize
    // on the one device (measured: slower than CPU). Routing only the FEW dominant
    // blocks per bond to the GPU instead lets it overlap the CPU swarm. Per-model
    // measured crossovers: d=2 (spin-1/2, big blocks) 128; d=3 (spin-1, maxblk ~315 at
    // chi=512) 256; d>=4 (Hubbard packed two-U(1), even finer -- maxblk ~207 at chi=512)
    // 192, a reproducible 1.16x over CPU-only (interleaved min-of-3; 128 floods, 208
    // routes nothing).
    return d >= 4 ? 192 : d == 3 ? 256 : 128;
}
// Minimum GEMM work (n_row * n_keep_mid * n_col) for a theta block to be built on
// the GPU (Phase B residency). Both spin-sector GEMMs of the block run on-device
// sharing one Ai1 upload, so transfer is amortised -- but only above this work do
// the O(.^3) GEMMs beat the H2D/D2H. Override with env QX_GPU_THETA_MINWORK.
static long long gpu_theta_minwork() {
    static long long v = []() {
        const char* e = std::getenv("QX_GPU_THETA_MINWORK");
        long long d = 1LL << 21;            // ~2.1e6 ≈ 128^3
        if (e) { long long x = std::atoll(e); if (x > 0) d = x; }
        return d;
    }();
    return v;
}
// Whether the GPU SVD should use the FP32 mixed-precision path (Phase C). It is a
// measured but modest win (~1.1-1.4x on the SVD phase; the Jacobi eigensolver is
// iteration-bound, so it captures only part of the FP32 throughput advantage) and
// it reduces precision, so it is OFF by default. `allow` is passed per-call by the
// caller and is true only where single precision is tolerable (real-time TEBD and
// the dynamics built on it); it is never a global, so an energy-exact path can never
// be cross-contaminated by a concurrent/earlier dynamics call. Env QX_GPU_FP32:
//   unset/0 -> FP64 always (default);  1 -> honour `allow`;  >=2 -> FP32 always (testing).
static bool gpu_fp32_mode(bool allow) {
    static int env = []() {
        const char* e = std::getenv("QX_GPU_FP32");
        return e ? std::atoi(e) : 0;
    }();
    if (env <= 0) return false;     // default: FP64
    if (env >= 2) return true;      // force FP32 everywhere (testing)
    return allow;                   // env==1: FP32 only where the caller permits it
}
#endif

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

// The physical (Sz) charges of the local Hilbert space, in 2*Sz integer units.
// Spin-1/2 is the default ({+1,-1}); the kernels below take a phys_charge array so
// the same code runs spin-1 ({+2,0,-2}), Hubbard, etc.
static const std::vector<int> SPIN_HALF_CHARGES = {+1, -1};

// One physical-state column/row group in a folded block tensor. A block tensor
// block_A[q_L] = Mat(dL, sum_s width_s) lays the local index s into contiguous
// column groups: group s maps the left bond q_L to right bond q = q_L + qphys[s],
// width = bs.dim(q), offset = cumulative width of the earlier states. theta rows use
// the same scheme with the opposite sign (left bond q = q_mid - qphys[s]).
struct PhysGroup { int s, q, off, width; };
static std::vector<PhysGroup> phys_groups(const BondSpace& bs, int base, int sign,
                                          const std::vector<int>& qphys) {
    std::vector<PhysGroup> g(qphys.size());
    int off = 0;
    for (int s = 0; s < (int)qphys.size(); ++s) {
        int q = base + sign * qphys[s];
        int w = bs.dim(q);
        g[s] = {s, q, off, w};
        off += w;
    }
    return g;
}
// Total folded width of all the groups (= number of columns/rows of the block).
static int phys_total(const std::vector<PhysGroup>& g) {
    int t = 0; for (auto& e : g) t += e.width; return t;
}

// Distribution of total charge over k sites, each drawing one local state from
// qphys (equal weight). DP convolution; path counts are relative weights only.
static std::map<int,double> charge_dist(int k, const std::vector<int>& qphys) {
    std::map<int,double> d; d[0] = 1.0;
    for (int i = 0; i < k; ++i) {
        std::map<int,double> nd;
        for (auto& [q, w] : d)
            for (int qs : qphys) nd[q + qs] += w;
        d.swap(nd);
    }
    return d;
}

// Build the BondSpace at bond i for a length-n chain, total Sz=0, max_bond chi, for
// a local Hilbert space with charges qphys. Distributes chi states proportionally to
// the number of paths through each reachable sector (left block at charge q, right
// block at -q). For spin-1/2 ({+1,-1}) the path weights are exactly the binomials,
// so this reproduces the original Heisenberg bond space bit-identically.
static BondSpace make_bond_space(int bond_i, int n, int max_bond,
                                 const std::vector<int>& qphys,
                                 int total_charge = 0) {
    int sites_L = bond_i + 1;
    int sites_R = n - sites_L;

    std::map<int,double> distL = charge_dist(sites_L, qphys);
    std::map<int,double> distR = charge_dist(sites_R, qphys);

    std::vector<std::pair<int,double>> sectors;   // (bond charge q, weight)
    for (auto& [q, wL] : distL) {
        auto it = distR.find(total_charge - q);   // right block carries total_charge - q
        if (it == distR.end()) continue;
        double w = wL * it->second;
        if (w > 0) sectors.emplace_back(q, w);
    }
    if (sectors.empty()) {
        BondSpace bs;
        bs.qns = {0}; bs.dims = {max_bond}; bs.offsets = {0};
        bs.total_dim = max_bond;
        return bs;
    }
    // The at-least-1-per-sector allocation below would blow the bond past max_bond when
    // there are MORE reachable sectors than max_bond -- routine for a packed two-U(1)
    // charge, whose mid-chain sector count is O(n^2) (N x Sz). Keep only the max_bond
    // highest-weight sectors (deterministic: weight desc, charge asc as tiebreak). The
    // spin chains at the tested sizes have far fewer sectors than chi, so nothing
    // changes there; and this is only the INITIAL bond-space guess -- the SVD-truncated
    // spaces take over from the first two-site update.
    if ((int)sectors.size() > max_bond) {
        std::sort(sectors.begin(), sectors.end(),
                  [](auto& a, auto& b){ return a.second != b.second ? a.second > b.second
                                                                    : a.first < b.first; });
        sectors.resize(max_bond);
    }
    std::sort(sectors.begin(), sectors.end());

    double total_w = 0;
    for (auto& [q, w] : sectors) total_w += w;

    // Proportional allocation, at least 1 per sector
    std::vector<std::pair<int,int>> raw;
    int allocated = 0;
    for (auto& [q, w] : sectors) {
        int d = std::max(1, (int)std::round((double)w / total_w * max_bond));
        raw.emplace_back(q, d);
        allocated += d;
    }
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

// Build the BondSpace at bond i for a length-n chain, total Sz=0, max_bond chi.
// Distributes chi states proportionally to the number of paths through each sector.
static BondSpace make_heisenberg_bond_space(int bond_i, int n, int max_bond) {
    // Spin-1/2 special case of the general builder (path weights = binomials).
    return make_bond_space(bond_i, n, max_bond, SPIN_HALF_CHARGES);
}

// -- Block-sparse environment -------------------------------------------------
//
// The block-sparse path is U(1)/Sz-symmetric: every MPO bond index carries a
// definite Sz charge (the bra-vs-ket QN shift the env block at that bond holds).
// For the Heisenberg MPO this is the fixed {0,+2,-2,0,0}; for a general
// Sz-conserving MPO it is *derived* from the operator structure (see
// derive_mpo_sym below), which is what lets the same fast engine run arbitrary
// Sz-conserving Hamiltonians, not just XXZ. The reference Heisenberg charges:
//   b=0 (I)    DeltaQN=0    b=1 (S+/X) DeltaQN=+2   b=2 (S-/X) DeltaQN=-2
//   b=3 (Sz/Z) DeltaQN=0    b=4 (full) DeltaQN=0
// Reference Heisenberg charges (what derive_mpo_sym reproduces for build_xxz_mpo);
// kept for documentation. The kernels below take a per-MPO `delta` array
// (= derive_mpo_sym(mpo).delta) so the same engine runs any Sz-conserving MPO,
// not just the 5-wide Heisenberg one. Comments mentioning HB_DELTA[b] describe
// the role that delta[b] now plays generally.
[[maybe_unused]] static constexpr int HB_CHI_W = 5;
[[maybe_unused]] static constexpr int HB_DELTA[5] = {0, +2, -2, 0, 0};
// MPOSym + derive_mpo_sym (the general per-bond Sz-charge derivation) live in
// dmrg_internal.hpp so the validation test can reach them.

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
    // blocks[b][q_ket] = Mat(dim_bra, dim_ket),  bra_qn = q_ket + delta[b]
    // (delta = the MPO's per-bond Sz charges). Sized to the MPO bond dimension by
    // init_block_env; boundary envs use only blocks[0] (the charge-0 vacuum),
    // hence the default size 1.
    std::vector<std::map<int,Mat>> blocks{ std::map<int,Mat>{} };
    BondSpace bs;  // ket bond space
};

// Pre-allocate zero blocks for a BlockEnv over an MPO with bond charges `delta`.
static void init_block_env(BlockEnv& env, const BondSpace& bs,
                           const std::vector<int>& delta) {
    env.bs = bs;
    env.blocks.assign(delta.size(), {});
    for (int b = 0; b < (int)delta.size(); ++b) {
        for (int qi = 0; qi < bs.n_sectors(); ++qi) {
            int q_ket = bs.qns[qi], d_ket = bs.dims[qi];
            int d_bra = bs.dim(q_ket + delta[b]);
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
                                  const MPOSite& W,
                                  const std::vector<int>& delta) {
    BlockEnv L_new;
    init_block_env(L_new, bs_R, delta);

    const std::vector<int>& qphys = W.phys_charge;

    for (int qi = 0; qi < bs_L.n_sectors(); ++qi) {
        int q_ket_L = bs_L.qns[qi];

        // Ket A tensor (row sector = q_ket_L); columns grouped by physical state.
        auto it_A_ket = block_A.find(q_ket_L);
        if (it_A_ket == block_A.end()) continue;
        const Mat& Ablk_ket = it_A_ket->second;
        auto ket_cols = phys_groups(bs_R, q_ket_L, +1, qphys);

        for (int b_in = 0; b_in < W.chi_l; ++b_in) {
            auto it_L = L.blocks[b_in].find(q_ket_L);
            if (it_L == L.blocks[b_in].end()) continue;
            const Mat& Lblk = it_L->second;  // (dL_bra, dL_ket), bra_qn = q_ket_L + delta[b_in]
            if (Lblk.size() == 0) continue;

            // Bra A tensor: bra left QN = q_ket_L + delta[b_in]
            int q_bra_L = q_ket_L + delta[b_in];
            auto it_A_bra = block_A.find(q_bra_L);
            if (it_A_bra == block_A.end()) continue;
            const Mat& Ablk_bra = it_A_bra->second;
            auto bra_cols = phys_groups(bs_R, q_bra_L, +1, qphys);

            for (int b_out = 0; b_out < W.chi_r; ++b_out) {
                const auto& Wab = W.W(b_in, b_out);

                for (int s = 0; s < (int)qphys.size(); ++s) {   // bra spin (conjugate A)
                    if (bra_cols[s].width == 0) continue;
                    int sz = qphys[s];
                    Eigen::Ref<const Mat> A_s_bra =
                        Ablk_bra.middleCols(bra_cols[s].off, bra_cols[s].width);

                    for (int sp = 0; sp < (int)qphys.size(); ++sp) {  // ket spin
                        S w = mscal(Wab(s, sp));
                        if (std::abs(w) < 1e-15) continue;
                        if (ket_cols[sp].width == 0) continue;
                        int spz = qphys[sp];

                        int q_ket_R = q_ket_L + spz;  // new ket right bond QN
                        // QN consistency: delta[b_out] = delta[b_in] + sz - spz
                        int req_delta = delta[b_in] + sz - spz;
                        if (delta[b_out] != req_delta) continue;

                        auto jt = L_new.blocks[b_out].find(q_ket_R);
                        if (jt == L_new.blocks[b_out].end()) continue;

                        Eigen::Ref<const Mat> A_sp_ket =
                            Ablk_ket.middleCols(ket_cols[sp].off, ket_cols[sp].width);

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
                                   const MPOSite& W,
                                   const std::vector<int>& delta) {
    BlockEnv R_new;
    init_block_env(R_new, bs_L, delta);

    const std::vector<int>& qphys = W.phys_charge;

    for (int qi = 0; qi < bs_L.n_sectors(); ++qi) {
        int q_ket_L = bs_L.qns[qi];

        // Ket A tensor (row sector = q_ket_L); columns grouped by physical state.
        auto it_A_ket = block_A.find(q_ket_L);
        if (it_A_ket == block_A.end()) continue;
        const Mat& Ablk_ket = it_A_ket->second;
        auto ket_cols = phys_groups(bs_R, q_ket_L, +1, qphys);

        for (int b_out = 0; b_out < W.chi_l; ++b_out) {
            auto jt = R_new.blocks[b_out].find(q_ket_L);
            if (jt == R_new.blocks[b_out].end()) continue;

            // Bra A tensor: bra left QN = q_ket_L + delta[b_out]
            int q_bra_L = q_ket_L + delta[b_out];
            auto it_A_bra = block_A.find(q_bra_L);
            if (it_A_bra == block_A.end()) continue;
            const Mat& Ablk_bra = it_A_bra->second;
            auto bra_cols = phys_groups(bs_R, q_bra_L, +1, qphys);

            for (int b_in = 0; b_in < W.chi_r; ++b_in) {
                const auto& Wab = W.W(b_out, b_in);

                for (int s = 0; s < (int)qphys.size(); ++s) {   // ket spin
                    if (ket_cols[s].width == 0) continue;
                    int sz = qphys[s];
                    int dR_s = ket_cols[s].width;
                    int q_ket_R = q_ket_L + sz;
                    Eigen::Ref<const Mat> A_s_ket =
                        Ablk_ket.middleCols(ket_cols[s].off, ket_cols[s].width);

                    for (int sp = 0; sp < (int)qphys.size(); ++sp) {  // bra spin
                        S w = mscal(Wab(sp, s));
                        if (std::abs(w) < 1e-15) continue;
                        if (bra_cols[sp].width == 0) continue;
                        int spz = qphys[sp];
                        int dR_sp = bra_cols[sp].width;

                        // QN consistency: delta[b_out] = delta[b_in] + sz - spz
                        // (here s is the KET spin, sp is the BRA spin).
                        int req_delta = delta[b_in] + sz - spz;
                        if (delta[b_out] != req_delta) continue;

                        auto it_R = R.blocks[b_in].find(q_ket_R);
                        if (it_R == R.blocks[b_in].end()) continue;
                        const Mat& Rblk = it_R->second;  // (dR_sp_bra, dR_s_ket)
                        if (Rblk.rows() != dR_sp || Rblk.cols() != dR_s) continue;

                        Eigen::Ref<const Mat> A_sp_bra =
                            Ablk_bra.middleCols(bra_cols[sp].off, bra_cols[sp].width);

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

// Combined two-site MPO weights, fixed per bond (sized to the MPO bond dims):
//   at(bL,bR, s1b*d+s2b, s1k*d+s2k) = sum_bm Wi(bL,bm)(s1b,s1k) * Wj(bm,bR)(s2b,s2k)
// The two-site bra/ket index packs the d-dimensional local states as s1*d+s2, so the
// per-(bL,bR) cell is a flat d2 x d2 (d2=d*d) block (was a fixed 4x4 for spin-1/2).
struct WComb {
    int nR = 0;                  // Wj.chi_r (column stride over bR)
    int d = 2, d2 = 4;           // local dim and d*d
    std::vector<S> w;            // [((bL*nR+bR)*d2 + brow)*d2 + bcol]
    S& at(int bL, int bR, int brow, int bcol) {
        return w[((size_t)(bL*nR + bR)*d2 + brow)*d2 + bcol];
    }
    const S& at(int bL, int bR, int brow, int bcol) const {
        return w[((size_t)(bL*nR + bR)*d2 + brow)*d2 + bcol];
    }
};

static WComb build_wcomb(const MPOSite& Wi, const MPOSite& Wj) {
    WComb wc;
    wc.nR = Wj.chi_r;
    wc.d  = Wi.d;
    wc.d2 = wc.d * wc.d;
    const int d = wc.d;
    wc.w.assign((size_t)Wi.chi_l * Wj.chi_r * wc.d2 * wc.d2, S(0));
    // Skip zero W-blocks BEFORE the d^4 inner loops: a lower-triangular MPO of width
    // chi has O(chi) nonzero blocks out of chi^2, so enumerating all (bL,bm,bR)
    // triples blind costs O(chi^3 d^4) -- 34.8s/4-sweeps on the width-34 cylinder MPO
    // (the real content of the "theta" profile bucket, which form_theta itself was
    // acquitted of). Precompute each site's nonzero (row, col) lists and iterate only
    // the sparse product structure: O(nnz_i * nnz_j / chi) triples.
    std::vector<std::vector<int>> nz_i(Wi.chi_r), nz_j(Wi.chi_r);
    for (int bL = 0; bL < Wi.chi_l; ++bL)
        for (int bm = 0; bm < Wi.chi_r; ++bm)
            if (Wi.W(bL, bm).squaredNorm() > 0) nz_i[bm].push_back(bL);
    for (int bm = 0; bm < Wj.chi_l; ++bm)
        for (int bR = 0; bR < Wj.chi_r; ++bR)
            if (Wj.W(bm, bR).squaredNorm() > 0) nz_j[bm].push_back(bR);
    for (int bm = 0; bm < Wi.chi_r; ++bm)
        for (int bL : nz_i[bm])
            for (int bR : nz_j[bm]) {
                const auto& wi = Wi.W(bL, bm);
                const auto& wj = Wj.W(bm, bR);
                for (int s1b = 0; s1b < d; ++s1b)
                    for (int s2b = 0; s2b < d; ++s2b)
                        for (int s1k = 0; s1k < d; ++s1k)
                            for (int s2k = 0; s2k < d; ++s2k)
                                wc.at(bL, bR, s1b*d+s2b, s1k*d+s2k) +=
                                    mscal(wi(s1b,s1k) * wj(s2b,s2k));
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
    const BondSpace& bs_L, const BondSpace& bs_R2, const std::vector<int>& delta)
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

    const std::vector<int>& qphys = Wi.phys_charge;
    const int d = Wi.d;

    // Memoise the bra-side phys_groups: they depend ONLY on q_mid_bra, but the loop
    // below reaches the same q_mid_bra from many (bL, bR, s1b) combinations, and each
    // phys_groups call heap-allocates a vector -- on a wide MPO (34 channels, W=4
    // cylinder) that was millions of allocations per bond update, the dominant cost
    // of the plan build.
    struct BraGroups { std::vector<PhysGroup> rows, cols; int n_row = 0, n_col = 0; };
    std::map<int, BraGroups> bra_memo;
    auto bra_of = [&](int q_mid_bra) -> const BraGroups& {
        auto it = bra_memo.find(q_mid_bra);
        if (it == bra_memo.end()) {
            BraGroups bg;
            bg.rows = phys_groups(bs_L,  q_mid_bra, -1, qphys);
            bg.cols = phys_groups(bs_R2, q_mid_bra, +1, qphys);
            bg.n_row = phys_total(bg.rows);
            bg.n_col = phys_total(bg.cols);
            it = bra_memo.emplace(q_mid_bra, std::move(bg)).first;
        }
        return it->second;
    };

    for (int q_mid_ket : ket_keys) {
        int ket_idx = (int)plan.ket_sectors.size();
        plan.ket_sectors.push_back(q_mid_ket);
        // theta[q_mid_ket] rows grouped by left-site state s1k, cols by right-site s2k.
        auto ket_rows = phys_groups(bs_L,  q_mid_ket, -1, qphys);
        auto ket_cols = phys_groups(bs_R2, q_mid_ket, +1, qphys);

        for (int s1k = 0; s1k < d; ++s1k) {
            int dL_k = ket_rows[s1k].width;
            if (dL_k == 0) continue;
            int row_k = ket_rows[s1k].off;
            int q_L_ket = ket_rows[s1k].q;   // = q_mid_ket - qphys[s1k]

            for (int s2k = 0; s2k < d; ++s2k) {
                int dR_k = ket_cols[s2k].width;
                if (dR_k == 0) continue;
                int col_k = ket_cols[s2k].off;
                int q_R_ket = ket_cols[s2k].q;  // = q_mid_ket + qphys[s2k]

                for (int bR = 0; bR < Wj.chi_r; ++bR) {
                    auto it_R = R.blocks[bR].find(q_R_ket);
                    if (it_R == R.blocks[bR].end()) continue;
                    const Mat& Rblk = it_R->second;
                    if (Rblk.cols() != dR_k) continue;
                    int dR_bra = (int)Rblk.rows();
                    int q_R_bra = q_R_ket + delta[bR];

                    typename HeffPlan::T1Group g;
                    g.ket_idx = ket_idx;
                    g.row_k = row_k; g.col_k = col_k; g.dL_k = dL_k; g.dR_k = dR_k;
                    g.Rblk = &Rblk;
                    plan.flops += 2.0 * dL_k * dR_k * dR_bra;  // T1 = theta_sub * R^T
                    plan.max_t1 = std::max(plan.max_t1, dL_k * dR_bra);

                    for (int bL = 0; bL < Wi.chi_l; ++bL) {
                        int q_L_bra = q_L_ket + delta[bL];
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

                        for (int s1b = 0; s1b < d; ++s1b) {
                            int q_mid_bra = q_L_bra + qphys[s1b];
                            const BraGroups& bg = bra_of(q_mid_bra);
                            if (bg.n_row == 0 || bg.n_col == 0) continue;
                            int dL_b = bg.rows[s1b].width;
                            if (dL_b != dL_bra) continue;
                            int row_b = bg.rows[s1b].off;

                            for (int s2b = 0; s2b < d; ++s2b) {
                                S w = Wc.at(bL, bR, s1b*d+s2b, s1k*d+s2k);
                                if (std::abs(w) < 1e-15) continue;
                                int dR_b = bg.cols[s2b].width;
                                if (dR_b == 0) continue;
                                if (q_mid_bra + qphys[s2b] != q_R_bra) continue;
                                if (dR_bra != dR_b) continue;
                                int col_b = bg.cols[s2b].off;

                                int idx = get_out_idx(q_mid_bra, bg.n_row, bg.n_col);
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
    const BondSpace& bs_L, const BondSpace& bs_R,
    const std::vector<int>& delta)
{
    ++g_block_matvecs;
    std::map<int,Mat> out;
    const std::vector<int>& qphys = W.phys_charge;

    for (auto& [q_L_ket, Mblk] : M) {
        int dL_ket = bs_L.dim(q_L_ket);
        auto ket_cols = phys_groups(bs_R, q_L_ket, +1, qphys);

        for (int s_ket = 0; s_ket < (int)qphys.size(); ++s_ket) {
            if (ket_cols[s_ket].width == 0) continue;
            int dR_s_ket = ket_cols[s_ket].width;
            int col_k = ket_cols[s_ket].off;
            int q_R_ket = ket_cols[s_ket].q;   // = q_L_ket + qphys[s_ket]
            Eigen::Ref<const Mat> M_sub = Mblk.block(0, col_k, dL_ket, dR_s_ket);

            for (int b_in = 0; b_in < W.chi_l; ++b_in) {
                int q_L_bra = q_L_ket + delta[b_in];
                int dL_bra  = bs_L.dim(q_L_bra);
                if (dL_bra == 0) continue;
                auto it_L = L.blocks[b_in].find(q_L_ket);
                if (it_L == L.blocks[b_in].end()) continue;
                const Mat& Lblk = it_L->second;   // (dL_bra, dL_ket)
                if (Lblk.rows() != dL_bra || Lblk.cols() != dL_ket) continue;

                Mat LM = Lblk * M_sub;            // (dL_bra, dR_s_ket)
                auto bra_cols = phys_groups(bs_R, q_L_bra, +1, qphys);

                for (int b_out = 0; b_out < W.chi_r; ++b_out) {
                    auto it_R = R.blocks[b_out].find(q_R_ket);
                    if (it_R == R.blocks[b_out].end()) continue;
                    const Mat& Rblk = it_R->second;   // (dR_bra, dR_s_ket)
                    if (Rblk.cols() != dR_s_ket) continue;
                    int dR_bra = (int)Rblk.rows();
                    int q_R_bra = q_R_ket + delta[b_out];

                    // ALL bra states s_bra with q_L_bra + qphys[s_bra] == q_R_bra
                    // contribute. qphys is NOT injective for Hubbard (|up> and |dn> both
                    // carry particle number +1), so several local states can share the
                    // same charge -- taking only the first silently drops half the
                    // single-site H_eff (which broke TDVP2 and single-site DMRG at d=4).
                    // For the spin models (injective qphys) this loop matches exactly one
                    // s_bra, so their behaviour is unchanged. T = L M R is s_bra-independent.
                    Mat T = LM * Rblk.transpose();   // (dL_bra, dR_bra)
                    auto it_out = out.find(q_L_bra);
                    if (it_out == out.end())
                        it_out = out.emplace(q_L_bra, Mat::Zero(dL_bra, phys_total(bra_cols))).first;
                    for (int s_bra = 0; s_bra < (int)qphys.size(); ++s_bra) {
                        if (q_L_bra + qphys[s_bra] != q_R_bra) continue;
                        if (bra_cols[s_bra].width != dR_bra) continue;
                        S w = mscal(W.W(b_in, b_out)(s_bra, s_ket));
                        if (std::abs(w) < 1e-15) continue;
                        it_out->second.block(0, bra_cols[s_bra].off, dL_bra, dR_bra).noalias() += w * T;
                    }
                }
            }
        }
    }
    // Ensure all input sectors are present (Galerkin: H_eff maps the space to itself).
    for (auto& [q, blk] : M)
        if (!out.count(q)) out[q] = Mat::Zero(blk.rows(), blk.cols());
    return out;
}

// Precomputed one-site H_eff plan -- the single-site analogue of HeffPlan. The
// environments, MPO and bond spaces are fixed across a Lanczos solve, but
// apply_heff_1site redoes ALL the QN bookkeeping (map lookups, phys_groups heap
// allocations, dimension checks) on every matvec; on the width-4 cylinder's
// fragmented sectors that bookkeeping dominated the single-site refinement phase
// (the hybrid's 12-sweep run spent 170s of 225s there). Build once per site,
// replay per matvec. The GEMM/write order matches apply_heff_1site exactly, so
// results are bit-identical; apply_heff_1site itself stays for the one-shot
// callers (TDVP2's bt_evolve_site).
struct Heff1Plan {
    std::vector<int> out_q, ket_q;
    std::vector<std::pair<int,int>> out_shape;      // (dL_bra, total bra cols)
    struct Write  { S w; int out_idx, col_b, dR_b; };
    struct RGroup { const Mat* Rblk; std::vector<Write> writes; };
    struct LGroup {
        int ket_idx, col_k, dR_k;                   // M_sub = ket.block(0, col_k, dL_ket, dR_k)
        const Mat* Lblk;
        std::vector<RGroup> rgroups;
    };
    std::vector<LGroup> groups;
};

static Heff1Plan build_heff1_plan(
    const BlockEnv& L, const MPOSite& W, const BlockEnv& R,
    const std::vector<int>& ket_keys,
    const BondSpace& bs_L, const BondSpace& bs_R, const std::vector<int>& delta)
{
    Heff1Plan plan;
    const std::vector<int>& qphys = W.phys_charge;
    std::map<int,int> out_index;
    std::map<int, std::vector<PhysGroup>> cols_memo;   // q_L -> phys_groups(bs_R, q_L, +1)
    auto cols_of = [&](int qL) -> const std::vector<PhysGroup>& {
        auto it = cols_memo.find(qL);
        if (it == cols_memo.end())
            it = cols_memo.emplace(qL, phys_groups(bs_R, qL, +1, qphys)).first;
        return it->second;
    };
    auto get_out = [&](int q_L_bra, int dL_bra) {
        auto it = out_index.find(q_L_bra);
        if (it != out_index.end()) return it->second;
        int idx = (int)plan.out_q.size();
        out_index[q_L_bra] = idx;
        plan.out_q.push_back(q_L_bra);
        plan.out_shape.push_back({dL_bra, phys_total(cols_of(q_L_bra))});
        return idx;
    };
    for (int ki = 0; ki < (int)ket_keys.size(); ++ki) {
        int q_L_ket = ket_keys[ki];
        plan.ket_q.push_back(q_L_ket);
        int dL_ket = bs_L.dim(q_L_ket);
        if (dL_ket == 0) continue;
        const auto& ket_cols = cols_of(q_L_ket);
        for (int s_ket = 0; s_ket < (int)qphys.size(); ++s_ket) {
            if (ket_cols[s_ket].width == 0) continue;
            int dR_s_ket = ket_cols[s_ket].width;
            int col_k = ket_cols[s_ket].off;
            int q_R_ket = ket_cols[s_ket].q;
            for (int b_in = 0; b_in < W.chi_l; ++b_in) {
                int q_L_bra = q_L_ket + delta[b_in];
                int dL_bra  = bs_L.dim(q_L_bra);
                if (dL_bra == 0) continue;
                auto it_L = L.blocks[b_in].find(q_L_ket);
                if (it_L == L.blocks[b_in].end()) continue;
                const Mat& Lblk = it_L->second;
                if (Lblk.rows() != dL_bra || Lblk.cols() != dL_ket) continue;

                typename Heff1Plan::LGroup lg;
                lg.ket_idx = ki; lg.col_k = col_k; lg.dR_k = dR_s_ket; lg.Lblk = &Lblk;
                const auto& bra_cols = cols_of(q_L_bra);
                for (int b_out = 0; b_out < W.chi_r; ++b_out) {
                    auto it_R = R.blocks[b_out].find(q_R_ket);
                    if (it_R == R.blocks[b_out].end()) continue;
                    const Mat& Rblk = it_R->second;
                    if (Rblk.cols() != dR_s_ket) continue;
                    int dR_bra = (int)Rblk.rows();
                    int q_R_bra = q_R_ket + delta[b_out];
                    typename Heff1Plan::RGroup rg; rg.Rblk = &Rblk;
                    for (int s_bra = 0; s_bra < (int)qphys.size(); ++s_bra) {
                        if (q_L_bra + qphys[s_bra] != q_R_bra) continue;
                        if (bra_cols[s_bra].width != dR_bra) continue;
                        S w = mscal(W.W(b_in, b_out)(s_bra, s_ket));
                        if (std::abs(w) < 1e-15) continue;
                        rg.writes.push_back({w, get_out(q_L_bra, dL_bra),
                                             bra_cols[s_bra].off, dR_bra});
                    }
                    if (!rg.writes.empty()) lg.rgroups.push_back(std::move(rg));
                }
                if (!lg.rgroups.empty()) plan.groups.push_back(std::move(lg));
            }
        }
    }
    return plan;
}

static std::map<int,Mat> apply_heff1_plan(const Heff1Plan& plan, const std::map<int,Mat>& M) {
    ++g_block_matvecs;
    std::vector<const Mat*> kets(plan.ket_q.size(), nullptr);
    for (int i = 0; i < (int)plan.ket_q.size(); ++i) {
        auto it = M.find(plan.ket_q[i]);
        if (it != M.end()) kets[i] = &it->second;
    }
    std::vector<Mat> out(plan.out_q.size());
    for (int i = 0; i < (int)plan.out_q.size(); ++i)
        out[i] = Mat::Zero(plan.out_shape[i].first, plan.out_shape[i].second);
    for (const auto& lg : plan.groups) {
        const Mat* ket = kets[lg.ket_idx];
        if (!ket) continue;
        Mat LM = (*lg.Lblk) * ket->block(0, lg.col_k, lg.Lblk->cols(), lg.dR_k);
        for (const auto& rg : lg.rgroups) {
            Mat T = LM * rg.Rblk->transpose();
            for (const auto& wr : rg.writes)
                out[wr.out_idx].block(0, wr.col_b, T.rows(), wr.dR_b).noalias() += wr.w * T;
        }
    }
    std::map<int,Mat> res;
    for (int i = 0; i < (int)plan.out_q.size(); ++i)
        res[plan.out_q[i]] = std::move(out[i]);
    for (auto& [q, blk] : M)
        if (!res.count(q)) res[q] = Mat::Zero(blk.rows(), blk.cols());
    return res;
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

// DMRG3S-style subspace expansion (Hubig/McCulloch/Schollwoeck): enrich the one-site
// centre M with the mixing block P = L * W * M before the truncating split,
//   P^{s}_{a,(b',beta)} = sum_{b,a',s'} L^b_{a,a'} W^{b b'}(s,s') M^{s'}_{a',beta},
// whose composite columns (b', beta) carry charge q_R_ket + delta[b'] -- so the SVD
// gets to KEEP directions (and whole charge sectors) the current bond does not have.
// This is the standard cure for the truncation-limited crawl the width-4 cylinder
// exposed (TeNPy's mixer analogue): plain single-site can never grow bond content.
// Returns the widened centre M_ext and its column bond space bs_ext (original bs_mid
// columns first, then the alpha-scaled P columns per (b', q_R_ket) pair). alpha sets
// the enrichment weight relative to ||M|| (P is Frobenius-normalised first).
static std::pair<std::map<int,Mat>, BondSpace> expand_M_LR(
    const std::map<int,Mat>& M, const BlockEnv& L, const MPOSite& W,
    const BondSpace& bs_L, const BondSpace& bs_mid,
    const std::vector<int>& delta, const std::vector<int>& qphys, double alpha)
{
    const int d = (int)qphys.size();
    // 1) enumerate the expansion column groups: (b', q_R_ket) -> charge q_ext, width
    //    dR(q_R_ket); assign each pair a column offset within its q_ext extension.
    struct Ext { int bprime, q_R_ket, off, width; };
    std::map<int, std::vector<Ext>> ext_groups;          // q_ext -> pairs
    std::map<int, int> ext_width;                        // q_ext -> total extension width
    for (int qi = 0; qi < bs_mid.n_sectors(); ++qi) {
        int q_R = bs_mid.qns[qi], dR = bs_mid.dims[qi];
        if (dR == 0) continue;
        for (int bp = 0; bp < W.chi_r; ++bp) {
            int q_ext = q_R + delta[bp];
            int& wsum = ext_width[q_ext];
            ext_groups[q_ext].push_back({bp, q_R, wsum, dR});
            wsum += dR;
        }
    }
    // 2) the widened bond space: original sectors first, extensions appended.
    std::vector<std::pair<int,int>> secs;
    std::set<int> all_q;
    for (int qi = 0; qi < bs_mid.n_sectors(); ++qi) all_q.insert(bs_mid.qns[qi]);
    for (auto& [q, wsum] : ext_width) all_q.insert(q);
    for (int q : all_q) {
        int wtot = bs_mid.dim(q) + (ext_width.count(q) ? ext_width[q] : 0);
        if (wtot > 0) secs.push_back({q, wtot});
    }
    BondSpace bs_ext = BondSpace::from_pairs(secs);
    // 3) build P into the extension columns. Loop the nonzero structure only.
    std::map<int,Mat> P;                                 // q_L_bra -> (dL_bra, ext cols layout)
    double Mnorm2 = 0; for (auto& [q, blk] : M) Mnorm2 += blk.squaredNorm();
    for (auto& [q_L_ket, Mblk] : M) {
        auto mcols = phys_groups(bs_mid, q_L_ket, +1, qphys);
        for (int b = 0; b < W.chi_l; ++b) {
            auto it_L = L.blocks[b].find(q_L_ket);
            if (it_L == L.blocks[b].end()) continue;
            const Mat& Lblk = it_L->second;              // (dL_bra, dL_ket)
            int q_L_bra = q_L_ket + delta[b];
            if (bs_L.dim(q_L_bra) != Lblk.rows() || Lblk.cols() != Mblk.rows()) continue;
            for (int sp = 0; sp < d; ++sp) {
                if (mcols[sp].width == 0) continue;
                int q_R_ket = mcols[sp].q;
                Mat LM = Lblk * Mblk.middleCols(mcols[sp].off, mcols[sp].width);
                for (int bp = 0; bp < W.chi_r; ++bp) {
                    for (int s = 0; s < d; ++s) {
                        S w = mscal(W.W(b, bp)(s, sp));
                        if (std::abs(w) < 1e-15) continue;
                        // destination: P[q_L_bra], row group s, extension column group
                        // (bp, q_R_ket) inside charge q_ext = q_R_ket + delta[bp].
                        int q_ext = q_R_ket + delta[bp];
                        auto pgrows = phys_groups(bs_ext, q_L_bra, +1, qphys);
                        if (pgrows[s].q != q_ext) continue;   // charge bookkeeping guard
                        auto& Pblk = P[q_L_bra];
                        if (Pblk.size() == 0)
                            Pblk = Mat::Zero(Lblk.rows(), phys_total(pgrows));
                        // find this (bp, q_R_ket) pair's offset within the extension
                        int pair_off = -1;
                        for (auto& e : ext_groups[q_ext])
                            if (e.bprime == bp && e.q_R_ket == q_R_ket) { pair_off = e.off; break; }
                        if (pair_off < 0) continue;
                        int col0 = pgrows[s].off + bs_mid.dim(q_ext) + pair_off;
                        Pblk.block(0, col0, Lblk.rows(), mcols[sp].width).noalias() += w * LM;
                    }
                }
            }
        }
    }
    // 4) Frobenius-normalise P to alpha * ||M||, then assemble M_ext = [M | alpha P].
    double Pnorm2 = 0; for (auto& [q, blk] : P) Pnorm2 += blk.squaredNorm();
    double scale = (Pnorm2 > 1e-300) ? alpha * std::sqrt(Mnorm2 / Pnorm2) : 0.0;
    std::map<int,Mat> Mext;
    std::set<int> qLs;
    for (auto& [q, _] : M) qLs.insert(q);
    for (auto& [q, _] : P) qLs.insert(q);
    for (int q_L : qLs) {
        int dL = bs_L.dim(q_L);
        if (dL == 0) continue;
        auto cols_ext = phys_groups(bs_ext, q_L, +1, qphys);
        Mat blk = Mat::Zero(dL, phys_total(cols_ext));
        auto itM = M.find(q_L);
        if (itM != M.end()) {
            auto cols_old = phys_groups(bs_mid, q_L, +1, qphys);
            for (int s = 0; s < d; ++s) {
                if (cols_old[s].width == 0) continue;
                blk.block(0, cols_ext[s].off, dL, cols_old[s].width) =
                    itM->second.middleCols(cols_old[s].off, cols_old[s].width);
            }
        }
        auto itP = P.find(q_L);
        if (itP != P.end() && scale > 0 && itP->second.rows() == dL)
            blk += scale * itP->second;   // P already laid out on the bs_ext columns
        Mext[q_L] = blk;
    }
    return {std::move(Mext), std::move(bs_ext)};
}

static SiteMove singlesite_split_LR(const std::map<int,Mat>& M,
                                    const BondSpace& bs_L, const BondSpace& bs_mid,
                                    const std::vector<int>& qphys = SPIN_HALF_CHARGES,
                                    int max_bond = -1, double cutoff = 0.0) {
    SiteMove mv;
    std::map<int,Mat> Umap;     // q_R -> U (rows x keep)
    std::vector<std::pair<int,int>> new_sectors;

    // Every right-bond sector q_R reachable as q_L + qphys[s] from some present M[q_L].
    std::set<int> qRs;
    for (auto& [q_L, _] : M)
        for (int qs : qphys) qRs.insert(q_L + qs);

    // Phase 1: per-sector SVDs (kept whole; truncation decided globally below).
    struct SecSVD { int q_R; Mat U, Vt; RVec sv; };
    std::vector<SecSVD> svds;
    for (int q_R : qRs) {
        int dRmid = bs_mid.dim(q_R);
        if (dRmid == 0) continue;
        // Stack the rows that map to q_R, grouped by physical state s: the s-rows come
        // from M[q_L=q_R-qphys[s]], the column block of M[q_L] mapping to q_R.
        auto src = phys_groups(bs_L, q_R, -1, qphys);
        int rows = phys_total(src);
        if (rows == 0) continue;
        Mat X = Mat::Zero(rows, dRmid);
        for (int s = 0; s < (int)qphys.size(); ++s) {
            if (src[s].width == 0) continue;
            int q_L = src[s].q;                 // = q_R - qphys[s]
            auto it = M.find(q_L);
            if (it == M.end()) continue;
            auto mcols = phys_groups(bs_mid, q_L, +1, qphys);  // M[q_L] column groups
            X.middleRows(src[s].off, src[s].width) =
                it->second.middleCols(mcols[s].off, dRmid);
        }
        Eigen::BDCSVD<Mat> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
        svds.push_back({q_R, svd.matrixU(), Mat(svd.matrixV().adjoint()),
                        RVec(svd.singularValues())});
    }
    // Phase 2: keep counts. Default (max_bond <= 0): keep everything -- the historical
    // lossless split. With a cap (the subspace-expansion path widens the columns far
    // past chi): global top-max_bond across sectors + relative cutoff, exactly the
    // block_svd_truncate policy.
    std::map<int,int> keep_q;
    if (max_bond > 0) {
        struct SVE { double sv; int si; };
        std::vector<SVE> all;
        double tot2 = 0;
        for (int si = 0; si < (int)svds.size(); ++si)
            for (int k = 0; k < svds[si].sv.size(); ++k) {
                all.push_back({svds[si].sv[k], si});
                tot2 += svds[si].sv[k] * svds[si].sv[k];
            }
        std::sort(all.begin(), all.end(), [](auto& a, auto& b){ return a.sv > b.sv; });
        int n_keep = (int)all.size();
        if (tot2 > 0 && cutoff > 0) {
            double disc = 0;
            while (n_keep > 1) {
                double svl = all[n_keep-1].sv;
                if ((disc + svl*svl) / tot2 > cutoff) break;
                disc += svl*svl; --n_keep;
            }
        }
        n_keep = std::min(n_keep, max_bond);
        for (int k = 0; k < n_keep; ++k) keep_q[svds[all[k].si].q_R]++;
    } else {
        for (auto& s : svds) keep_q[s.q_R] = (int)s.sv.size();
    }
    for (auto& sec : svds) {
        int keep = keep_q.count(sec.q_R) ? keep_q[sec.q_R] : 0;
        if (keep == 0) continue;
        Umap[sec.q_R] = sec.U.leftCols(keep);
        Mat Vt = sec.Vt.topRows(keep);                                  // keep x dRmid
        for (int k = 0; k < keep; ++k) Vt.row(k) *= sec.sv[k];
        mv.C[sec.q_R] = Vt;                                             // S V^dag
        new_sectors.emplace_back(sec.q_R, keep);
    }
    std::sort(new_sectors.begin(), new_sectors.end());
    mv.new_mid = BondSpace::from_pairs(new_sectors);

    // Reassemble A[q_L]: column group s comes from U[q_R=q_L+qphys[s]], the rows of
    // that U belonging to physical state s (which map back to q_L).
    for (int qi = 0; qi < bs_L.n_sectors(); ++qi) {
        int q_L = bs_L.qns[qi], dL = bs_L.dims[qi];
        auto acols = phys_groups(mv.new_mid, q_L, +1, qphys);
        Mat A = Mat::Zero(dL, phys_total(acols));
        for (int s = 0; s < (int)qphys.size(); ++s) {
            int keep = acols[s].width;
            if (keep == 0) continue;
            int q_R = acols[s].q;               // = q_L + qphys[s]
            auto it = Umap.find(q_R);
            if (it == Umap.end()) continue;
            auto urows = phys_groups(bs_L, q_R, -1, qphys);
            A.middleCols(acols[s].off, keep) = it->second.middleRows(urows[s].off, dL);
        }
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

// Default Lanczos iteration cap. The local two-site eigensolve does not need high
// accuracy inside a DMRG sweep -- the sweeps recover it -- and in the TRUNCATION-
// limited regime (width-4 cylinders) the tol-based stop burns ~44 matvecs per bond
// on accuracy the following SVD immediately throws away. 25 matches common practice
// (TeNPy's N_max=20); the chain/ED tests converge in far fewer anyway. QX_LANCZOS_MAX.
static int lanczos_max_iter() {
    static int v = []() {
        const char* e = std::getenv("QX_LANCZOS_MAX");
        return (e && std::atoi(e) > 0) ? std::atoi(e) : 25;
    }();
    return v;
}
static std::pair<double, std::map<int,Mat>> block_lanczos(
    const std::function<std::map<int,Mat>(const std::map<int,Mat>&)>& matvec,
    const std::map<int,Mat>& init,
    int max_iter = -1, double tol = 1e-10)
{
    if (max_iter < 0) max_iter = lanczos_max_iter();
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
    // U_grp[s][q_mid] = the left-site-state-s rows of U at q_mid (was U_up/U_dn for
    // spin-1/2). Group s spans the rows of theta[q_mid] coming from left bond
    // q_mid - qphys[s]; size = qphys.size() (2 for spin-1/2, 3 for spin-1, ...).
    std::vector<std::map<int,Mat>> U_grp;
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
        using _clk = std::chrono::steady_clock; _clk::time_point _t;
        if (g_svd_prof) _t = _clk::now();
        Mat G = M * M.adjoint();                                  // m x m
        if (g_svd_prof) { g_svd_gemm += std::chrono::duration<double>(_clk::now()-_t).count(); _t=_clk::now(); }
        Eigen::SelfAdjointEigenSolver<Mat> es(G);
        if (g_svd_prof) { g_svd_eig += std::chrono::duration<double>(_clk::now()-_t).count(); _t=_clk::now(); }
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
        if (g_svd_prof) g_svd_back += std::chrono::duration<double>(_clk::now()-_t).count();
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

#ifdef QX_HAS_LAPACKE
// SVD via the Gram trick with the (smaller) Hermitian eigensolve done by LAPACK
// xheevd/xsyevd (OpenBLAS, divide-and-conquer) instead of Eigen's built-in solver
// which lags at large m. With EIGEN_USE_BLAS the surrounding GEMMs (M Mᴴ, Mᴴ U)
// also run on OpenBLAS. Same interface/semantics as thin_svd_gram; used for LARGE
// blocks (small blocks keep the pure-Eigen Gram path, lower fixed overhead).
// Measured faster than a direct LAPACK xgesdd here (the Gram trick is fewer FLOPs).
static bool gram_eigh_lapacke(Mat& G /*destroyed -> eigenvectors*/, int d, Eigen::VectorXd& w) {
    lapack_int info;
    if constexpr (std::is_same_v<S, Cx>)
        info = LAPACKE_zheevd(LAPACK_COL_MAJOR, 'V', 'U', (lapack_int)d,
                              reinterpret_cast<lapack_complex_double*>(G.data()), (lapack_int)d, w.data());
    else
        info = LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'V', 'U', (lapack_int)d,
                              reinterpret_cast<double*>(G.data()), (lapack_int)d, w.data());
    return info == 0;                      // G: eigenvectors (cols), w: eigenvalues (ascending)
}
static bool thin_svd_lapacke(const Mat& M, Mat& U, Mat& V, RVec& sv) {
    const int m = (int)M.rows(), nc = (int)M.cols();
    if (m <= nc) {
        Mat G = M * M.adjoint();                          // m×m Hermitian
        Eigen::VectorXd w(m);
        if (!gram_eigh_lapacke(G, m, w)) return false;
        U.resize(m, m); sv.resize(m);
        for (int j = 0; j < m; ++j) {
            double lam = w(m-1-j); sv(j) = std::sqrt(lam > 0 ? lam : 0.0);
            U.col(j) = G.col(m-1-j);                      // descending order
        }
        Mat MhU = M.adjoint() * U; V.resize(nc, m);
        double smax = (sv.size() && sv(0) > 0) ? sv(0) : 1.0;
        for (int j = 0; j < m; ++j)
            if (sv(j) > 1e-14 * smax) V.col(j) = MhU.col(j) / sv(j); else V.col(j).setZero();
    } else {
        Mat G = M.adjoint() * M;                          // n×n Hermitian
        Eigen::VectorXd w(nc);
        if (!gram_eigh_lapacke(G, nc, w)) return false;
        V.resize(nc, nc); sv.resize(nc);
        for (int j = 0; j < nc; ++j) {
            double lam = w(nc-1-j); sv(j) = std::sqrt(lam > 0 ? lam : 0.0);
            V.col(j) = G.col(nc-1-j);
        }
        Mat MV = M * V; U.resize(m, nc);
        double smax = (sv.size() && sv(0) > 0) ? sv(0) : 1.0;
        for (int j = 0; j < nc; ++j)
            if (sv(j) > 1e-14 * smax) U.col(j) = MV.col(j) / sv(j); else U.col(j).setZero();
    }
    return true;
}
// Direct thin SVD via LAPACK xgesdd (divide-and-conquer). A SINGLE optimized call
// (no Gram matrix, no separate eigensolve + back-transform), and numerically better
// (no condition-number squaring). gesdd is a *different* routine from the xsyevd/
// xheevd that returns wrong results on this box's OpenBLAS, so it is safe to use.
// Returns M = U diag(sv) V^dag with U (m x k), V (n x k), k=min(m,n).
static bool thin_svd_gesdd(const Mat& M, Mat& U, Mat& V, RVec& sv) {
    const int m = (int)M.rows(), nc = (int)M.cols(), k = std::min(m, nc);
    if (k == 0) { U.resize(m,0); V.resize(nc,0); sv.resize(0); return true; }
    Mat A = M;                                   // xgesdd overwrites A (col-major)
    U.resize(m, k); sv.resize(k);
    Mat VT(k, nc);
    lapack_int info;
    if constexpr (std::is_same_v<S, Cx>)
        info = LAPACKE_zgesdd(LAPACK_COL_MAJOR, 'S', (lapack_int)m, (lapack_int)nc,
                  reinterpret_cast<lapack_complex_double*>(A.data()), (lapack_int)m, sv.data(),
                  reinterpret_cast<lapack_complex_double*>(U.data()), (lapack_int)m,
                  reinterpret_cast<lapack_complex_double*>(VT.data()), (lapack_int)k);
    else
        info = LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'S', (lapack_int)m, (lapack_int)nc,
                  reinterpret_cast<double*>(A.data()), (lapack_int)m, sv.data(),
                  reinterpret_cast<double*>(U.data()), (lapack_int)m,
                  reinterpret_cast<double*>(VT.data()), (lapack_int)k);
    if (info != 0) return false;
    V = VT.adjoint();                            // V (nc x k); V.col(j) = conj(VT.row(j))
    return true;
}
// Per-block min-dim at/above which a block uses the direct LAPACK xgesdd instead of
// the Gram-eigensolve. gesdd's optimized divide-and-conquer beats Eigen's per-block
// eigensolve for LARGE blocks (measured ~1.2x at chi=512); for small/rectangular
// blocks the Gram trick (eigensolve of the smaller dimension, lower fixed overhead)
// still wins, so keep them on Gram. Tunable with QX_GESDD_MIN.
static int gesdd_min() {
    static int v = []() {
        const char* e = std::getenv("QX_GESDD_MIN");
        return (e && std::atoi(e) > 0) ? std::atoi(e) : 160;   // tuned crossover
    }();
    return v;
}
// Minimum block min-dim to route the Gram eigensolve to LAPACK xsyevd/xheevd.
// DEFAULT: off. The conda-forge OpenBLAS on this box returns WRONG results from
// LAPACKE_dsyevd/zheevd at some block sizes -- reproduced as a corrupted real DMRG
// ground state (E0 garbage at chi=512, while chi=256/1024 happened to be fine), which
// the ED test (n=6, tiny blocks <96) never exercises. EIGEN_USE_BLAS still routes the
// Gram trick's dominant GEMMs (M Mᴴ, Mᴴ U) to OpenBLAS -- the bulk of the speedup and
// numerically correct. Opt back in with QX_LAPACKE_MIN=<n> only on an OpenBLAS whose
// LAPACKE eigensolver is verified correct.
static int lapacke_min() {
    static int v = []() {
        const char* e = std::getenv("QX_LAPACKE_MIN");
        return (e && std::atoi(e) > 0) ? std::atoi(e) : 1000000000;   // default: disabled
    }();
    return v;
}
#endif

#ifdef QX_USE_CUDA
// GPU Gram-trick SVD for large blocks (Phase A). Same interface/semantics as
// thin_svd_gram. Only the complex path is on the GPU (the TEBD dynamics suite the
// plan targets); the real (DMRG) scalar falls back to the CPU, which is already a
// strong win there. Returns false on any device error -> caller uses the CPU path.
static bool thin_svd_gram_gpu(const Mat& M, Mat& U, Mat& V, RVec& sv, bool allow_fp32) {
    if constexpr (!std::is_same_v<S, Cx>) {
        return false;
    } else {
        const int m = (int)M.rows(), nc = (int)M.cols();
        const int k = std::min(m, nc);
        U.resize(m, k); V.resize(nc, k); sv.resize(k);
        auto* Md = reinterpret_cast<const std::complex<double>*>(M.data());
        auto* Ud = reinterpret_cast<std::complex<double>*>(U.data());
        auto* Vd = reinterpret_cast<std::complex<double>*>(V.data());
        // Phase C: try FP32 first when permitted; if its Jacobi didn't converge
        // (returns false before writing outputs) fall back to FP64 *on the GPU*
        // (still fast) rather than dropping a large block to the CPU.
        if (gpu_fp32_mode(allow_fp32) && qx_gpu::thin_svd_gram_gpu_c(Md, m, nc, Ud, Vd, sv.data()))
            return true;
        return qx_gpu::thin_svd_gram_gpu_z(Md, m, nc, Ud, Vd, sv.data());  // FP64 GPU
    }
}
#endif

static BlockSVDResult block_svd_truncate(
    const std::map<int,Mat>& theta,
    const BondSpace& bs_L,
    const BondSpace& bs_R2,
    int max_bond, double cutoff,
    bool allow_fp32 = false,   // Phase C: FP32 GPU SVD permitted (real-time TEBD only)
    bool stable = false,       // direct (BDCSVD) SVD instead of the Gram trick
    const std::vector<int>& qphys = SPIN_HALF_CHARGES)
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
        // Skip empty AND all-zero blocks: a zero Schmidt block carries no weight, so
        // its SVD (sv=0, truncated away) is pure overhead. bt_expand_theta zero-fills
        // many reachable-but-empty sectors each step; the gate fills only some, leaving
        // a swarm of zero blocks. (They re-expand next step, so no state is lost.)
        if (blk.size() != 0 && blk.squaredNorm() > 1e-24) { svq.push_back(q_mid); svblk.push_back(&blk); }
    // Longest-processing-time-first: SVD the largest blocks first so the dynamic
    // schedule does not strand the dominant central-Sz eigensolve at the very end
    // while the other threads idle (the spin-1 swarm has one big block + many tiny).
    if (svq.size() > 2) {
        std::vector<int> ord(svq.size());
        for (int i = 0; i < (int)ord.size(); ++i) ord[i] = i;
        auto wk = [&](int a){ long long r = svblk[a]->rows(), c = svblk[a]->cols();
                              return std::min(r,c) * std::min(r,c) * std::max(r,c); };  // ~ SVD flops
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b){ return wk(a) > wk(b); });
        std::vector<int> q2(svq.size()); std::vector<const Mat*> b2(svq.size());
        for (int i = 0; i < (int)ord.size(); ++i) { q2[i] = svq[ord[i]]; b2[i] = svblk[ord[i]]; }
        svq.swap(q2); svblk.swap(b2);
    }
    g_tebd_nsvd += (long long)svq.size();
    for (auto* p : svblk) g_tebd_maxblk = std::max(g_tebd_maxblk, (long long)std::min(p->rows(), p->cols()));
    std::vector<SVDBlk> sb(svq.size());
#ifdef QX_HAS_LAPACKE
    // Pin OpenBLAS to 1 thread across this parallel loop (nested BLAS threads crash),
    // restoring the host's setting when the guard leaves scope at function return.
    OpenBlasSingleThreadGuard _blas_guard;
#endif
    // `stable`: a direct SVD (Eigen BDCSVD) instead of the Gram trick. The Gram
    // trick (eigensolve of M*M^dag) squares the condition number, so at a degenerate
    // Schmidt spectrum (e.g. a DMRG ground state sitting exactly at a level crossing
    // like the J1-J2 Majumdar-Ghosh point) it amplifies the ~1e-15 FP differences of
    // multi-threaded GEMMs into non-orthonormal Schmidt vectors -> a non-canonical
    // MPS -> a (wrong) energy below the true ground state. The direct SVD is
    // condition-number stable and keeps the state canonical. Used by the DMRG
    // ground-state path; the real-time dynamics keep the fast Gram/GPU route (no
    // exact degeneracy there, and speed matters).
    // Run serial when QX_SVD_PROFILE is on: the fine gram-split accumulators
    // (g_svd_gemm/eig/back, plain doubles in thin_svd_gram) are +='d in this loop, so
    // parallel profiling would both race them and make the split meaningless.
    #pragma omp parallel for schedule(dynamic) if(svq.size() > 1 && !g_svd_prof)
    for (int i = 0; i < (int)svq.size(); ++i) {
        if (stable) {
            // The stability requirement is a DIRECT SVD (no Gram condition-squaring).
            // LAPACK xgesdd is exactly that -- same guarantee as Eigen's BDCSVD, but
            // several times faster on the mid/large blocks that dominate the GS sweep
            // (BDCSVD's constant factor was why the Hubbard chi=256 GS lost to TeNPy).
            // Fall back to BDCSVD for small blocks (lower fixed overhead) and on any
            // gesdd failure. The Majumdar-Ghosh degeneracy regression (deterministic,
            // variational) guards this path.
#ifdef QX_HAS_LAPACKE
            if (std::min(svblk[i]->rows(), svblk[i]->cols()) >= gesdd_min() &&
                thin_svd_gesdd(*svblk[i], sb[i].U, sb[i].V, sb[i].s))
                { g_tebd_nlapacke++; continue; }
#endif
            Eigen::BDCSVD<Mat> bsvd(*svblk[i], Eigen::ComputeThinU | Eigen::ComputeThinV);
            sb[i].U = bsvd.matrixU(); sb[i].V = bsvd.matrixV(); sb[i].s = bsvd.singularValues();
            continue;
        }
#ifdef QX_USE_CUDA
        // Large blocks -> GPU (cuSOLVER) Gram-eigensolve; the dominant central-Sz
        // block is the O(chi^3) Amdahl bottleneck this is meant to crush. Tiny tail
        // blocks stay on the CPU (transfer-bound). Falls back to CPU on any error.
        if (std::min(svblk[i]->rows(), svblk[i]->cols()) >= gpu_min_blk((int)qphys.size()) &&
            thin_svd_gram_gpu(*svblk[i], sb[i].U, sb[i].V, sb[i].s, allow_fp32))
            { _Pragma("omp atomic") g_tebd_ngpu++; continue; }
#endif
#ifdef QX_HAS_LAPACKE
        // Direct LAPACK xgesdd for LARGE blocks: its divide-and-conquer beats Eigen's
        // per-block Gram eigensolve (the spin-1 large-chi bottleneck) and is condition-
        // stable. Small/rectangular blocks stay on the Gram trick (it eigensolves the
        // smaller dimension, with lower fixed overhead).
        if (std::min(svblk[i]->rows(), svblk[i]->cols()) >= gesdd_min() &&
            thin_svd_gesdd(*svblk[i], sb[i].U, sb[i].V, sb[i].s))
            { _Pragma("omp atomic") g_tebd_nlapacke++; continue; }
        // Gram eigensolve via LAPACK xsyevd/xheevd -- disabled by default (wrong
        // results on this box's OpenBLAS); only for blocks >= lapacke_min.
        if (std::min(svblk[i]->rows(), svblk[i]->cols()) >= lapacke_min() &&
            thin_svd_lapacke(*svblk[i], sb[i].U, sb[i].V, sb[i].s))
            { _Pragma("omp atomic") g_tebd_nlapacke++; continue; }
#endif
        thin_svd_gram(*svblk[i], sb[i].U, sb[i].V, sb[i].s);
    }
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
    // Guard the degenerate all-zero theta: the zero-block skip above can leave all_sv
    // empty (unreachable for a normalized MPS -- total norm ~1 guarantees a surviving
    // block -- but the skip removed that invariant), and the max(1,...) would otherwise
    // index all_sv[0] below. Clamp so the keep/trunc loops are no-ops on an empty spectrum.
    n_keep = std::min(n_keep, (int)all_sv.size());
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
    res.U_grp.assign(qphys.size(), {});
    res.trunc_err = 0;
    for (int k = n_keep; k < (int)all_sv.size(); ++k)
        res.trunc_err += all_sv[k].sv * all_sv[k].sv;

    // Build new bond space and extract U/Vt blocks. U's rows are grouped by the
    // left-site physical state s (rows from left bond q_mid - qphys[s]).
    std::vector<std::pair<int,int>> new_sectors;
    for (auto& [q_mid, svd] : svds) {
        int keep = keep_per_q.count(q_mid) ? keep_per_q[q_mid] : 0;
        if (keep == 0) continue;
        const Mat& U = svd->U;
        auto rows = phys_groups(bs_L, q_mid, -1, qphys);
        for (int s = 0; s < (int)qphys.size(); ++s)
            if (rows[s].width > 0)
                res.U_grp[s][q_mid] = U.middleRows(rows[s].off, rows[s].width).leftCols(keep);
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
    bool absorb_sv_into_Ai1,
    const std::vector<int>& qphys = SPIN_HALF_CHARGES)
{
    std::map<int,Mat> theta;
    const bool d2 = (qphys.size() == 2 && qphys[0] == +1 && qphys[1] == -1);

    // Find all q_mid values: q_mid = q_L + qphys[s] for some present left sector.
    std::set<int> q_mids;
    for (auto& [q_L, _] : block_Ai)
        for (int qs : qphys) { int q = q_L + qs; if (bs_mid.dim(q) > 0) q_mids.insert(q); }

    // The per-q_mid assemblies are independent, and on fragmented bonds (width-W
    // cylinders: hundreds of small sectors) this loop was 55% of the DMRG sweep and
    // essentially serial (many small GEMMs, no BLAS threading). Two passes: the GPU-
    // eligible blocks first (serial -- one device), then the CPU swarm OMP-parallel,
    // each thread building its own block (within-block arithmetic order unchanged ->
    // bit-identical results). CPU pure-assembly lambda used by both passes:
    auto build_cpu = [&](int q_mid) -> Mat {
        // theta[q_mid] rows grouped by left-site state, cols by right-site state.
        auto rows = phys_groups(bs_L,  q_mid, -1, qphys);
        auto cols = phys_groups(bs_R2, q_mid, +1, qphys);
        int n_row = phys_total(rows);
        int n_col = phys_total(cols);
        if (n_row == 0 || n_col == 0) return Mat();

        // Get A[i+1][q_mid] with optional sv absorption
        auto it1 = block_Ai1.find(q_mid);
        if (it1 == block_Ai1.end()) return Mat();

        Mat th = Mat::Zero(n_row, n_col);
        Mat Ai1 = it1->second;  // (n_keep_mid, n_col)
        if (absorb_sv_into_Ai1) {
            auto it_sv = sv_mid.find(q_mid);
            if (it_sv != sv_mid.end()) {
                const RVec& sv = it_sv->second;
                int sv_len = std::min((int)sv.size(), (int)Ai1.rows());
                for (int k = 0; k < sv_len; ++k) Ai1.row(k) *= sv[k];
            }
        }

        // For each left-site state s1: A[i][q_L=q_mid-qphys[s1]] contributes rows to th.
        int n_keep_mid = bs_mid.dim(q_mid);
        for (int s1 = 0; s1 < (int)qphys.size(); ++s1) {
            int q_L = rows[s1].q;       // = q_mid - qphys[s1]
            int dL  = rows[s1].width;
            if (dL == 0) continue;
            auto it0 = block_Ai.find(q_L);
            if (it0 == block_Ai.end()) continue;
            const Mat& Ai = it0->second;
            auto mcols = phys_groups(bs_mid, q_L, +1, qphys);  // A[i][q_L] column groups
            int col_off_Ai = mcols[s1].off;     // group mapping q_L -> q_mid is state s1
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
            int row_off = rows[s1].off;
            if (row_off + dL > n_row) continue;
            th.block(row_off, 0, dL, n_col).noalias() += Ai_s1 * Ai1;
        }
        return th;
    };

    // Pass 1 (serial): GPU-eligible blocks -- large, sv-free (the TEBD/dynamics path;
    // one device, so no parallelism here). Everything else goes to the CPU swarm.
    std::vector<int> pending; pending.reserve(q_mids.size());
    for (int q_mid : q_mids) {
        bool done = false;
#ifdef QX_USE_CUDA
        if constexpr (std::is_same_v<S, Cx>) {
            auto rows = phys_groups(bs_L,  q_mid, -1, qphys);
            auto cols = phys_groups(bs_R2, q_mid, +1, qphys);
            int n_row = phys_total(rows), n_col = phys_total(cols);
            auto it1 = block_Ai1.find(q_mid);
            const int n_keep_mid = bs_mid.dim(q_mid);
            if (n_row > 0 && n_col > 0 && it1 != block_Ai1.end() &&
                sv_mid.empty() && n_keep_mid > 0 &&
                (int)it1->second.rows() == n_keep_mid && (int)it1->second.cols() == n_col &&
                (long long)n_row * n_keep_mid * n_col >= gpu_theta_minwork()) {
                const Mat& Ai1r = it1->second;
                if (d2) {
                    int dL_up = rows[0].width, dL_dn = rows[1].width;
                    const Cx* Aup = nullptr; const Cx* Adn = nullptr;
                    bool gpu_ok = true;
                    if (dL_up > 0) {                          // s1=up: q_L = q_mid-1, col_off 0
                        auto it0 = block_Ai.find(q_mid - 1);
                        if (it0 != block_Ai.end() && n_keep_mid <= (int)it0->second.cols()) {
                            if ((int)it0->second.rows() == dL_up)
                                Aup = reinterpret_cast<const Cx*>(it0->second.data());
                            else gpu_ok = false;
                        }
                    }
                    if (gpu_ok && dL_dn > 0) {                // s1=dn: q_L = q_mid+1
                        auto it0 = block_Ai.find(q_mid + 1);
                        int dR_up_mid = bs_mid.dim(q_mid + 2);
                        if (it0 != block_Ai.end() && dR_up_mid + n_keep_mid <= (int)it0->second.cols()) {
                            if ((int)it0->second.rows() == dL_dn)
                                Adn = reinterpret_cast<const Cx*>(it0->second.data())
                                      + (size_t)dR_up_mid * dL_dn;
                            else gpu_ok = false;
                        }
                    }
                    if (gpu_ok && (Aup || Adn)) {
                        Mat th(n_row, n_col);
                        if (qx_gpu::form_theta_block_z(Aup, dL_up, Adn, dL_dn,
                                reinterpret_cast<const Cx*>(Ai1r.data()), n_keep_mid, n_col,
                                reinterpret_cast<Cx*>(th.data()))) {
                            if (th.norm() > 1e-14) theta[q_mid] = std::move(th);
                            done = true;
                        }
                    }
                } else {   // d>2: one GEMM per physical-state row group on-device
                    std::vector<const Cx*> As(qphys.size(), nullptr);
                    std::vector<int> dLs(qphys.size(), 0);
                    bool gpu_ok = true;
                    for (int s1 = 0; s1 < (int)qphys.size(); ++s1) {
                        int q_L = rows[s1].q, dL = rows[s1].width;
                        dLs[s1] = dL;
                        if (dL == 0) continue;
                        auto it0 = block_Ai.find(q_L);
                        if (it0 == block_Ai.end()) continue;
                        auto mcols = phys_groups(bs_mid, q_L, +1, qphys);
                        int coff = mcols[s1].off;
                        if (coff + n_keep_mid > (int)it0->second.cols()
                            || (int)it0->second.rows() != dL) { gpu_ok = false; break; }
                        As[s1] = reinterpret_cast<const Cx*>(it0->second.data())
                                 + (size_t)coff * dL;
                    }
                    if (gpu_ok) {
                        Mat th(n_row, n_col);
                        if (qx_gpu::form_theta_block_z_multi(As.data(), dLs.data(),
                                (int)qphys.size(),
                                reinterpret_cast<const Cx*>(Ai1r.data()), n_keep_mid, n_col,
                                n_row, reinterpret_cast<Cx*>(th.data()))) {
                            if (th.norm() > 1e-14) theta[q_mid] = std::move(th);
                            done = true;
                        }
                    }
                }
            }
        }
#endif
        if (!done) pending.push_back(q_mid);
    }

    // Pass 2 (parallel): the CPU swarm. Each thread assembles its own independent
    // block (per-block arithmetic order unchanged -> bit-identical). OpenBLAS is
    // pinned to 1 thread across the region (nested BLAS threads crash).
    std::vector<Mat> built(pending.size());
    {
#ifdef QX_HAS_LAPACKE
        OpenBlasSingleThreadGuard _blas_guard;
#endif
        #pragma omp parallel for schedule(dynamic) if(pending.size() > 1)
        for (int i = 0; i < (int)pending.size(); ++i)
            built[i] = build_cpu(pending[i]);
    }
    for (int i = 0; i < (int)pending.size(); ++i)
        if (built[i].size() != 0 && built[i].norm() > 1e-14)
            theta[pending[i]] = std::move(built[i]);
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

// <2 S^z_i> (the Sz charge, = <sigma^z> for spin-1/2) for a center tensor M: each
// physical-state column group s carries charge qphys[s], weighted by its probability.
// obs_of_charge decodes: plain charges pass through (2Sz, or N for single-U(1) Hubbard);
// packed two-U(1) charges yield the particle number N -- so the Hubbard staggered
// observable is the same <n_i> under either symmetry choice.
static double center_sz(const std::map<int,Mat>& M, const BondSpace& bsR,
                        const std::vector<int>& qphys = SPIN_HALF_CHARGES) {
    double num = 0, z = 0;
    for (auto& [q_L, blk] : M) {
        auto cols = phys_groups(bsR, q_L, +1, qphys);
        for (int s = 0; s < (int)qphys.size(); ++s) {
            if (cols[s].width == 0) continue;
            double w = blk.middleCols(cols[s].off, cols[s].width).squaredNorm();
            num += obs_of_charge(qphys[s]) * w;   // qphys = 2*Sz (or packed (N,2Sz) -> N)
            z += w;
        }
    }
    return z > 0 ? num / z : 0.0;
}

// staggered magnetization on a COPY (center sweep L->R, non-destructive).
static double measure_stag(std::vector<std::map<int,Mat>> A,
                           std::vector<BondSpace> bond, int n,
                           const std::vector<int>& qphys = SPIN_HALF_CHARGES) {
    double m = 0;
    for (int i = 0; i < n; ++i) {
        m += ((i % 2 == 0) ? 1.0 : -1.0) * center_sz(A[i], bond[i+1], qphys);
        if (i < n - 1) {
            SiteMove mv = singlesite_split_LR(A[i], bond[i], bond[i+1], qphys);
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

// Two-component per-site profiles (<N_i>, <2Sz_i>) on a COPY -- the same centre
// walk as measure_stag, decoding BOTH components of the packed two-U(1) charge
// (obs2_of_charge). The stripe-order observable pair for doped Hubbard lattices.
static void measure_profiles(std::vector<std::map<int,Mat>> A,
                             std::vector<BondSpace> bond, int n,
                             const std::vector<int>& qphys,
                             std::vector<double>& site_n,
                             std::vector<double>& site_2sz) {
    site_n.assign(n, 0.0); site_2sz.assign(n, 0.0);
    for (int i = 0; i < n; ++i) {
        double num_n = 0, num_s = 0, z = 0;
        for (auto& [q_L, blk] : A[i]) {
            auto cols = phys_groups(bond[i+1], q_L, +1, qphys);
            for (int s = 0; s < (int)qphys.size(); ++s) {
                if (cols[s].width == 0) continue;
                double w = blk.middleCols(cols[s].off, cols[s].width).squaredNorm();
                double Nv, Sv; obs2_of_charge(qphys[s], Nv, Sv);
                num_n += Nv * w; num_s += Sv * w; z += w;
            }
        }
        if (z > 0) { site_n[i] = num_n / z; site_2sz[i] = num_s / z; }
        if (i < n - 1) {
            SiteMove mv = singlesite_split_LR(A[i], bond[i], bond[i+1], qphys);
            A[i] = mv.A; bond[i+1] = mv.new_mid;
            std::map<int,Mat> next;
            for (auto& [q_R, Cblk] : mv.C) {
                auto it = A[i+1].find(q_R);
                if (it != A[i+1].end()) next[q_R] = Cblk * it->second;
            }
            A[i+1] = next;
        }
    }
}

// ----- reusable block-TDVP2 helpers (shared by run_tdvp2 and METTS) ----------
// Full block-MPS state: site tensors, bond spaces, and L/R environments.
struct BMPS {
    std::vector<std::map<int,Mat>> A;
    std::vector<BondSpace> bond;
    std::vector<BlockEnv> L, R;
    int n = 0;
};

static void bt_expand_theta(std::map<int,Mat>& theta, const BondSpace& bsL, const BondSpace& bsR2,
                            const std::vector<int>& qphys = SPIN_HALF_CHARGES) {
    for (int qi = 0; qi < bsL.n_sectors(); ++qi) {
        int q_L = bsL.qns[qi];
        for (int s = 0; s < (int)qphys.size(); ++s) {
            int q_mid = q_L + qphys[s];
            if (theta.count(q_mid)) continue;
            int nr = phys_total(phys_groups(bsL,  q_mid, -1, qphys));
            int nc = phys_total(phys_groups(bsR2, q_mid, +1, qphys));
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
                           const BlockEnv& Re, const BondSpace& bsL, const BondSpace& bsR, Cx tau,
                           const std::vector<int>& delta) {
    auto mv = [&](const Vec& vin) -> Vec {
        std::map<int,Mat> tb; vec_to_theta(vin, M, tb);
        auto raw = apply_heff_1site(Le, W, Re, tb, bsL, bsR, delta);
        std::map<int,Mat> outn;
        for (auto& [q, ref] : M) { auto it = raw.find(q);
            outn[q] = (it!=raw.end()&&it->second.rows()==ref.rows()&&it->second.cols()==ref.cols())
                      ? it->second : Mat::Zero(ref.rows(), ref.cols()); }
        return theta_to_vec(outn);
    };
    Vec ev = qx_tdvp::expmv(tau, mv, theta_to_vec(M));
    std::map<int,Mat> out; vec_to_theta(ev, M, out); M = out;
}
// The bt_* / cps_init / measurement helpers fold the physical index into d charge-
// shifted column groups (phys_groups), like the GS path. They default qphys to
// spin-1/2 so the TEBD/thermal callers (still d=2) are unchanged; the TDVP path
// passes the real qphys, so spin-1 (d=3) TDVP runs. (TEBD's gates are still d=2.)
static std::map<int,Mat> bt_A_from_U(const BlockSVDResult& svd, const BondSpace& bsL,
                                     const BondSpace& bsMid,
                                     const std::vector<int>& qphys = SPIN_HALF_CHARGES) {
    std::map<int,Mat> A;
    for (int qi = 0; qi < bsL.n_sectors(); ++qi) {
        int q_L = bsL.qns[qi], dL = bsL.dims[qi];
        auto cols = phys_groups(bsMid, q_L, +1, qphys);
        Mat blk = Mat::Zero(dL, phys_total(cols));
        for (int s = 0; s < (int)qphys.size(); ++s)
            if (cols[s].width > 0 && svd.U_grp[s].count(cols[s].q))
                blk.middleCols(cols[s].off, cols[s].width) = svd.U_grp[s].at(cols[s].q);
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
static std::map<int,Mat> bt_center_RL(const BlockSVDResult& svd, const BondSpace& bsL,
                                      const BondSpace& bsMid,
                                      const std::vector<int>& qphys = SPIN_HALF_CHARGES) {
    std::map<int,Mat> M;
    auto scaled = [&](Mat u, int q) {
        auto it = svd.sv.find(q);
        if (it != svd.sv.end()) { const RVec& s=it->second;
            int L2=std::min((int)s.size(),(int)u.cols()); for(int k=0;k<L2;++k) u.col(k)*=s[k]; }
        return u;
    };
    for (int qi = 0; qi < bsL.n_sectors(); ++qi) {
        int q_L = bsL.qns[qi], dL = bsL.dims[qi];
        auto cols = phys_groups(bsMid, q_L, +1, qphys);
        Mat blk = Mat::Zero(dL, phys_total(cols));
        for (int s = 0; s < (int)qphys.size(); ++s)
            if (cols[s].width > 0 && svd.U_grp[s].count(cols[s].q))
                blk.middleCols(cols[s].off, cols[s].width) = scaled(svd.U_grp[s].at(cols[s].q), cols[s].q);
        M[q_L] = blk;
    }
    return M;
}

// Build a block-MPS for an arbitrary Sz product state; sz[i] is the per-site Sz
// charge (= qphys[state], e.g. +1/-1 for spin-1/2, +2/-2 for a spin-1 Neel state).
// NOTE: the state is selected BY CHARGE (first qphys match), which is ambiguous for
// a non-injective qphys -- Hubbard charge 1 means |up> or |dn> and this would pick
// |up>. Fine for every current caller (the Hubbard CDW start uses only the unique
// charges {0,2}); a per-site STATE-INDEX init (like run()'s init_config) is the API
// to use if singly-occupied Hubbard product states are ever needed here.
static BMPS cps_init(const std::vector<MPOSite>& mpo, int n, const std::vector<int>& sz,
                     const std::vector<int>& qphys = SPIN_HALF_CHARGES) {
    BMPS s; s.n = n;
    std::vector<int> qb(n + 1); qb[0] = 0;
    for (int i = 0; i < n; ++i) qb[i+1] = qb[i] + sz[i];
    s.bond.resize(n + 1);
    for (int i = 0; i <= n; ++i) s.bond[i] = BondSpace::from_pairs({{qb[i], 1}});
    s.A.resize(n);
    for (int i = 0; i < n; ++i) {
        int q_L = qb[i];
        auto cols = phys_groups(s.bond[i+1], q_L, +1, qphys);
        Mat blk = Mat::Zero(1, phys_total(cols));
        for (int t = 0; t < (int)qphys.size(); ++t)   // the column group of state sz[i]
            if (qphys[t] == sz[i]) { blk(0, cols[t].off) = S(1); break; }
        s.A[i][q_L] = blk;
    }
    const std::vector<int> delta = derive_mpo_sym(mpo).delta;
    s.L.assign(n + 1, BlockEnv()); s.R.assign(n + 1, BlockEnv());
    { BlockEnv L0; L0.bs = s.bond[0]; L0.blocks[0][qb[0]] = Mat::Constant(1,1,S(1)); s.L[0] = L0; }
    { BlockEnv Rn; Rn.bs = s.bond[n]; Rn.blocks[0][qb[n]] = Mat::Constant(1,1,S(1)); s.R[n] = Rn; }
    for (int i = n - 1; i >= 0; --i)
        s.R[i] = block_update_right(s.R[i+1], s.A[i], s.bond[i], s.bond[i+1], mpo[i], delta);
    return s;
}

// One symmetric second-order block TDVP2 step (mutates s; centre returns to 0).
static void tdvp2_step_blocks(BMPS& s, const std::vector<MPOSite>& mpo,
                              double dt, bool imag, int chi_max, double cutoff) {
    const int n = s.n;
    const Cx fwd = imag ? Cx(-dt/2,0) : Cx(0,-dt/2);
    const Cx bwd = imag ? Cx(+dt/2,0) : Cx(0,+dt/2);
    const std::vector<int> delta = derive_mpo_sym(mpo).delta;
    const std::vector<int>& qphys = mpo[0].phys_charge;
    const std::map<int,RVec> no_sv;
    auto& A = s.A; auto& bond = s.bond; auto& L = s.L; auto& R = s.R;
    for (int i = 0; i < n - 1; ++i) {                      // left -> right
        auto theta = form_theta(A[i], A[i+1], no_sv, bond[i], bond[i+1], bond[i+2], true, qphys);
        bt_expand_theta(theta, bond[i], bond[i+2], qphys);
        WComb wc = build_wcomb(mpo[i], mpo[i+1]);
        std::vector<int> kk; for (auto& [q,_] : theta) kk.push_back(q);
        HeffPlan plan = build_heff_plan(L[i], mpo[i], mpo[i+1], R[i+2], wc, kk, bond[i], bond[i+2], delta);
        bt_evolve_theta(theta, plan, fwd);
        auto svd = block_svd_truncate(theta, bond[i], bond[i+2], chi_max, cutoff, false, false, qphys);
        bond[i+1] = svd.new_mid_bs;
        A[i] = bt_A_from_U(svd, bond[i], bond[i+1], qphys);
        L[i+1] = block_update_left(L[i], A[i], bond[i], bond[i+1], mpo[i], delta);
        auto M = bt_center_LR(svd);
        if (i < n - 2) bt_evolve_site(M, L[i+1], mpo[i+1], R[i+2], bond[i+1], bond[i+2], bwd, delta);
        A[i+1] = M;
    }
    for (int i = n - 2; i >= 0; --i) {                    // right -> left
        auto theta = form_theta(A[i], A[i+1], no_sv, bond[i], bond[i+1], bond[i+2], true, qphys);
        bt_expand_theta(theta, bond[i], bond[i+2], qphys);
        WComb wc = build_wcomb(mpo[i], mpo[i+1]);
        std::vector<int> kk; for (auto& [q,_] : theta) kk.push_back(q);
        HeffPlan plan = build_heff_plan(L[i], mpo[i], mpo[i+1], R[i+2], wc, kk, bond[i], bond[i+2], delta);
        bt_evolve_theta(theta, plan, fwd);
        auto svd = block_svd_truncate(theta, bond[i], bond[i+2], chi_max, cutoff, false, false, qphys);
        bond[i+1] = svd.new_mid_bs;
        A[i+1] = svd.Vt;
        R[i+1] = block_update_right(R[i+2], A[i+1], bond[i+1], bond[i+2], mpo[i+1], delta);
        auto M = bt_center_RL(svd, bond[i], bond[i+1], qphys);
        if (i > 0) bt_evolve_site(M, L[i], mpo[i], R[i+1], bond[i], bond[i+1], bwd, delta);
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

// ---- General (any local dimension) two-site gate -----------------------------
// The d=2 apply_gate_theta hand-codes the four spin-1/2 states. For d>2 we build the
// gate from the MPO and apply it Sz-block-wise (below). The bond Hamiltonian of a
// uniform lower-triangular nearest-neighbour MPO splits as
//   h_bond = sum_{b != start,done} W(start,b) (x) W(b,done)   (the pure two-site term)
//   h_site = W(start,done)                                    (the on-site term, if any)
// with start = chi-1 (the emit row) and done = 0 (the complete column). The b=done
// carry channel is h_site (x) I (the LEFT site's on-site term) and b=start is
// I (x) h_site (the RIGHT site's): summing them blindly counts every interior site's
// on-site term TWICE over the bond sweep (each interior site sits in two bonds). So
// the gate takes boundary weights (cL, cR): h2 = h_bond + cL h_site(x)I + cR I(x)h_site,
// 1/2 for interior sites, 1 at the chain ends. Models with no on-site term (spin
// Heisenberg/XXZ: W(start,done)=0) are weight-independent -- bit-identical to the old
// single-gate path. Hubbard's on-site U n_up n_dn is the first model that needs this.
// The gate is G = exp(tau * h2).
static Mat build_gate_from_mpo(const std::vector<MPOSite>& mpo, Cx tau,
                               double cL = 0.5, double cR = 0.5) {
    int bulk = -1, w = 0;
    for (int i = 0; i < (int)mpo.size(); ++i)
        if (mpo[i].chi_l == mpo[i].chi_r && mpo[i].chi_l >= w) { w = mpo[i].chi_l; bulk = i; }
    if (bulk < 0) throw std::runtime_error("build_gate_from_mpo: no square bulk MPO site");
    const MPOSite& W = mpo[bulk];
    const int d = W.d, d2 = d * d, chi = W.chi_r;
    const int start = chi - 1, done = 0;
    Eigen::MatrixXcd h2 = Eigen::MatrixXcd::Zero(d2, d2);
    for (int b = 0; b < chi; ++b) {
        if (b == start || b == done) continue;   // carry channels -> h_site, weighted below
        const Mat& Wl = W.W(start, b);   // d x d (left site)
        const Mat& Wr = W.W(b, done);    // d x d (right site)
        for (int s1 = 0; s1 < d; ++s1)
            for (int s2 = 0; s2 < d; ++s2)
                for (int s1p = 0; s1p < d; ++s1p)
                    for (int s2p = 0; s2p < d; ++s2p)
                        h2(s1*d + s2, s1p*d + s2p) += Wl(s1, s1p) * Wr(s2, s2p);
    }
    const Mat& hs = W.W(start, done);    // on-site term (zero for the spin models)
    for (int s1 = 0; s1 < d; ++s1)
        for (int s1p = 0; s1p < d; ++s1p)
            for (int s2 = 0; s2 < d; ++s2) {
                h2(s1*d + s2, s1p*d + s2) += cL * hs(s1, s1p);   // h_site (x) I
                h2(s2*d + s1, s2*d + s1p) += cR * hs(s1, s1p);   // I (x) h_site
            }
    h2 = 0.5 * (h2 + h2.adjoint().eval());   // symmetrize away FP non-Hermiticity
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(h2);
    Eigen::ArrayXcd ev = (tau * es.eigenvalues().array().cast<Cx>()).exp();
    Eigen::MatrixXcd G = es.eigenvectors() * ev.matrix().asDiagonal() * es.eigenvectors().adjoint();
    if constexpr (std::is_same_v<S, Cx>) return G;
    else                                 return G.real();   // (TEBD is Cx-only; dead for double)
}

// Apply a precomputed d2 x d2 Sz-conserving gate G (= exp(tau h_bond)) to theta. For
// a fixed (q_L, q_R), every two-site state (s1,s2) with qphys[s1]+qphys[s2] = q_R-q_L
// lives in the (dim q_L x dim q_R) sub-block of theta[q_L+qphys[s1]]; the gate mixes
// just those (one total-Sz sector). Generalizes apply_gate_theta to any local dim.
static void apply_gate_theta_general(std::map<int,Mat>& theta, const BondSpace& bsL,
                                     const BondSpace& bsR, const Mat& G, int d,
                                     const std::vector<int>& qphys) {
    // distinct two-site totals (qphys[s1]+qphys[s2]); for each q_L, q_R = q_L + tot.
    std::set<int> tots;
    for (int a : qphys) for (int b : qphys) tots.insert(a + b);
    for (int li = 0; li < bsL.n_sectors(); ++li) {
        int q_L = bsL.qns[li], dL = bsL.dims[li];
        for (int tot : tots) {
            int q_R = q_L + tot, dR = bsR.dim(q_R);
            if (dR == 0) continue;
            // Gather the present (s1,s2) sub-blocks (copies, so writes don't alias).
            std::vector<int> have;                  // packed index s1*d+s2
            std::vector<Mat> oldb(d * d);
            std::vector<std::tuple<int,int,int>> loc(d * d);  // (q_mid, row_off, col_off)
            for (int s1 = 0; s1 < d; ++s1)
                for (int s2 = 0; s2 < d; ++s2) {
                    if (qphys[s1] + qphys[s2] != tot) continue;
                    int q_mid = q_L + qphys[s1];
                    auto it = theta.find(q_mid);
                    if (it == theta.end()) continue;
                    auto rows = phys_groups(bsL, q_mid, -1, qphys);
                    auto cols = phys_groups(bsR, q_mid, +1, qphys);
                    if (rows[s1].width != dL || cols[s2].width != dR) continue;
                    int p = s1*d + s2;
                    have.push_back(p);
                    oldb[p] = it->second.block(rows[s1].off, cols[s2].off, dL, dR);
                    loc[p]  = {q_mid, rows[s1].off, cols[s2].off};
                }
            // new[p1] = sum over present p0 of G(p1,p0) * old[p0]
            for (int p1 : have) {
                Mat acc = Mat::Zero(dL, dR);
                for (int p0 : have) {
                    S g = G(p1, p0);
                    if (std::abs(g) > 1e-15) acc.noalias() += g * oldb[p0];
                }
                auto [q_mid, ro, co] = loc[p1];
                theta[q_mid].block(ro, co, dL, dR) = acc;
            }
        }
    }
}

// One symmetric second-order block TEBD step (L->R then R->L with half gates, so
// every bond is acted on with the full dt and the sweep is time-symmetric).
// No environments and no Krylov -- just a gate + SVD per bond, the reason TEBD is
// cheaper than TDVP for nearest-neighbour models.
static void tebd2_step_blocks(BMPS& s, double dt, bool imag, int chi_max,
                              double cutoff, double J = 1.0, double delta = 1.0,
                              bool allow_fp32 = false,
                              const std::vector<MPOSite>* mpo = nullptr) {
    const int n = s.n;
    const Cx tau = imag ? Cx(-dt/2, 0) : Cx(0, -dt/2);
    // d=2 keeps the exact hand-coded XXZ gate (bit-identical); d>2 builds the gate
    // from the MPO's 2-site bond Hamiltonian. qphys=spin-1/2 for d=2 (= the default
    // everywhere below, so d=2 is unchanged).
    const int d = (mpo && !mpo->empty()) ? (*mpo)[0].d : 2;
    const std::vector<int>& qphys = (d > 2) ? (*mpo)[0].phys_charge : SPIN_HALF_CHARGES;
    const Gate g = (d == 2) ? build_xxz_gate(J, delta, tau) : Gate{};
    // d>2: three gate variants weight the on-site term (h_site = W(start,done)) by the
    // sites' bond membership -- 1/2 interior, 1 at a chain end (bond 0 owns site 0
    // fully; bond n-2 owns site n-1 fully). Models without an on-site term (spin-1
    // Heisenberg) make all three identical -- bit-identical to the old single gate.
    Mat Gmid, Glft, Grgt;
    if (d > 2) {
        Gmid = build_gate_from_mpo(*mpo, tau, 0.5, 0.5);
        Glft = build_gate_from_mpo(*mpo, tau, 1.0, (n == 2) ? 1.0 : 0.5);
        Grgt = build_gate_from_mpo(*mpo, tau, 0.5, 1.0);
    }
    auto apply_gate = [&](std::map<int,Mat>& th, const BondSpace& bL, const BondSpace& bR, int i) {
        if (d == 2) { apply_gate_theta(th, bL, bR, g); return; }
        const Mat& G = (i == 0) ? Glft : (i == n - 2) ? Grgt : Gmid;
        apply_gate_theta_general(th, bL, bR, G, d, qphys);
    };
    const std::map<int,RVec> no_sv;
    auto& A = s.A; auto& bond = s.bond;
    const bool prof = (std::getenv("QX_TEBD_PROFILE") != nullptr);
    using pclk = std::chrono::steady_clock;
    auto ptick = [&](double& acc, pclk::time_point t0){ if (prof) acc += std::chrono::duration<double>(pclk::now()-t0).count(); };
    for (int i = 0; i < n - 1; ++i) {                      // left -> right
        auto tt = pclk::now();
        auto theta = form_theta(A[i], A[i+1], no_sv, bond[i], bond[i+1], bond[i+2], true, qphys);
        bt_expand_theta(theta, bond[i], bond[i+2], qphys);
        ptick(g_tebd_theta, tt); tt = pclk::now();
        apply_gate(theta, bond[i], bond[i+2], i);
        ptick(g_tebd_gate, tt); tt = pclk::now();
        auto svd = block_svd_truncate(theta, bond[i], bond[i+2], chi_max, cutoff, allow_fp32, false, qphys);
        ptick(g_tebd_svd, tt); tt = pclk::now();
        bond[i+1] = svd.new_mid_bs;
        A[i]   = bt_A_from_U(svd, bond[i], bond[i+1], qphys);
        A[i+1] = bt_center_LR(svd);
        ptick(g_tebd_recomb, tt);
    }
    for (int i = n - 2; i >= 0; --i) {                    // right -> left
        auto tt = pclk::now();
        auto theta = form_theta(A[i], A[i+1], no_sv, bond[i], bond[i+1], bond[i+2], true, qphys);
        bt_expand_theta(theta, bond[i], bond[i+2], qphys);
        ptick(g_tebd_theta, tt); tt = pclk::now();
        apply_gate(theta, bond[i], bond[i+2], i);
        ptick(g_tebd_gate, tt); tt = pclk::now();
        auto svd = block_svd_truncate(theta, bond[i], bond[i+2], chi_max, cutoff, allow_fp32, false, qphys);
        ptick(g_tebd_svd, tt); tt = pclk::now();
        bond[i+1] = svd.new_mid_bs;
        A[i+1] = svd.Vt;
        A[i]   = bt_center_RL(svd, bond[i], bond[i+1], qphys);
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
    auto theta = form_theta(s.A[0], s.A[1], no_sv, s.bond[0], s.bond[1], s.bond[2], true,
                            mpo[0].phys_charge);
    WComb wc = build_wcomb(mpo[0], mpo[1]);
    std::vector<int> kk; for (auto& [q,_] : theta) kk.push_back(q);
    const std::vector<int> delta = derive_mpo_sym(mpo).delta;
    HeffPlan plan = build_heff_plan(s.L[0], mpo[0], mpo[1], s.R[2], wc, kk, s.bond[0], s.bond[2], delta);
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
    // Neel start: alternate the highest-Sz local state (charge qphys[0]) and the
    // lowest (qphys.back()) -> |up dn ...> for spin-1/2, |+1 -1 ...> for spin-1.
    const std::vector<int>& qphys = mpo[0].phys_charge;
    std::vector<int> neel(n);
    for (int i=0;i<n;++i) neel[i] = (i%2==0) ? qphys.front() : qphys.back();
    BMPS s = cps_init(mpo, n, neel, qphys);

    TDVP2Out out;
    out.times.push_back(0.0);
    out.stag.push_back(measure_stag(s.A, s.bond, n, qphys));
    auto t0 = clk::now();
    for (int step = 0; step < n_steps; ++step) {
        tdvp2_step_blocks(s, mpo, dt, imag, chi_max, cutoff);
        out.times.push_back((step+1)*dt);
        out.stag.push_back(measure_stag(s.A, s.bond, n, qphys));
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
    const std::vector<int>& qphys = mpo[0].phys_charge;   // spin-1/2 {+1,-1}, spin-1 {+2,0,-2}, ...
    std::vector<int> neel(n);
    for (int i=0;i<n;++i) neel[i] = (i%2==0) ? qphys.front() : qphys.back();
    BMPS s = cps_init(mpo, n, neel, qphys);
    const bool prof = (std::getenv("QX_TEBD_PROFILE") != nullptr);
    if (prof) { g_tebd_theta=g_tebd_gate=g_tebd_svd=g_tebd_recomb=0; g_tebd_maxblk=g_tebd_nsvd=g_tebd_nlapacke=g_tebd_ngpu=0; }
    g_svd_prof = (std::getenv("QX_SVD_PROFILE") != nullptr);
    if (g_svd_prof) { g_svd_gemm=g_svd_eig=g_svd_back=0; }

    TDVP2Out out;
    auto t0 = clk::now();
    for (int step = 0; step < n_steps; ++step) {
        tebd2_step_blocks(s, dt, imag, chi_max, cutoff, J, delta, /*allow_fp32=*/!imag, &mpo);
        if (measure_stride > 0 && (step % measure_stride == 0 || step == n_steps - 1)) {
            out.times.push_back((step+1)*dt);
            out.stag.push_back(measure_stag(s.A, s.bond, n, qphys));
        }
    }
    out.wall_s = std::chrono::duration<double>(clk::now() - t0).count();
    if (prof) {
        double tot = g_tebd_theta+g_tebd_gate+g_tebd_svd+g_tebd_recomb;
        std::fprintf(stderr, "[QX_TEBD_PROFILE n=%d chi=%d] evo=%.3fs theta=%.3f(%.0f%%) gate=%.3f(%.0f%%) svd=%.3f(%.0f%%) recomb=%.3f(%.0f%%) maxblk=%d nsvd=%lld\n",
            n, chi_max, out.wall_s, g_tebd_theta, 100*g_tebd_theta/tot, g_tebd_gate, 100*g_tebd_gate/tot,
            g_tebd_svd, 100*g_tebd_svd/tot, g_tebd_recomb, 100*g_tebd_recomb/tot,
            (int)g_tebd_maxblk, g_tebd_nsvd);
        std::fprintf(stderr, "   nsvd=%lld nlapacke=%lld ngpu=%lld (large blocks: LAPACK zheevd / GPU cuSOLVER)\n",
                     g_tebd_nsvd, g_tebd_nlapacke, g_tebd_ngpu);
    }
    if (g_svd_prof) {
        double st = g_svd_gemm+g_svd_eig+g_svd_back;
        std::fprintf(stderr, "   [SVD split] gram-gemm=%.3f(%.0f%%) eigensolve=%.3f(%.0f%%) back=%.3f(%.0f%%)  (of %.3fs gram-SVD)\n",
            g_svd_gemm,100*g_svd_gemm/st, g_svd_eig,100*g_svd_eig/st, g_svd_back,100*g_svd_back/st, st);
    }
    rebuild_mpo_envs(s, mpo);   // TEBD does not maintain MPO envs; bt_energy needs them
    out.energy = bt_energy(s, mpo);
    out.max_bond = 0;
    for (auto& b : s.bond) { int t=0; for (int d : b.dims) t+=d; out.max_bond = std::max(out.max_bond, t); }
    return out;
}

// ===== Dynamical structure factor S(k,omega) ================================
// Apply the Sz charge operator (2*Sz = diag over qphys[s]) on one site of a
// block-MPS in place: scale each physical-state column group s by qphys[s]. For
// spin-1/2 this is sigma^z = diag(+1,-1) (unitary; preserves norm/canonical form);
// for spin-1 it is diag(+2,0,-2) (NOT unitary -- the |0> column is zeroed -- but
// the overlap-env measurements below handle the resulting unnormalized state). The
// qphys[s]==+1 group is a no-op (skipped), so d=2 is bit-identical. Only the MPO
// environments of the modified state go stale.
static void apply_sz_site(BMPS& s, int site,
                          const std::vector<int>& qphys = SPIN_HALF_CHARGES) {
    for (auto& [q_L, blk] : s.A[site]) {
        auto cols = phys_groups(s.bond[site+1], q_L, +1, qphys);
        for (int sc = 0; sc < (int)qphys.size(); ++sc) {
            if (cols[sc].width == 0 || qphys[sc] == 1) continue;   // +1 group: no-op
            blk.middleCols(cols[sc].off, cols[sc].width) *= S(qphys[sc]);
        }
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
    const std::vector<int> delta = derive_mpo_sym(mpo).delta;
    for (int i = n - 1; i >= 0; --i)
        s.R[i] = block_update_right(s.R[i+1], s.A[i], s.bond[i], s.bond[i+1], mpo[i], delta);
}

// Two-MPS overlap left environments OL[i][q] = <bra[0..i-1] | ket[0..i-1]> as a
// (d_bra_i(q) x d_ket_i(q)) block per shared Sz sector q. bra and ket share the
// total Sz, so the overlap is block-diagonal in Sz.
static std::vector<std::map<int,Mat>> overlap_left_envs(const BMPS& bra, const BMPS& ket,
        const std::vector<int>& qphys = SPIN_HALF_CHARGES) {
    int n = bra.n;
    std::vector<std::map<int,Mat>> OL(n + 1);
    OL[0][0] = Mat::Constant(1, 1, S(1));
    for (int i = 0; i < n; ++i) {
        for (auto& [q_L, Olk] : OL[i]) {
            auto ib = bra.A[i].find(q_L); auto ik = ket.A[i].find(q_L);
            if (ib == bra.A[i].end() || ik == ket.A[i].end()) continue;
            const Mat& Ab = ib->second; const Mat& Ak = ik->second;
            auto bcols = phys_groups(bra.bond[i+1], q_L, +1, qphys);
            auto kcols = phys_groups(ket.bond[i+1], q_L, +1, qphys);
            for (int s = 0; s < (int)qphys.size(); ++s) {
                int q_R = q_L + qphys[s], wb = bcols[s].width, wk = kcols[s].width;
                if (wb == 0 || wk == 0) continue;
                Eigen::Ref<const Mat> Abs = Ab.middleCols(bcols[s].off, wb);
                Eigen::Ref<const Mat> Aks = Ak.middleCols(kcols[s].off, wk);
                Mat contrib = Abs.adjoint() * (Olk * Aks);   // (wb x wk)
                auto& dst = OL[i+1][q_R];
                if (dst.size()==0) dst = contrib; else dst += contrib;
            }
        }
    }
    return OL;
}

// Two-MPS overlap right environments OR[i][q] = <bra[i..n-1] | ket[i..n-1]>.
static std::vector<std::map<int,Mat>> overlap_right_envs(const BMPS& bra, const BMPS& ket,
        const std::vector<int>& qphys = SPIN_HALF_CHARGES) {
    int n = bra.n;
    std::vector<std::map<int,Mat>> OR(n + 1);
    for (int qi = 0; qi < bra.bond[n].n_sectors(); ++qi)
        OR[n][bra.bond[n].qns[qi]] = Mat::Constant(1, 1, S(1));
    for (int i = n - 1; i >= 0; --i) {
        for (auto& [q_L, Ab] : bra.A[i]) {
            auto ik = ket.A[i].find(q_L);
            if (ik == ket.A[i].end()) continue;
            const Mat& Ak = ik->second;
            auto bcols = phys_groups(bra.bond[i+1], q_L, +1, qphys);
            auto kcols = phys_groups(ket.bond[i+1], q_L, +1, qphys);
            for (int s = 0; s < (int)qphys.size(); ++s) {
                int q_R = q_L + qphys[s], wb = bcols[s].width, wk = kcols[s].width;
                if (wb == 0 || wk == 0) continue;
                auto it = OR[i+1].find(q_R);
                if (it == OR[i+1].end()) continue;
                Eigen::Ref<const Mat> Abs = Ab.middleCols(bcols[s].off, wb);
                Eigen::Ref<const Mat> Aks = Ak.middleCols(kcols[s].off, wk);
                // conj on the bra (consistent with OL): OR = sum_s conj(Abs) OR' Aks^T
                Mat contrib = Abs.conjugate() * (it->second * Aks.transpose());  // (d_bra_L x d_ket_L)
                auto& dst = OR[i][q_L];
                if (dst.size()==0) dst = contrib; else dst += contrib;
            }
        }
    }
    return OR;
}

// <bra| (2 Sz_j) |ket> for every site j (2*Sz = charge qphys[s]), given the envs.
// op_wts: per-local-state diagonal operator weights; default (nullptr) = qphys[s], the
// 2*Sz charge itself (the historical sigma^z measurement). For a packed-charge model
// the charge is NOT the observable, so callers pass explicit weights (e.g. Hubbard spin
// {0,+1,-1,0} or charge fluctuation {-1,0,0,+1}).
static std::vector<S> measure_sz_all(const BMPS& bra, const BMPS& ket,
                                     const std::vector<std::map<int,Mat>>& OL,
                                     const std::vector<std::map<int,Mat>>& OR,
                                     const std::vector<int>& qphys = SPIN_HALF_CHARGES,
                                     const std::vector<double>* op_wts = nullptr) {
    int n = bra.n;
    std::vector<S> out(n, S(0));
    for (int j = 0; j < n; ++j) {
        S val = 0;
        for (auto& [q_L, Olk] : OL[j]) {
            auto ib = bra.A[j].find(q_L); auto ik = ket.A[j].find(q_L);
            if (ib == bra.A[j].end() || ik == ket.A[j].end()) continue;
            const Mat& Ab = ib->second; const Mat& Ak = ik->second;
            auto bcols = phys_groups(bra.bond[j+1], q_L, +1, qphys);
            auto kcols = phys_groups(ket.bond[j+1], q_L, +1, qphys);
            for (int s = 0; s < (int)qphys.size(); ++s) {
                int q_R = q_L + qphys[s], wb = bcols[s].width, wk = kcols[s].width;
                if (wb == 0 || wk == 0) continue;
                auto it = OR[j+1].find(q_R);
                if (it == OR[j+1].end()) continue;
                double w = op_wts ? (*op_wts)[s] : (double)qphys[s];
                if (w == 0.0) continue;
                Eigen::Ref<const Mat> Abs = Ab.middleCols(bcols[s].off, wb);
                Eigen::Ref<const Mat> Aks = Ak.middleCols(kcols[s].off, wk);
                Mat left = Abs.adjoint() * (Olk * Aks);          // (wb x wk)
                val += S(w) * (left.array() * it->second.array()).sum();
            }
        }
        out[j] = val;
    }
    return out;
}

// Apply a diagonal single-site operator in place: scale physical-state column group
// sc by diag[sc]. Used to build string operators (Sz endpoints + the exp(i pi Sz)
// phase string) for the Haldane string-order parameter.
static void apply_site_diag(BMPS& s, int site, const std::vector<S>& diag,
                            const std::vector<int>& qphys) {
    for (auto& [q_L, blk] : s.A[site]) {
        auto cols = phys_groups(s.bond[site+1], q_L, +1, qphys);
        for (int sc = 0; sc < (int)diag.size(); ++sc)
            if (cols[sc].width > 0) blk.middleCols(cols[sc].off, cols[sc].width) *= diag[sc];
    }
}

// Den Nijs-Rommelse string order parameter of the ground state:
//   O = - <gs| Sz_{i0} ( prod_{i0<k<j0} e^{i pi Sz_k} ) Sz_{j0} |gs> / <gs|gs>,
// the hidden-order signature of the gapped Haldane phase (nonzero for spin-1, where
// a trivial/Neel-ordered phase gives 0). Sz = qphys/2 (physical), and for integer Sz
// the phase e^{i pi Sz} is real (+-1), so the whole correlator is real.
static double string_order(const std::vector<MPOSite>& mpo, int n, int i0, int j0,
                           int chi, double cutoff, int n_sweeps) {
    const std::vector<int>& qphys = mpo[0].phys_charge;
    const int d = (int)qphys.size();
    BMPS gs;
    run(mpo, n, chi, cutoff, n_sweeps, 1e-12, &gs);
    const double PI = std::acos(-1.0);
    std::vector<S> szdiag(d), phdiag(d);
    for (int sc = 0; sc < d; ++sc) {
        double sz = qphys[sc] / 2.0;                 // physical Sz of state sc
        szdiag[sc] = S(sz);
        phdiag[sc] = S(std::cos(PI * sz));           // e^{i pi Sz} (real for integer Sz)
    }
    BMPS phi = gs;
    apply_site_diag(phi, i0, szdiag, qphys);
    for (int k = i0 + 1; k < j0; ++k) apply_site_diag(phi, k, phdiag, qphys);
    apply_site_diag(phi, j0, szdiag, qphys);
    auto OLc = overlap_left_envs(gs, phi, qphys);
    auto OLz = overlap_left_envs(gs, gs,  qphys);
    double C = OLc[n].count(0) ? rp(OLc[n].at(0)(0, 0)) : 0.0;
    double Z = OLz[n].count(0) ? rp(OLz[n].at(0)(0, 0)) : 1.0;
    return -(C / Z);
}

struct SpectralOut {
    std::vector<double> times;             // length n_steps+1
    int l0 = 0, n = 0, max_bond = 0;
    double E0 = 0, wall_s = 0;
    // C[t][j] = <psi0| sigma^z_j e^{-i(H-E0)t} sigma^z_{l0} |psi0>  (real, imag parts)
    std::vector<std::vector<double>> C_re, C_im;
};

// Dynamical structure factor data: C_{j,l0}(t) = <psi0| W_j e^{-i(H-E0)t} W_{l0} |psi0>
// for a central source l0 and a DIAGONAL single-site operator W. Default W = the charge
// itself (sigma^z for spin-1/2). op_wts overrides with per-state weights -- required for
// packed-charge models, where the charge is not the observable (Hubbard spin
// {0,+1,-1,0} = 2Sz, charge fluctuation {-1,0,0,+1} = n-1 at half filling). The GS is
// found in the total_charge sector from init_config (both default to the spin choices).
static SpectralOut run_spectral(const std::vector<MPOSite>& mpo, int n,
                                double dt, int n_steps, int chi_max, double cutoff,
                                int n_sweeps, double J = 1.0, double delta = 1.0,
                                int total_charge = 0,
                                const std::vector<int>* init_config = nullptr,
                                const std::vector<double>* op_wts = nullptr) {
    static_assert(std::is_same_v<S, Cx>, "spectral requires complex scalar");
    using clk = std::chrono::steady_clock;
    SpectralOut out; out.n = n; out.l0 = n / 2;
    const std::vector<int>& qphys = mpo[0].phys_charge;
    const int d = mpo[0].d;

    // 1. ground state (complex BMPS, centre on site 0)
    BMPS psi0;
    DMRGResult gs = run(mpo, n, chi_max, cutoff, n_sweeps, 1e-12, &psi0,
                        total_charge, init_config);
    out.E0 = gs.energy;

    // Normalisation Z = <psi0|psi0> in the overlap convention (the same machinery
    // measures the correlations, so C is normalised exactly by this ratio). The
    // top overlap env sits in the chain's TOTAL charge sector (0 for spin models).
    double Z;
    { auto OLn = overlap_left_envs(psi0, psi0, qphys);
      Z = OLn[n].count(total_charge) ? std::real(OLn[n].at(total_charge)(0,0)) : 1.0; }
    if (std::abs(Z) < 1e-300) Z = 1.0;

    // 2. phi = W_{l0} |psi0>  (still right-canonical; TEBD needs no MPO envs)
    BMPS phi = psi0;
    if (op_wts) {
        std::vector<S> dg(op_wts->size());
        for (size_t s = 0; s < op_wts->size(); ++s) dg[s] = S((*op_wts)[s]);
        apply_site_diag(phi, out.l0, dg, qphys);
    } else {
        apply_sz_site(phi, out.l0, qphys);
    }

    // 3. real-time evolve, measuring C_{j,l0}(t) at each step
    auto t0 = clk::now();
    for (int step = 0; step <= n_steps; ++step) {
        double t = step * dt;
        auto OL = overlap_left_envs(psi0, phi, qphys);
        auto OR = overlap_right_envs(psi0, phi, qphys);
        auto ov = measure_sz_all(psi0, phi, OL, OR, qphys, op_wts);   // <psi0| W_j |phi(t)>
        S phase = std::exp(Cx(0.0, out.E0 * t));
        std::vector<double> re(n), im(n);
        for (int j = 0; j < n; ++j) { S c = phase * ov[j] / S(Z); re[j] = c.real(); im[j] = c.imag(); }
        out.C_re.push_back(re); out.C_im.push_back(im); out.times.push_back(t);
        for (auto& b : phi.bond) { int s=0; for (int d2 : b.dims) s+=d2; out.max_bond = std::max(out.max_bond, s); }
        if (step < n_steps)
            tebd2_step_blocks(phi, dt, /*imag*/false, chi_max, cutoff, J, delta,
                              /*allow_fp32=*/true, d > 2 ? &mpo : nullptr);  // fast NN route
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
            tebd2_step_blocks(psi, dt, /*imag*/false, chi_max, cutoff, Jf, deltaf,
                              /*allow_fp32=*/true);
    }
    out.wall_s = std::chrono::duration<double>(clk::now() - t0).count();
    return out;
}

// ===== Apply a block MPO to a block MPS (for operator-applied observables) ====
// Compress a block MPS to chi_max by symmetric SVD sweeps (centre ends at 0).
static void compress_bmps(BMPS& p, int chi_max, double cutoff,
                          const std::vector<int>& qphys = SPIN_HALF_CHARGES) {
    const int n = p.n; const std::map<int,RVec> no_sv;
    for (int i = 0; i < n - 1; ++i) {
        auto theta = form_theta(p.A[i], p.A[i+1], no_sv, p.bond[i], p.bond[i+1], p.bond[i+2], true, qphys);
        auto svd = block_svd_truncate(theta, p.bond[i], p.bond[i+2], chi_max, cutoff, false, false, qphys);
        p.bond[i+1] = svd.new_mid_bs;
        p.A[i]   = bt_A_from_U(svd, p.bond[i], p.bond[i+1], qphys);
        p.A[i+1] = bt_center_LR(svd);
    }
    for (int i = n - 2; i >= 0; --i) {
        auto theta = form_theta(p.A[i], p.A[i+1], no_sv, p.bond[i], p.bond[i+1], p.bond[i+2], true, qphys);
        auto svd = block_svd_truncate(theta, p.bond[i], p.bond[i+2], chi_max, cutoff, false, false, qphys);
        p.bond[i+1] = svd.new_mid_bs;
        p.A[i+1] = svd.Vt;
        p.A[i]   = bt_center_RL(svd, p.bond[i], p.bond[i+1], qphys);
    }
}

// phi = W|psi> for a block MPO W with any Sz-conserving bond structure. The
// combined bond carries q_c = q_mps + delta[mpo_bond] (delta = the MPO's derived
// per-bond charges); the product is built exactly (combined bonds up to W-wide
// larger), then SVD-compressed to chi_max.
static BMPS apply_mpo_to_bmps(const BMPS& s, const std::vector<MPOSite>& W,
                              int chi_max, double cutoff) {
    const int n = s.n;
    const std::vector<int> wdelta = derive_mpo_sym(W).delta;
    const std::vector<int>& qphys = W[0].phys_charge;
    BMPS p; p.n = n; p.A.resize(n); p.bond.resize(n + 1);
    p.L.assign(n + 1, BlockEnv()); p.R.assign(n + 1, BlockEnv());
    auto mpodim = [&](int ib) { return (ib == 0) ? W[0].chi_l : W[ib-1].chi_r; };
    auto hbd    = [&](int ib, int b) { return (mpodim(ib) == 1) ? 0 : wdelta[b]; };

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
            auto ocols = phys_groups(p.bond[i+1], q_cL, +1, qphys);   // output column groups
            Mat blk = Mat::Zero(dLc, phys_total(ocols));
            for (auto& cL : compsL) {
                auto itA = s.A[i].find(cL.q_mps);
                if (itA == s.A[i].end()) continue;
                const Mat& A = itA->second;
                auto icols = phys_groups(s.bond[i+1], cL.q_mps, +1, qphys);  // input groups
                for (int sp = 0; sp < (int)qphys.size(); ++sp) {   // mps physical (in)
                    if (icols[sp].width == 0) continue;
                    int q_mps_R = cL.q_mps + qphys[sp];
                    Eigen::Ref<const Mat> Asp = A.middleCols(icols[sp].off, icols[sp].width);
                    for (int so = 0; so < (int)qphys.size(); ++so) {   // output physical
                        int q_cR = q_cL + qphys[so], col_base = ocols[so].off;
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
    compress_bmps(p, chi_max, cutoff, qphys);
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
    BMPS phi0 = apply_mpo_to_bmps(psi0, Wj, chi_max, cutoff);   // phi = J|psi0> (FP64)
    BMPS phi  = phi0;

    auto t0 = clk::now();
    for (int step = 0; step <= n_steps; ++step) {
        auto OL = overlap_left_envs(phi0, phi);
        S ov = OL[n].count(0) ? OL[n].at(0)(0,0) : S(0);
        S c = std::exp(Cx(0.0, out.E0 * step * dt)) * ov / S(Zpsi);
        out.times.push_back(step * dt);
        out.C_re.push_back(c.real()); out.C_im.push_back(c.imag());
        for (auto& b : phi.bond) { int s=0; for (int dm : b.dims) s+=dm; out.max_bond = std::max(out.max_bond, s); }
        if (step < n_steps) tebd2_step_blocks(phi, dt, /*imag*/false, chi_max, cutoff, J, delta,
                                              /*allow_fp32=*/true);
    }
    out.wall_s = std::chrono::duration<double>(clk::now() - t0).count();
    return out;
}

// ===== Finite temperature via ancilla purification (thermofield double) ======
// rho(beta) = e^{-beta H}/Z as the pure state |psi(beta)> = e^{-beta H/2}|psi0>
// on an interleaved 2*n_phys chain: physical sites at even positions, ancillas
// at odd. |psi0> = product of (|ud> - |du>)/sqrt(2) Bell pairs, whose physical
// reduced density matrix is I/2^N -- the beta = 0 Gibbs state. Any maximally
// entangled pair state works; the singlet is the Sz-conserving choice (every
// pair carries total Sz 0), so the whole U(1) block machinery applies as-is.

// Build the beta=0 purified state. Right-canonical with the centre on site 0,
// norm 1 (matching the cps_init convention). Bond QNs: 0 between pairs, +-1
// inside a pair. MPO environments are left empty -- call rebuild_mpo_envs
// before any bt_energy (thermal_step_blocks itself needs no environments).
static BMPS purified_init(int n_phys, const std::vector<int>& qphys = SPIN_HALF_CHARGES) {
    const int n = 2 * n_phys, d = (int)qphys.size();
    // Each pair |s>_p |sbar>_a carries the fixed total P = qphys.front()+qphys.back():
    // 0 for the spin models (up pairs with dn, +2 with -2, ...), pack(2,0) for the
    // packed Hubbard charges (|0> pairs with |updn>, |up> with |dn> -- every pair holds
    // exactly 2 particles at Sz 0). The pairing s -> P - qphys[s] must be a BIJECTION
    // (each state a unique partner) or the beta=0 physical RDM is not maximally mixed:
    // that requires an injective qphys -- true for the spins and the packed two-U(1)
    // Hubbard, NOT for single-U(1) Hubbard ({0,1,1,2}); fail loud rather than cool a
    // wrong Gibbs state.
    const int P = qphys.front() + qphys.back();
    for (int a = 0; a < d; ++a) {
        int partner = -1;
        for (int b = 0; b < d; ++b) if (qphys[b] == P - qphys[a]) {
            if (partner >= 0) throw std::runtime_error(
                "purified_init: qphys is not injective (two local states share a charge); "
                "the beta=0 pairing is ambiguous -- use the packed two-U(1) charges");
            partner = b;
        }
        if (partner < 0) throw std::runtime_error(
            "purified_init: local state has no conjugate partner (qphys[t] == P - qphys[s])");
    }
    BMPS s; s.n = n;
    s.bond.resize(n + 1);
    // Odd bonds (inside pair p) carry p*P + the d physical-state charges; even bonds
    // (between pairs) sit at the accumulated pair total p*P (0 for the spin models).
    s.bond[0] = BondSpace::from_pairs({{0, 1}});
    for (int p = 0; p < n_phys; ++p) {
        std::vector<std::pair<int,int>> odd;
        for (int qc : qphys) odd.push_back({p * P + qc, 1});
        std::sort(odd.begin(), odd.end());
        s.bond[2*p+1] = BondSpace::from_pairs(odd);
        s.bond[2*p+2] = BondSpace::from_pairs({{(p + 1) * P, 1}});
    }
    s.A.resize(n);
    const double amp = 1.0 / std::sqrt((double)d);   // |Phi> = (1/sqrt d) sum_s |s>_p |s_bar>_a
    for (int p = 0; p < n_phys; ++p) {
        // physical site 2p (q_L = p*P): amplitude 1/sqrt(d) in each physical-state column.
        auto pcols = phys_groups(s.bond[2*p+1], p * P, +1, qphys);
        Mat phys = Mat::Zero(1, phys_total(pcols));
        for (int sc = 0; sc < d; ++sc) {
            if (pcols[sc].width == 0) continue;
            double sign = (d == 2 && sc == 1) ? -1.0 : +1.0;   // keep the spin-1/2 singlet sign
            phys(0, pcols[sc].off) = S(sign * amp);
        }
        s.A[2*p][p * P] = phys;
        // ancilla site 2p+1: per bond sector q_L = p*P + qphys[sc], the ancilla state
        // with charge P - qphys[sc] returns the bond to (p+1)*P.
        for (int sc = 0; sc < d; ++sc) {
            int qL = p * P + qphys[sc];
            auto acols = phys_groups(s.bond[2*p+2], qL, +1, qphys);
            Mat anc = Mat::Zero(1, phys_total(acols));
            for (int t = 0; t < d; ++t)
                if (qphys[t] == P - qphys[sc] && acols[t].width > 0) { anc(0, acols[t].off) = S(1); break; }
            s.A[2*p+1][qL] = anc;
        }
    }
    s.L.assign(n + 1, BlockEnv());
    s.R.assign(n + 1, BlockEnv());
    return s;
}

// d^2 x d^2 SWAP permutation gate |s1,s2> -> |s2,s1> (Sz-conserving), for the d>2
// swap-and-carry cooling. d=2 uses the hand-coded Gate {1; [[0,1],[1,0]]} instead.
static Mat build_swap_gate(int d) {
    Mat G = Mat::Zero(d * d, d * d);
    for (int s1 = 0; s1 < d; ++s1)
        for (int s2 = 0; s2 < d; ++s2)
            G(s2 * d + s1, s1 * d + s2) = S(1);   // output (s2,s1) <- input (s1,s2)
    return G;
}

// One symmetric swap-and-carry cooling step: applies e^{-dtau*H} to the
// interleaved purified chain (so beta advances by 2*dtau). The physical
// neighbours (p_i, p_{i+1}) sit two sites apart; the L->R half-sweep puts a
// SWAP on each even bond -- carrying p_i rightward past its ancilla -- and the
// physical half-gate exp(-(dtau/2) h) on each odd bond, which by then joins p_i
// and p_{i+1} adjacently. The R->L half-sweep mirrors the sequence (gate then
// swap), restoring the p a p a ... layout. The full step is a palindrome, so
// the Trotter error is O(dtau^3) per step (2nd order), and the bond order is
// monotonic, so the canonical centre stays adjacent to the active bond and
// ends back on site 0 (exactly as in tebd2_step_blocks). SWAP is Sz-conserving:
// d=2 uses the Gate format {pdiag=1; Sz=0 block [[0,1],[1,0]]}, d>2 build_swap_gate.

static void thermal_step_blocks(BMPS& s, double dtau, int chi_max, double cutoff,
                                double J = 1.0, double delta = 1.0,
                                const std::vector<int>& qphys = SPIN_HALF_CHARGES,
                                const std::vector<MPOSite>* phys_mpo = nullptr) {
    const int n = s.n, d = (int)qphys.size();
    const std::map<int,RVec> no_sv;
    auto& A = s.A; auto& bond = s.bond;
    // d=2: the analytic XXZ gate + the Gate-format SWAP (bit-identical). d>2: the
    // physical bond gate from the (non-purified) MPO + the general d^2 SWAP.
    Gate gx2{}, gsw2{}; Mat gxMid, gxLft, gxRgt, gswG;
    if (d == 2) { gx2 = build_xxz_gate(J, delta, Cx(-dtau/2, 0));
                  gsw2 = Gate{S(1), S(0), S(1), S(1), S(0)}; }
    else {
        // On-site terms (Hubbard U) get boundary weights exactly as in tebd2_step_blocks:
        // the gated (odd) interleaved bond i joins the physical pair ((i-1)/2, (i-1)/2+1);
        // the FIRST physical bond (i==1) owns physical site 0 fully, the LAST (i==n-3)
        // owns site n_phys-1 fully, interior sites get 1/2 from each side. Models with
        // no on-site term (spin-1) make the three variants identical -- bit-identical.
        gxMid = build_gate_from_mpo(*phys_mpo, Cx(-dtau/2, 0), 0.5, 0.5);
        gxLft = build_gate_from_mpo(*phys_mpo, Cx(-dtau/2, 0), 1.0, (n == 4) ? 1.0 : 0.5);
        gxRgt = build_gate_from_mpo(*phys_mpo, Cx(-dtau/2, 0), 0.5, 1.0);
        gswG = build_swap_gate(d);
    }
    auto do_bond = [&](int i, bool swap) {
        auto theta = form_theta(A[i], A[i+1], no_sv, bond[i], bond[i+1], bond[i+2], true, qphys);
        bt_expand_theta(theta, bond[i], bond[i+2], qphys);
        if (d == 2) apply_gate_theta(theta, bond[i], bond[i+2], swap ? gsw2 : gx2);
        else {
            const Mat& G = swap ? gswG
                         : (i == 1) ? gxLft : (i == n - 3) ? gxRgt : gxMid;
            apply_gate_theta_general(theta, bond[i], bond[i+2], G, d, qphys);
        }
        return block_svd_truncate(theta, bond[i], bond[i+2], chi_max, cutoff, false, false, qphys);
    };
    for (int i = 0; i <= n - 3; ++i) {                 // left -> right
        auto svd = do_bond(i, (i % 2 == 0));           // SWAP on even bonds, gate on odd
        bond[i+1] = svd.new_mid_bs;
        A[i]   = bt_A_from_U(svd, bond[i], bond[i+1], qphys);
        A[i+1] = bt_center_LR(svd);
    }
    for (int i = n - 3; i >= 0; --i) {                 // right -> left (mirror)
        auto svd = do_bond(i, (i % 2 == 0));
        bond[i+1] = svd.new_mid_bs;
        A[i+1] = svd.Vt;
        A[i]   = bt_center_RL(svd, bond[i], bond[i+1], qphys);
    }
    // imaginary time is non-unitary: renormalize (centre at site 0)
    double nz = 0; for (auto& [q, b] : A[0]) nz += b.squaredNorm();
    nz = std::sqrt(nz); if (nz > 0) for (auto& [q, b] : A[0]) b /= nz;
}

struct ThermalOut {
    std::vector<double> betas, E, chi_zz;
    std::vector<std::vector<double>> sz, szsz;   // [checkpoint][j_phys]
    int n_phys = 0, c = 0, max_bond = 0;
    double wall_s = 0;
};

// Cooling driver: purified_init, n_steps of thermal_step_blocks (each advancing
// beta by dbeta, i.e. dtau = dbeta/2), measuring E / <Z_j> / <Z_c Z_j> / chi_zz
// at checkpoints. All observables act on physical (even) sites only and are
// normalised by Z = <psi|psi> in the overlap convention, so truncation
// norm-drift cancels exactly.
static ThermalOut run_thermal(int n_phys, double J, double delta,
                              double dbeta, int n_steps, int chi_max,
                              double cutoff, int measure_stride, bool with_chi,
                              int d = 2, double mu = 0.0) {
    static_assert(std::is_same_v<S, Cx>, "thermal requires complex scalar");
    using clk = std::chrono::steady_clock;
    const int n = 2 * n_phys, c_phys = n_phys / 2;
    ThermalOut out; out.n_phys = n_phys; out.c = c_phys;
    // d=2: spin-1/2 XXZ purified chain (the analytic gate path). d=3: spin-1 purified
    // chain. d=4: fermionic Hubbard purified chain with the packed two-U(1) charge --
    // by the (J, delta) -> (t, U) convention: J = hopping t, delta = on-site U. The
    // cooling gate comes from a small non-purified physical MPO in every d>2 case.
    std::vector<int> qphys =
        (d == 2) ? SPIN_HALF_CHARGES :
        (d == 3) ? std::vector<int>{+2, 0, -2}
                 : std::vector<int>{pack_q2(0,0), pack_q2(1,+1), pack_q2(1,-1), pack_q2(2,0)};
    auto mpo = (d == 2) ? qx_dmrg::build_xxz_purified_mpo(n_phys, J, delta)
             : (d == 3) ? qx_dmrg::build_spin1_purified_mpo(n_phys, J)
                        : qx_dmrg::build_hubbard_purified_mpo(n_phys, J, delta, mu);
    std::vector<MPOSite> phys_mpo;
    if (d == 3) phys_mpo = qx_dmrg::build_spin1_heisenberg_mpo(3, J);   // bond gate source
    if (d == 4) phys_mpo = qx_dmrg::build_hubbard_mpo(3, J, delta, /*two_u1*/true, mu);
    // The purified chain's total charge: n_phys * (pair total P). 0 for the spin
    // models; pack(2,0) per pair for Hubbard. The top overlap env sits there.
    const int qtot = n_phys * (qphys.front() + qphys.back());
    // Observable weights: the charge VALUE for spins (2Sz); packed charges decode to
    // the particle number n_j via obs_of_charge -- so "Z_j" below means n_j for d=4.
    std::vector<double> wts(qphys.size());
    for (size_t k = 0; k < qphys.size(); ++k) wts[k] = obs_of_charge(qphys[k]);
    std::vector<S> wdiag(wts.size());
    for (size_t k = 0; k < wts.size(); ++k) wdiag[k] = S(wts[k]);
    BMPS s = purified_init(n_phys, qphys);

    auto measure = [&](double beta) {
        rebuild_mpo_envs(s, mpo);                      // bt_energy needs MPO envs
        out.betas.push_back(beta);
        out.E.push_back(bt_energy(s, mpo));
        auto OL = overlap_left_envs(s, s, qphys);
        auto OR = overlap_right_envs(s, s, qphys);
        double Z = OL[n].count(qtot) ? rp(OL[n].at(qtot)(0, 0)) : 1.0;
        if (std::abs(Z) < 1e-300) Z = 1.0;
        auto zall = measure_sz_all(s, s, OL, OR, qphys, &wts);
        std::vector<double> zj(n_phys);
        for (int p = 0; p < n_phys; ++p) zj[p] = rp(zall[2*p]) / Z;
        // <W_j W_src> for all j from one source: phi = W_src|psi> + envs (W is
        // diagonal, so phi keeps the same charge block structure).
        auto zrow = [&](int p_src) {
            BMPS phi = s;
            apply_site_diag(phi, 2 * p_src, wdiag, qphys);
            auto OL2 = overlap_left_envs(s, phi, qphys);
            auto OR2 = overlap_right_envs(s, phi, qphys);
            auto v = measure_sz_all(s, phi, OL2, OR2, qphys, &wts);
            std::vector<double> r(n_phys);
            for (int p = 0; p < n_phys; ++p) r[p] = rp(v[2*p]) / Z;
            return r;
        };
        std::vector<double> zz_c;
        if (with_chi) {     // all pairs -> chi_zz = beta * Var(sum_j Z_j) / n_phys
            double m2 = 0;
            for (int p_src = 0; p_src < n_phys; ++p_src) {
                auto row = zrow(p_src);
                if (p_src == c_phys) zz_c = row;
                for (double v : row) m2 += v;
            }
            double m1 = 0; for (double v : zj) m1 += v;
            out.chi_zz.push_back(beta * (m2 - m1 * m1) / n_phys);
        } else {
            zz_c = zrow(c_phys);
            out.chi_zz.push_back(0.0);
        }
        out.sz.push_back(std::move(zj));
        out.szsz.push_back(std::move(zz_c));
    };

    auto t0 = clk::now();
    measure(0.0);
    for (int step = 1; step <= n_steps; ++step) {
        thermal_step_blocks(s, dbeta / 2, chi_max, cutoff, J, delta, qphys,
                            (d > 2) ? &phys_mpo : nullptr);
        for (auto& b : s.bond) { int t = 0; for (int dd : b.dims) t += dd;
            out.max_bond = std::max(out.max_bond, t); }
        if ((measure_stride > 0 && step % measure_stride == 0) || step == n_steps)
            measure(step * dbeta);
    }
    out.wall_s = std::chrono::duration<double>(clk::now() - t0).count();
    return out;
}

// ===== Finite-temperature dynamics (F1): real-time purified evolution ========
// One real-time swap-and-carry sweep over the interleaved chain. Bonds in
// [lo,hi] carry a SWAP or the homogeneous gate `g` (by parity:
// `(i%2==0)==swap_on_even` selects the SWAP); bonds outside [lo,hi] get an
// identity (transport-only) step so the canonical centre still traverses the
// whole chain and returns to site 0. The sweep is a palindrome (L->R then R->L
// with half-gates), so it is 2nd-order in dt; SWAPs are exact permutations and
// add no Trotter error. Real time is unitary -- no renormalization.
//   swap_on_even=true,  lo=0, hi=n-3  -> physical sublattice (even sites)
//   swap_on_even=false, lo=1, hi=n-2  -> ancilla  sublattice (odd  sites)
// The physical sweep stops at n-3: the last physical pair is gated at bond n-3,
// so bond n-2 carries no gate. Leaving it as identity transport (rather than a
// SWAP that is undone next half-sweep) avoids a needless SWAP+SVD -- and any
// doubt about a chi-limited SVD there perturbing the state (the terminal bond's
// Schmidt rank is <=2, so it never actually truncates, but this keeps it clean
// and matches F0's thermal_step_blocks range).
static void rt_sweep(BMPS& s, bool swap_on_even, int lo, int hi, Cx tau_half,
                     int chi_max, double cutoff, double J, double delta,
                     const std::vector<int>& qphys = SPIN_HALF_CHARGES,
                     const std::vector<MPOSite>* phys_mpo = nullptr) {
    const int n = s.n, d = (int)qphys.size();
    const std::map<int,RVec> no_sv;
    auto& A = s.A; auto& bond = s.bond;
    // d=2: analytic XXZ gate + Gate SWAP (bit-identical). d>2: MPO-derived physical
    // gate (at tau_half) + general d^2 SWAP, applied via apply_gate_theta_general.
    // On-site terms (Hubbard U) take boundary weights: the sweep's FIRST gated bond
    // owns its left site's on-site term fully, the LAST its right site's (interior
    // sites get 1/2 from each neighbouring gate). Models with no on-site term
    // (spins; also every current MPO) make the three variants identical.
    Gate gxxz2{}, gsw2{}; Mat gxMid, gxLft, gxRgt, gswG;
    int firstGate = -1, lastGate = -1;
    for (int i = lo; i <= hi; ++i)
        if ((i % 2 == 0) != swap_on_even) { if (firstGate < 0) firstGate = i; lastGate = i; }
    if (d == 2) { gxxz2 = build_xxz_gate(J, delta, tau_half);
                  gsw2 = Gate{S(1), S(0), S(1), S(1), S(0)}; }
    else {
        gxMid = build_gate_from_mpo(*phys_mpo, tau_half, 0.5, 0.5);
        gxLft = build_gate_from_mpo(*phys_mpo, tau_half, 1.0, (firstGate == lastGate) ? 1.0 : 0.5);
        gxRgt = build_gate_from_mpo(*phys_mpo, tau_half, 0.5, 1.0);
        gswG = build_swap_gate(d);
    }
    enum Kind { NONE, GATE, SWAP };
    auto kind = [&](int i) -> Kind {
        if (i < lo || i > hi) return NONE;                   // identity transport
        return ((i % 2 == 0) == swap_on_even) ? SWAP : GATE;
    };
    auto theta_at = [&](int i, Kind k) {
        auto theta = form_theta(A[i], A[i+1], no_sv, bond[i], bond[i+1], bond[i+2], true, qphys);
        bt_expand_theta(theta, bond[i], bond[i+2], qphys);
        if (k != NONE) {
            if (d == 2) apply_gate_theta(theta, bond[i], bond[i+2], k==SWAP ? gsw2 : gxxz2);
            else {
                const Mat& G = (k == SWAP) ? gswG
                             : (i == firstGate) ? gxLft : (i == lastGate) ? gxRgt : gxMid;
                apply_gate_theta_general(theta, bond[i], bond[i+2], G, d, qphys);
            }
        }
        return block_svd_truncate(theta, bond[i], bond[i+2], chi_max, cutoff,
                                  /*allow_fp32=*/true, /*stable=*/false, qphys);
    };
    for (int i = 0; i <= n - 2; ++i) {                       // left -> right
        auto svd = theta_at(i, kind(i));
        bond[i+1] = svd.new_mid_bs;
        A[i]   = bt_A_from_U(svd, bond[i], bond[i+1], qphys);
        A[i+1] = bt_center_LR(svd);
    }
    for (int i = n - 2; i >= 0; --i) {                       // right -> left
        auto svd = theta_at(i, kind(i));
        bond[i+1] = svd.new_mid_bs;
        A[i+1] = svd.Vt;
        A[i]   = bt_center_RL(svd, bond[i], bond[i+1], qphys);
    }
}

// One symmetric real-time step e^{-i dt H_phys} (and, if `disentangle`, the
// backward-ancilla gauge e^{+i dt H_anc}). H_phys and H_anc act on disjoint
// sublattices and therefore commute exactly, so the two sweeps compose with no
// added Trotter error -- the only error is the standard O(dt^3) within each
// nearest-neighbour sweep. Each gate-bond is hit twice (L->R + R->L) with
// tau_half, so a physical pair sees e^{-i dt h} and an ancilla pair e^{+i dt h}.
static void purified_rt_step(BMPS& s, double dt, int chi_max, double cutoff,
                             double J, double delta, bool disentangle,
                             const std::vector<int>& qphys = SPIN_HALF_CHARGES,
                             const std::vector<MPOSite>* phys_mpo = nullptr) {
    const int n = s.n;
    rt_sweep(s, /*swap_on_even=*/true,  /*lo=*/0, /*hi=*/n-3, Cx(0, -dt/2),
             chi_max, cutoff, J, delta, qphys, phys_mpo);    // physical forward
    if (disentangle)
        rt_sweep(s, /*swap_on_even=*/false, /*lo=*/1, /*hi=*/n-2, Cx(0, +dt/2),
                 chi_max, cutoff, J, delta, qphys, phys_mpo);// ancilla backward
}

struct SpectralThermalOut {
    std::vector<double> times;
    std::vector<std::vector<double>> C_re, C_im;
    std::vector<int> bond_t, bond_a_t, bond_b_t;
    int l0 = 0, n_phys = 0, max_bond = 0;
    double beta = 0, E = 0, wall_s = 0;
    bool disentangled = false;
};

// Finite-T dynamical structure factor C_{j,l0}(t;T). See the header comment on
// run_spectral_thermal_blocks for the physics.
static SpectralThermalOut run_spectral_thermal(int n_phys, double J, double delta,
                                               double beta, int n_cool,
                                               double dt, int n_steps, int chi_max,
                                               double cutoff, bool disentangle, int d = 2,
                                               double mu = 0.0) {
    static_assert(std::is_same_v<S, Cx>, "thermal spectral requires complex scalar");
    using clk = std::chrono::steady_clock;
    const int n = 2 * n_phys, l0 = n_phys / 2;
    SpectralThermalOut out; out.n_phys = n_phys; out.l0 = l0;
    out.beta = beta; out.disentangled = disentangle;
    // d=2: spin-1/2 XXZ purified chain. d=3: spin-1. d=4: fermionic Hubbard by the
    // (J, delta) -> (t, U) convention, packed two-U(1), spin source (2Sz) so this is
    // the finite-T SPIN structure factor of the Mott chain. Cooling and real-time
    // gates come from a small non-purified physical MPO for every d>2.
    std::vector<int> qphys =
        (d == 2) ? SPIN_HALF_CHARGES :
        (d == 3) ? std::vector<int>{+2, 0, -2}
                 : std::vector<int>{pack_q2(0,0), pack_q2(1,+1), pack_q2(1,-1), pack_q2(2,0)};
    auto mpo = (d == 2) ? qx_dmrg::build_xxz_purified_mpo(n_phys, J, delta)
             : (d == 3) ? qx_dmrg::build_spin1_purified_mpo(n_phys, J)
                        : qx_dmrg::build_hubbard_purified_mpo(n_phys, J, delta, mu);
    std::vector<MPOSite> phys_mpo;
    if (d == 3) phys_mpo = qx_dmrg::build_spin1_heisenberg_mpo(3, J);
    if (d == 4) phys_mpo = qx_dmrg::build_hubbard_mpo(3, J, delta, /*two_u1*/true, mu);
    const std::vector<MPOSite>* pm = (d > 2) ? &phys_mpo : nullptr;
    // Correlator operator: the charge itself for the spins (2Sz); for the packed
    // Hubbard charges pass the explicit 2Sz weights (the charge is not the observable).
    const std::vector<double> w_hub_spin = {0.0, +1.0, -1.0, 0.0};
    const std::vector<double>* wts = (d == 4) ? &w_hub_spin : nullptr;
    const int qtot = n_phys * (qphys.front() + qphys.back());

    auto bond_of = [](const BMPS& st) {
        int mb = 0; for (auto& b : st.bond) { int t = 0; for (int dd : b.dims) t += dd;
            mb = std::max(mb, t); }
        return mb;
    };

    // 1. cool to |psi(beta)> = e^{-beta H/2}|psi0>  (n_cool imag-time steps,
    //    each e^{-dtau H}; total e^{-(n_cool*dtau) H} = e^{-(beta/2) H}).
    BMPS psi = purified_init(n_phys, qphys);
    const double dtau = beta / (2.0 * std::max(1, n_cool));   // per-step argument
    for (int k = 0; k < n_cool; ++k)
        thermal_step_blocks(psi, dtau, chi_max, cutoff, J, delta, qphys, pm);
    rebuild_mpo_envs(psi, mpo);
    out.E = bt_energy(psi, mpo);

    // normalisation Z = <psi|psi> (overlap convention; constant in t). The top
    // overlap env sits at the chain's total charge (0 for spins, n_phys*P for Hubbard).
    double Z;
    { auto OL = overlap_left_envs(psi, psi, qphys);
      Z = OL[n].count(qtot) ? rp(OL[n].at(qtot)(0,0)) : 1.0; }
    if (std::abs(Z) < 1e-300) Z = 1.0;

    // 2. A = |psi(beta)>, B = W_{l0}|psi(beta)>  (physical site 2*l0; W = 2Sz)
    BMPS A = psi, B = psi;
    if (wts) {
        std::vector<S> dg(wts->size());
        for (size_t k = 0; k < wts->size(); ++k) dg[k] = S((*wts)[k]);
        apply_site_diag(B, 2 * l0, dg, qphys);
    } else {
        apply_sz_site(B, 2 * l0, qphys);
    }

    // 3. real-time evolve both; measure C_{j,l0}(t) = <A(t)|W_{2j}|B(t)>/Z
    auto t0 = clk::now();
    for (int step = 0; step <= n_steps; ++step) {
        auto OL = overlap_left_envs(A, B, qphys);
        auto OR = overlap_right_envs(A, B, qphys);
        auto ov = measure_sz_all(A, B, OL, OR, qphys, wts);
        std::vector<double> re(n_phys), im(n_phys);
        for (int p = 0; p < n_phys; ++p) {
            S c = ov[2*p] / S(Z); re[p] = c.real(); im[p] = c.imag();
        }
        out.C_re.push_back(re); out.C_im.push_back(im);
        out.times.push_back(step * dt);
        int ba = bond_of(A), bb = bond_of(B);
        out.bond_a_t.push_back(ba); out.bond_b_t.push_back(bb);
        out.bond_t.push_back(std::max(ba, bb));
        out.max_bond = std::max(out.max_bond, std::max(ba, bb));
        if (step < n_steps) {
            purified_rt_step(A, dt, chi_max, cutoff, J, delta, disentangle, qphys, pm);
            purified_rt_step(B, dt, chi_max, cutoff, J, delta, disentangle, qphys, pm);
        }
    }
    out.wall_s = std::chrono::duration<double>(clk::now() - t0).count();
    return out;
}

// ===== Finite-temperature transport (F2): current-current correlator =========
struct ConductivityThermalOut {
    std::vector<double> times, C_re, C_im;
    std::vector<int> bond_t;
    int n_phys = 0, max_bond = 0;
    double beta = 0, E = 0, wall_s = 0;
    bool disentangled = false;
};

// Spin conductivity correlator C(t;T) = <J(t)J(0)>_beta on the purified chain.
// Structurally the F1 spectral routine with the local sigma^z source replaced by
// the extensive total current J (a 5-wide MPO): cool to |psi(beta)>, set
// B = J|psi>, real-time evolve both A = e^{-iHt}|psi> and B = e^{-iHt}J|psi>, and
// measure C(t) = <A(t)|J|B(t)> = <J A(t)|B(t)> (J Hermitian) via apply_mpo_to_bmps
// + an overlap, normalised by Z = <psi|psi>.
static ConductivityThermalOut run_conductivity_thermal(int n_phys, double J, double delta,
                                                       double beta, int n_cool,
                                                       double dt, int n_steps, int chi_max,
                                                       double cutoff, bool disentangle, int d = 2,
                                                       double mu = 0.0) {
    static_assert(std::is_same_v<S, Cx>, "thermal conductivity requires complex scalar");
    using clk = std::chrono::steady_clock;
    const int n = 2 * n_phys;
    ConductivityThermalOut out; out.n_phys = n_phys;
    out.beta = beta; out.disentangled = disentangle;
    // d=2: spin-1/2 XXZ + spin current. d=3: spin-1. d=4: fermionic Hubbard by the
    // (J, delta) -> (t, U) convention -- J becomes the CHARGE current. Cooling and
    // real-time gates come from a small non-purified physical MPO for every d>2.
    std::vector<int> qphys =
        (d == 2) ? SPIN_HALF_CHARGES :
        (d == 3) ? std::vector<int>{+2, 0, -2}
                 : std::vector<int>{pack_q2(0,0), pack_q2(1,+1), pack_q2(1,-1), pack_q2(2,0)};
    auto Hmpo = (d == 2) ? qx_dmrg::build_xxz_purified_mpo(n_phys, J, delta)
              : (d == 3) ? qx_dmrg::build_spin1_purified_mpo(n_phys, J)
                         : qx_dmrg::build_hubbard_purified_mpo(n_phys, J, delta, mu);
    auto Jmpo = (d == 2) ? qx_dmrg::build_current_purified_mpo(n_phys, J)
              : (d == 3) ? qx_dmrg::build_spin1_current_purified_mpo(n_phys, J)
                         : qx_dmrg::build_hubbard_current_purified_mpo(n_phys, J);
    std::vector<MPOSite> phys_mpo;
    if (d == 3) phys_mpo = qx_dmrg::build_spin1_heisenberg_mpo(3, J);
    if (d == 4) phys_mpo = qx_dmrg::build_hubbard_mpo(3, J, delta, /*two_u1*/true, mu);
    const std::vector<MPOSite>* pm = (d > 2) ? &phys_mpo : nullptr;
    const int qtot = n_phys * (qphys.front() + qphys.back());

    auto bond_of = [](const BMPS& st) {
        int mb = 0; for (auto& b : st.bond) { int t = 0; for (int dd : b.dims) t += dd;
            mb = std::max(mb, t); }
        return mb;
    };

    // 1. cool to |psi(beta)> = e^{-beta H/2}|psi0>
    BMPS psi = purified_init(n_phys, qphys);
    const double dtau = beta / (2.0 * std::max(1, n_cool));
    for (int k = 0; k < n_cool; ++k)
        thermal_step_blocks(psi, dtau, chi_max, cutoff, J, delta, qphys, pm);
    rebuild_mpo_envs(psi, Hmpo);
    out.E = bt_energy(psi, Hmpo);

    double Z;
    { auto OL = overlap_left_envs(psi, psi, qphys);
      Z = OL[n].count(qtot) ? rp(OL[n].at(qtot)(0, 0)) : 1.0; }
    if (std::abs(Z) < 1e-300) Z = 1.0;

    // 2. A = |psi>, B = J|psi>  (apply_mpo_to_bmps derives qphys from Jmpo)
    BMPS A = psi;
    BMPS B = apply_mpo_to_bmps(psi, Jmpo, chi_max, cutoff);

    // 3. evolve both; C(t) = <J A(t)|B(t)>/Z
    auto t0 = clk::now();
    for (int step = 0; step <= n_steps; ++step) {
        BMPS JA = apply_mpo_to_bmps(A, Jmpo, chi_max, cutoff);
        auto OL = overlap_left_envs(JA, B, qphys);
        S ov = OL[n].count(qtot) ? OL[n].at(qtot)(0, 0) : S(0);
        S c = ov / S(Z);
        out.times.push_back(step * dt);
        out.C_re.push_back(c.real()); out.C_im.push_back(c.imag());
        out.bond_t.push_back(std::max(bond_of(A), bond_of(B)));
        out.max_bond = std::max(out.max_bond, out.bond_t.back());
        if (step < n_steps) {
            purified_rt_step(A, dt, chi_max, cutoff, J, delta, disentangle, qphys, pm);
            purified_rt_step(B, dt, chi_max, cutoff, J, delta, disentangle, qphys, pm);
        }
    }
    out.wall_s = std::chrono::duration<double>(clk::now() - t0).count();
    return out;
}

// -- Block-sparse DMRG sweep --------------------------------------------------

static DMRGResult run(const std::vector<MPOSite>& mpo, int n,
                      int max_bond, double cutoff,
                      int n_sweeps, double tol,
                      BMPS* out_state = nullptr,
                      int total_charge = 0,
                      const std::vector<int>* init_config = nullptr) {
    auto t_start = std::chrono::steady_clock::now();

    // MPO bond Sz charges (Heisenberg's {0,+2,-2,0,0} or whatever the operator
    // structure implies for a general Sz-conserving model) -- threaded into every
    // env / H_eff kernel below. qphys are the local Sz charges (spin-1/2 {+1,-1},
    // spin-1 {+2,0,-2}, ...); the bond spaces and the Neel start derive from them.
    const std::vector<int> delta = derive_mpo_sym(mpo).delta;
    const std::vector<int>& qphys = mpo[0].phys_charge;
    const int d = mpo[0].d;
    // Validate a caller-supplied initial product config: correct length, in-range local
    // state indices, and a total charge equal to total_charge -- otherwise the start
    // would sit in the wrong sector (or index qphys out of bounds) and silently mislead.
    if (init_config) {
        if ((int)init_config->size() != n)
            throw std::runtime_error("run_dmrg_blocks: init_config length must equal n");
        int qsum = 0;
        for (int s : *init_config) {
            if (s < 0 || s >= d)
                throw std::runtime_error("run_dmrg_blocks: init_config state index out of range [0,d)");
            qsum += qphys[s];
        }
        if (qsum != total_charge)
            throw std::runtime_error("run_dmrg_blocks: sum of init_config charges != total_charge");
    }
    // Use the numerically-stable direct SVD for the ground-state truncation so a
    // DMRG that lands on an exactly degenerate ground state (e.g. a frustrated
    // chain at a level crossing) stays canonical/variational. Opt back into the
    // fast Gram/LAPACKE/GPU path with QX_DMRG_FAST_SVD for known-non-degenerate
    // problems (and to exercise those backends).
    const bool dmrg_stable = (std::getenv("QX_DMRG_FAST_SVD") == nullptr);

    // Build bond spaces. The chain carries a fixed total U(1) charge: 0 for the
    // Sz-conserving spin models (GS sits at Sz_total=0), or e.g. half-filling
    // N=n for the Hubbard chain (total_charge != 0). The right boundary carries the
    // whole accumulated charge; make_bond_space pairs left-q with right-(Q-q).
    std::vector<BondSpace> bond(n + 1);
    bond[0] = BondSpace::from_pairs({{0, 1}});             // left boundary: qn=0, dim=1
    bond[n] = BondSpace::from_pairs({{total_charge, 1}});  // right boundary carries the total
    for (int i = 0; i < n - 1; ++i)
        bond[i + 1] = make_bond_space(i, n, max_bond, qphys, total_charge);

    // Initialize block MPS to a product ("Neel-like") state. Default: alternate the
    // highest-Sz local state (index 0) and lowest (index d-1) -- |up dn ...> for
    // spin-1/2, |+1 -1 ...> for spin-1 -- so the start sits in Sz_total=0. A caller
    // may instead pass init_config (a local-state index per site) summing to
    // total_charge -- e.g. Hubbard half-filling |up dn up dn ...> (indices 1,2,...).
    // Random init leads to self-consistent local minima; the product start gives
    // physically motivated environments so the first L->R sweep finds the subspace.
    std::vector<std::map<int,Mat>> block_A(n);
    std::vector<std::map<int,RVec>> block_sv(n - 1);

    {
        int q_neel = 0;   // accumulated charge before site i
        for (int i = 0; i < n; ++i) {
            const BondSpace& bsL = bond[i], &bsR = bond[i + 1];
            // Zero-initialise all sectors at this site
            for (int qi = 0; qi < bsL.n_sectors(); ++qi) {
                int q_L = bsL.qns[qi], dL = bsL.dims[qi];
                block_A[i][q_L] = Mat::Zero(dL, phys_total(phys_groups(bsR, q_L, +1, qphys)));
            }
            // Place the product amplitude in the correct sector and physical-state column.
            // FAIL LOUD if the state's charge path was pruned from the bond-space guess
            // (make_bond_space keeps only the max_bond highest-weight sectors, so an
            // atypical init_config at small chi could lose its sector) -- a silent skip
            // here would start the DMRG from the ZERO state and converge to garbage.
            // The default Neel/SDW configs sit at the path-count peak, never pruned.
            int s_neel = init_config ? (*init_config)[i] : ((i % 2 == 0) ? 0 : d - 1);
            auto it = block_A[i].find(q_neel);
            auto cols = phys_groups(bsR, q_neel, +1, qphys);
            if (bsL.dim(q_neel) <= 0 || it == block_A[i].end() || it->second.rows() == 0 ||
                cols[s_neel].width == 0)
                throw std::runtime_error(
                    "run_dmrg_blocks: the initial product state's charge sector at bond " +
                    std::to_string(i) + " is missing from the bond-space guess (pruned at "
                    "this max_bond); raise max_bond or choose a more typical init_config");
            it->second(0, cols[s_neel].off) = S(1.0);
            q_neel += qphys[s_neel];
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
        // Right boundary: chi_r=1 at site n-1, so only MPO index b=0 is active. The
        // right-vacuum scalar sits in bond[n]'s single sector, whose charge is the
        // chain's total U(1) charge (0 for Sz models, half-filling N=n for Hubbard) --
        // NOT necessarily 0. Placing it at charge 0 when total_charge != 0 leaves every
        // environment empty (no charge match) and the DMRG stuck at the product state.
        Rn.blocks[0][total_charge] = Mat::Constant(1, 1, S(1));
        R[n] = Rn;
    }
    for (int i = n - 1; i >= 0; --i)
        R[i] = block_update_right(R[i+1], block_A[i], bond[i], bond[i+1], mpo[i], delta);

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
    auto expand_theta_sectors = [&qphys](std::map<int,Mat>& theta,
                                   const BondSpace& bs_L, const BondSpace& bs_R2) {
        for (int qi = 0; qi < bs_L.n_sectors(); ++qi) {
            int q_L = bs_L.qns[qi];
            for (int s = 0; s < (int)qphys.size(); ++s) {
                int q_mid = q_L + qphys[s];
                if (theta.count(q_mid)) continue;
                int n_row = phys_total(phys_groups(bs_L,  q_mid, -1, qphys));
                int n_col = phys_total(phys_groups(bs_R2, q_mid, +1, qphys));
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

    // Hybrid: two-site warm-up until the bond dimensions stabilise (the bond
    // distribution across QN sectors is what single-site can't adapt), then switch to
    // cheaper single-site refinement. Opt out with QX_DMRG_NO_SINGLESITE=1.
    // Single-site runs for every model (opt out with QX_DMRG_NO_SINGLESITE=1). It used
    // to corrupt the state at total_charge!=0 (Hubbard), but that was the non-injective-
    // qphys bug in apply_heff_1site (|up> and |dn> share particle number, and only the
    // first was summed) -- now fixed, so single-site converges the Hubbard GS to the same
    // energy as the two-site path (checked vs the Lieb-Wu showcase).
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
                                    bond[i], bond[i+1], bond[i+2], true, qphys);
            expand_theta_sectors(theta, bond[i], bond[i+2]);
            WComb wc = build_wcomb(mpo[i], mpo[i+1]);  // fixed per bond
            std::vector<int> ket_keys; ket_keys.reserve(theta.size());
            for (auto& [q, _] : theta) ket_keys.push_back(q);
            HeffPlan plan = build_heff_plan(L[i], mpo[i], mpo[i+1], R[i+2],
                                            wc, ket_keys, bond[i], bond[i+2], delta);
            tick(t_theta, tt);
            auto mv = [&](const std::map<int,Mat>& v) {
                return apply_heff_plan(plan, v);
            };
            tt = clk::now();
            auto [E, psi_blk] = block_lanczos(mv, theta);
            tick(t_lanczos, tt);
            E0 = E;

            tt = clk::now();
            auto svd = block_svd_truncate(psi_blk, bond[i], bond[i+2], max_bond, cutoff,
                                          /*allow_fp32*/false, /*stable*/dmrg_stable, qphys);
            tick(t_svd, tt);
            sweep_trunc += svd.trunc_err;
            bond[i+1] = svd.new_mid_bs;

            // Rebuild A[i]: U_grp[s][q_mid] fills the physical-state-s column group of A[i][q_L].
            for (int qi = 0; qi < bond[i].n_sectors(); ++qi) {
                int q_L = bond[i].qns[qi], dL = bond[i].dims[qi];
                auto acols = phys_groups(bond[i+1], q_L, +1, qphys);
                Mat blk = Mat::Zero(dL, phys_total(acols));
                for (int s = 0; s < (int)qphys.size(); ++s) {
                    if (acols[s].width == 0 || !svd.U_grp[s].count(acols[s].q)) continue;
                    blk.middleCols(acols[s].off, acols[s].width) = svd.U_grp[s].at(acols[s].q);
                }
                block_A[i][q_L] = blk;
            }
            block_sv[i] = svd.sv;
            block_A[i+1] = svd.Vt;

            tt = clk::now();
            L[i+1] = block_update_left(L[i], block_A[i], bond[i], bond[i+1], mpo[i], delta);
            tick(t_env, tt);
        }

        // Right-to-left half sweep
        for (int i = n - 2; i >= 0; --i) {
            auto tt = clk::now();
            auto theta = form_theta(block_A[i], block_A[i+1], block_sv[i],
                                    bond[i], bond[i+1], bond[i+2], false, qphys);
            expand_theta_sectors(theta, bond[i], bond[i+2]);
            WComb wc = build_wcomb(mpo[i], mpo[i+1]);  // fixed per bond
            std::vector<int> ket_keys; ket_keys.reserve(theta.size());
            for (auto& [q, _] : theta) ket_keys.push_back(q);
            HeffPlan plan = build_heff_plan(L[i], mpo[i], mpo[i+1], R[i+2],
                                            wc, ket_keys, bond[i], bond[i+2], delta);
            tick(t_theta, tt);
            auto mv = [&](const std::map<int,Mat>& v) {
                return apply_heff_plan(plan, v);
            };
            tt = clk::now();
            auto [E, psi_blk] = block_lanczos(mv, theta);
            tick(t_lanczos, tt);
            E0 = E;

            tt = clk::now();
            auto svd = block_svd_truncate(psi_blk, bond[i], bond[i+2], max_bond, cutoff,
                                          /*allow_fp32*/false, /*stable*/dmrg_stable, qphys);
            tick(t_svd, tt);
            sweep_trunc += svd.trunc_err;
            bond[i+1] = svd.new_mid_bs;

            for (int qi = 0; qi < bond[i].n_sectors(); ++qi) {
                int q_L = bond[i].qns[qi], dL = bond[i].dims[qi];
                auto acols = phys_groups(bond[i+1], q_L, +1, qphys);
                Mat blk = Mat::Zero(dL, phys_total(acols));
                for (int s = 0; s < (int)qphys.size(); ++s) {
                    if (acols[s].width == 0 || !svd.U_grp[s].count(acols[s].q)) continue;
                    blk.middleCols(acols[s].off, acols[s].width) = svd.U_grp[s].at(acols[s].q);
                }
                block_A[i][q_L] = blk;
            }
            block_sv[i] = svd.sv;
            block_A[i+1] = svd.Vt;

            auto te = clk::now();
            R[i+1] = block_update_right(R[i+2], block_A[i+1], bond[i+1], bond[i+2], mpo[i+1], delta);
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
        // Centre on site 0 (after the warm-up's final R->L it sits at bond 1):
        // absorb the bond-1 singular values into each physical-state column group.
        std::map<int,Mat> M = block_A[0];
        for (auto& [q_L, blk] : M) {
            auto cols = phys_groups(bond[1], q_L, +1, qphys);
            for (int s = 0; s < (int)qphys.size(); ++s) {
                if (cols[s].width == 0 || !block_sv[0].count(cols[s].q)) continue;
                const RVec& sv = block_sv[0].at(cols[s].q);
                for (int c = 0; c < cols[s].width && c < sv.size(); ++c)
                    blk.col(cols[s].off + c) *= sv[c];
            }
        }

        // DMRG3S subspace-expansion (mixer) schedule: enrich the L->R splits with the
        // L*W*M mixing block at weight mix_alpha (relative to ||M||), decaying per
        // sweep and switching off entirely below 1e-6 -- the final sweeps are then
        // strictly variational, so converged fixed points (and every ED assertion)
        // are unchanged; the mixer only changes the PATH there. This is the standard
        // cure for the truncation-limited crawl (plain single-site can never grow
        // bond content) that made TeNPy's per-sweep convergence beat ours on the
        // width-4 cylinder. QX_MIX0 / QX_MIX_DECAY tune it; QX_MIX0=0 disables.
        double mix_alpha = [](){ const char* e = std::getenv("QX_MIX0");
                                 return e ? std::atof(e) : 0.1; }();
        const double mix_decay = [](){ const char* e = std::getenv("QX_MIX_DECAY");
                                       return e ? std::atof(e) : 0.7; }();
        for (int sweep = ss_start; sweep < n_sweeps; ++sweep) {
            double E0 = 0;
            // L->R: optimise each site, move the centre rightward. The one-site H_eff
            // plan hoists all QN bookkeeping out of the Lanczos matvecs (bit-identical
            // to apply_heff_1site; the cylinder's fragmented sectors made the per-
            // matvec bookkeeping the dominant cost of this whole phase).
            for (int i = 0; i < n; ++i) {
                std::vector<int> mk; mk.reserve(M.size());
                for (auto& [q, _] : M) mk.push_back(q);
                Heff1Plan p1 = build_heff1_plan(L[i], mpo[i], R[i+1], mk,
                                                bond[i], bond[i+1], delta);
                auto mv = [&](const std::map<int,Mat>& v) {
                    return apply_heff1_plan(p1, v);
                };
                auto tt = clk::now();
                auto [E, Mopt] = block_lanczos(mv, M);
                tick(t_lanczos, tt);
                E0 = E;
                if (i < n - 1) {
                    tt = clk::now();
                    SiteMove sp;
                    BondSpace old_mid = bond[i+1];
                    if (mix_alpha > 0) {
                        auto [Mext, bs_ext] = expand_M_LR(Mopt, L[i], mpo[i], bond[i],
                                                          bond[i+1], delta, qphys, mix_alpha);
                        sp = singlesite_split_LR(Mext, bond[i], bs_ext, qphys,
                                                 max_bond, cutoff);
                    } else {
                        sp = singlesite_split_LR(Mopt, bond[i], bond[i+1], qphys);
                    }
                    tick(t_svd, tt);
                    block_A[i] = sp.A;
                    bond[i+1] = sp.new_mid;
                    std::map<int,Mat> Mn;   // absorb centre into the next site
                    if (mix_alpha > 0) {
                        // The centre's columns live on the WIDENED bond; the old right
                        // tensor has rows only for the original columns -- the extension
                        // part multiplies implicit zero rows and drops. Sectors NEW to
                        // the bond get explicit zero blocks so the next site's Lanczos
                        // (Galerkin-projected onto its init sectors) can rotate weight
                        // into the enriched directions.
                        for (auto& [q_R, Cb] : sp.C) {
                            int rows = (int)Cb.rows();
                            int ncols = phys_total(phys_groups(bond[i+2], q_R, +1, qphys));
                            if (ncols == 0) continue;
                            Mat blk = Mat::Zero(rows, ncols);
                            auto it = block_A[i+1].find(q_R);
                            int d_old = old_mid.dim(q_R);
                            if (it != block_A[i+1].end() && d_old > 0 &&
                                it->second.cols() == ncols && Cb.cols() >= d_old)
                                blk = Cb.leftCols(d_old) * it->second;
                            Mn[q_R] = std::move(blk);
                        }
                    } else {
                        for (auto& [q_R, Cb] : sp.C) {
                            auto it = block_A[i+1].find(q_R);
                            if (it != block_A[i+1].end()) Mn[q_R] = Cb * it->second;
                        }
                    }
                    M = std::move(Mn);
                    tt = clk::now();
                    L[i+1] = block_update_left(L[i], block_A[i], bond[i], bond[i+1], mpo[i], delta);
                    tick(t_env, tt);
                } else {
                    M = std::move(Mopt);
                }
            }
            // R->L: optimise each site, move the centre leftward (same plan hoisting).
            for (int i = n - 1; i >= 0; --i) {
                std::vector<int> mk; mk.reserve(M.size());
                for (auto& [q, _] : M) mk.push_back(q);
                Heff1Plan p1 = build_heff1_plan(L[i], mpo[i], R[i+1], mk,
                                                bond[i], bond[i+1], delta);
                auto mv = [&](const std::map<int,Mat>& v) {
                    return apply_heff1_plan(p1, v);
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
                        auto oldcols = phys_groups(old_bond_i, q_Lp, +1, qphys); // A[i-1] old cols
                        auto newcols = phys_groups(sp.new_mid, q_Lp, +1, qphys); // new (kept) cols
                        Mat Mb = Mat::Zero(dLp, phys_total(newcols));
                        for (int s = 0; s < (int)qphys.size(); ++s) {
                            int q_R = newcols[s].q;             // = q_Lp + qphys[s]
                            if (newcols[s].width == 0 || !sp.C.count(q_R)) continue;
                            Mb.middleCols(newcols[s].off, newcols[s].width) =
                                Ab.middleCols(oldcols[s].off, oldcols[s].width) * sp.C[q_R];
                        }
                        Mp[q_Lp] = Mb;
                    }
                    M = std::move(Mp);
                    tt = clk::now();
                    R[i] = block_update_right(R[i+1], block_A[i], bond[i], bond[i+1], mpo[i], delta);
                    tick(t_env, tt);
                } else {
                    M = std::move(Mopt);
                }
            }
            // Leave the centre on site 0 so the state is consistent.
            block_A[0] = M;
            for (auto& [q, blk] : block_sv[0]) blk = RVec::Ones(blk.size());
            // Mixer decay: strictly variational once alpha hits zero.
            mix_alpha *= mix_decay;
            if (mix_alpha < 1e-6) mix_alpha = 0.0;

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
    if (d == 2 && std::getenv("QX_DMRG_1SITE_CHECK")) {   // spin-1/2-only debug check
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
        auto HM = apply_heff_1site(L[0], mpo[0], R[1], M0, bond[0], bond[1], delta);
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
        // Absorb the bond-1 singular values into A[0]. After the two-site loop's
        // R->L half-sweep, block_A[0] holds only the left-canonical U factors -- the
        // Schmidt weights live separately in block_sv[0] (form_theta re-inserts them
        // during sweeps). Exporting the bare A-chain SILENTLY DROPPED them: the
        // stripped state is still charge-conserving (sum rules hold) and the reported
        // Lanczos E0 is right, but every correlator measured on it is wrong at the
        // Schmidt-spread level (~1e-2; caught by the Hubbard T=0 spectral vs ED, with
        // TeNPy as referee -- latent for d=2 under the 1.5e-2 Trotter tolerance).
        // After the single-site phase block_sv[0] is Ones, so this is a no-op there.
        for (auto& [q_L, blk] : out_state->A[0]) {
            auto cols = phys_groups(bond[1], q_L, +1, qphys);
            for (int s = 0; s < (int)qphys.size(); ++s) {
                if (cols[s].width == 0 || !block_sv[0].count(cols[s].q)) continue;
                const RVec& sv = block_sv[0].at(cols[s].q);
                for (int c = 0; c < cols[s].width && c < sv.size(); ++c)
                    blk.col(cols[s].off + c) *= sv[c];
            }
        }
        out_state->L.assign(n + 1, BlockEnv());
        out_state->R.assign(n + 1, BlockEnv());
        { BlockEnv L0; L0.bs = bond[0]; L0.blocks[0][0] = Mat::Constant(1,1,S(1)); out_state->L[0] = L0; }
        { BlockEnv Rn; Rn.bs = bond[n];
          for (int qi = 0; qi < bond[n].n_sectors(); ++qi)
              Rn.blocks[0][bond[n].qns[qi]] = Mat::Constant(1,1,S(1));
          out_state->R[n] = Rn; }
        for (int i = n - 1; i >= 0; --i)
            out_state->R[i] = block_update_right(out_state->R[i+1], block_A[i], bond[i], bond[i+1], mpo[i], delta);
    }

    // QX_PROFILES=1: per-site charge profiles of the converged state (one extra
    // non-destructive centre walk). Same bond-1 Schmidt-weight absorption as the
    // out_state export above (no-op after the single-site phase, sv[0] == Ones).
    if (const char* pe = std::getenv("QX_PROFILES"); pe && std::atoi(pe) != 0) {
        auto Aprof = block_A;
        for (auto& [q_L, blk] : Aprof[0]) {
            auto cols = phys_groups(bond[1], q_L, +1, qphys);
            for (int s = 0; s < (int)qphys.size(); ++s) {
                if (cols[s].width == 0 || !block_sv[0].count(cols[s].q)) continue;
                const RVec& sv = block_sv[0].at(cols[s].q);
                for (int c = 0; c < cols[s].width && c < sv.size(); ++c)
                    blk.col(cols[s].off + c) *= sv[c];
            }
        }
        measure_profiles(Aprof, bond, n, qphys, result.site_n, result.site_2sz);
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
            for (int a = 0; a < op.rows(); ++a)
                for (int b = 0; b < op.cols(); ++b)
                    if (std::abs(op(a, b).imag()) > 1e-13) return false;
    return true;
}

// Entry point: dispatch to the real or complex block-sparse core.
DMRGResult run_dmrg_blocks(const std::vector<MPOSite>& mpo, int n,
                           int max_bond, double cutoff,
                           int n_sweeps, double tol,
                           int total_charge,
                           const std::vector<int>* init_config) {
    require_uniform_local_dim(mpo, "run_dmrg_blocks");
    if (mpo_is_real(mpo))
        return BlockDMRG<double>::run(mpo, n, max_bond, cutoff, n_sweeps, tol,
                                      nullptr, total_charge, init_config);
    return BlockDMRG<Cx>::run(mpo, n, max_bond, cutoff, n_sweeps, tol,
                              nullptr, total_charge, init_config);
}

// Entry point: Haldane string-order parameter of the ground state.
double string_order_blocks(const std::vector<MPOSite>& mpo, int n, int i0, int j0,
                           int max_bond, double cutoff, int n_sweeps) {
    require_uniform_local_dim(mpo, "string_order_blocks");
    if (mpo_is_real(mpo))
        return BlockDMRG<double>::string_order(mpo, n, i0, j0, max_bond, cutoff, n_sweeps);
    return BlockDMRG<Cx>::string_order(mpo, n, i0, j0, max_bond, cutoff, n_sweeps);
}

// Entry point: block-sparse TDVP2 time evolution (always complex).
TDVPBlockResult run_tdvp2_blocks(const std::vector<MPOSite>& mpo, int n,
                                 double dt, int n_steps, int chi_max,
                                 double cutoff, bool imag) {
    require_uniform_local_dim(mpo, "run_tdvp2_blocks");   // TDVP2 supports d>2
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
    require_uniform_local_dim(mpo, "run_tebd2_blocks");   // gate built from the MPO for d>2
    auto o = BlockDMRG<Cx>::run_tebd2(mpo, n, dt, n_steps, chi_max, cutoff, imag, measure_stride, J, delta);
    TDVPBlockResult r;
    r.energy = o.energy; r.times = o.times; r.stag = o.stag;
    r.max_bond = o.max_bond; r.wall_time_s = o.wall_s;
    return r;
}

// Entry point: dynamical structure factor S(k,omega) data (always complex).
SpectralResult run_spectral_blocks(const std::vector<MPOSite>& mpo, int n,
                                   double dt, int n_steps, int chi_max,
                                   double cutoff, int n_sweeps, double J, double delta,
                                   int total_charge,
                                   const std::vector<int>* init_config,
                                   const std::vector<double>* op_wts) {
    require_uniform_local_dim(mpo, "run_spectral_blocks");   // d>2 evolves via the MPO gate
    auto o = BlockDMRG<Cx>::run_spectral(mpo, n, dt, n_steps, chi_max, cutoff, n_sweeps,
                                         J, delta, total_charge, init_config, op_wts);
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
    require_local_dim2(mpo_i, "run_loschmidt_blocks");   // TEBD-gate evolution is spin-1/2-only
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
    require_local_dim2(mpo_H, "run_conductivity_blocks");   // TEBD-gate evolution is spin-1/2-only
    auto o = BlockDMRG<Cx>::run_conductivity(mpo_H, n, dt, n_steps, chi_max, cutoff,
                                             n_sweeps, J, delta);
    ConductivityResult r;
    r.times = o.times; r.C_re = o.C_re; r.C_im = o.C_im;
    r.n = o.n; r.max_bond = o.max_bond; r.E0 = o.E0; r.wall_time_s = o.wall_s;
    return r;
}

// Entry point: finite-temperature statics via ancilla purification (always
// complex -- the cooling gates are real, but keeping the complex core means the
// GPU SVD path and the downstream real-time (F1) evolution apply unchanged).
ThermalResult run_thermal_blocks(int n_phys, double J, double delta,
                                 double dbeta, int n_steps, int chi_max,
                                 double cutoff, int measure_stride, bool with_chi,
                                 int d, double mu) {
    auto o = BlockDMRG<Cx>::run_thermal(n_phys, J, delta, dbeta, n_steps,
                                        chi_max, cutoff, measure_stride, with_chi, d, mu);
    ThermalResult r;
    r.betas = o.betas; r.E = o.E; r.chi_zz = o.chi_zz;
    r.sz = o.sz; r.szsz = o.szsz;
    r.n_phys = o.n_phys; r.c = o.c; r.max_bond = o.max_bond;
    r.wall_time_s = o.wall_s;
    return r;
}

// Entry point: finite-T dynamical structure factor S(k,omega,T) data (F1).
SpectralThermalResult run_spectral_thermal_blocks(
        int n_phys, double J, double delta, double beta, int n_cool,
        double dt, int n_steps, int chi_max, double cutoff, bool disentangle, int d,
        double mu) {
    auto o = BlockDMRG<Cx>::run_spectral_thermal(n_phys, J, delta, beta, n_cool,
                                                 dt, n_steps, chi_max, cutoff,
                                                 disentangle, d, mu);
    SpectralThermalResult r;
    r.times = o.times; r.C_re = o.C_re; r.C_im = o.C_im;
    r.bond_t = o.bond_t; r.bond_a_t = o.bond_a_t; r.bond_b_t = o.bond_b_t;
    r.l0 = o.l0; r.n_phys = o.n_phys; r.max_bond = o.max_bond;
    r.beta = o.beta; r.E = o.E; r.wall_time_s = o.wall_s;
    r.disentangled = o.disentangled;
    return r;
}

// Entry point: finite-T spin conductivity current-current correlator (F2).
ConductivityThermalResult run_conductivity_thermal_blocks(
        int n_phys, double J, double delta, double beta, int n_cool,
        double dt, int n_steps, int chi_max, double cutoff, bool disentangle, int d,
        double mu) {
    auto o = BlockDMRG<Cx>::run_conductivity_thermal(n_phys, J, delta, beta, n_cool,
                                                     dt, n_steps, chi_max, cutoff,
                                                     disentangle, d, mu);
    ConductivityThermalResult r;
    r.times = o.times; r.C_re = o.C_re; r.C_im = o.C_im; r.bond_t = o.bond_t;
    r.n_phys = o.n_phys; r.max_bond = o.max_bond;
    r.beta = o.beta; r.E = o.E; r.wall_time_s = o.wall_s;
    r.disentangled = o.disentangled;
    return r;
}

} // namespace qx_dmrg

