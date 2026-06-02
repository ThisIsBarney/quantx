#pragma once
// SU(2) two-site DMRG driver (stages 5-6): MPS init with exact multiplet bonds,
// two-site solve (dense H_eff in the (2jr+1)-weighted inner product), SVD split
// with SU(2) (2j+1) canonical weights + truncation, and the sweep loop.
//
// Reduced inner product: <Theta,Theta'> = sum_{jl,jc,jr} (2jr+1) Tr(Th^T Th').
// (Verified: this is the fusion-tree norm; it makes H_eff self-adjoint and
//  reproduces the full-space energy.)

#include "dmrg_su2_core.hpp"
#include <Eigen/Dense>
#include <map>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace qx_su2 {

// coarse phase profiling (printed by run_su2_dmrg when QX_SU2_PROFILE is set)
struct SU2Prof { double build = 0, lanczos = 0, svd = 0, grow = 0, warm = 0; };
inline SU2Prof& su2_prof() { static SU2Prof p; return p; }
inline double su2_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// number of fusion paths from spin 0 to each multiplet after k spin-(tphys/2) sites
inline std::vector<std::map<int,long long>> left_paths(int N, int tphys = 1) {
    std::vector<std::map<int,long long>> P(N + 1);
    P[0][0] = 1;
    for (int k = 1; k <= N; ++k)
        for (auto& [j, c] : P[k-1])
            for (int jn = std::abs(j - tphys); jn <= j + tphys; jn += 2)
                P[k][jn] += c;
    return P;
}

// exact bond multiplet structure for a length-N chain, total spin 0
inline std::vector<SpinBond> exact_bonds(int N, int tphys = 1) {
    auto P = left_paths(N, tphys);
    std::vector<SpinBond> bonds(N + 1);
    for (int k = 0; k <= N; ++k) {
        SpinBond b;
        for (auto& [j, c] : P[k]) {
            long long m = std::min(c, P[N-k].count(j) ? P[N-k][j] : 0LL);
            if (m > 0) { b.tS.push_back(j); b.deg.push_back((int)m); }
        }
        bonds[k] = b;
    }
    return bonds;
}

// cap a bond's total multiplicity to ~max_mult (for the initial MPS; sweeps
// then grow/adjust bonds via the SVD up to the truncation cap). max_mult<=0
// keeps the bond exact.
inline SpinBond cap_bond(const SpinBond& ex, int max_mult) {
    if (max_mult <= 0) return ex;
    int n = ex.n();
    int per = std::max(1, max_mult / std::max(1, n));
    SpinBond b;
    for (int i = 0; i < n; ++i) {
        int d = std::min(ex.deg[i], per);
        if (d > 0) { b.tS.push_back(ex.tS[i]); b.deg.push_back(d); }
    }
    return b;
}

// random initial MPS with the given bond structure
inline std::vector<Site> init_mps(const std::vector<SpinBond>& bonds,
                                  std::mt19937& rng, int tphys = 1) {
    int N = (int)bonds.size() - 1;
    std::uniform_real_distribution<double> d(-1, 1);
    std::vector<Site> A(N);
    for (int i = 0; i < N; ++i) {
        A[i].left = bonds[i];
        A[i].right = bonds[i+1];
        for (int li = 0; li < bonds[i].n(); ++li)
        for (int ri = 0; ri < bonds[i+1].n(); ++ri) {
            int tjl = bonds[i].tS[li], tjr = bonds[i+1].tS[ri];
            if (!tri_ok(tjl, tphys, tjr)) continue;    // jl (x) S_phys -> jr
            Mat m(bonds[i].deg[li], bonds[i+1].deg[ri]);
            for (int a = 0; a < m.rows(); ++a) for (int b = 0; b < m.cols(); ++b) m(a,b) = d(rng);
            A[i].blk[{tjl, tjr}] = m;
        }
    }
    return A;
}

inline Env boundary_left()  { Env L; L.blk[{0, SU2Heisenberg::BEGIN, 0}] = (Mat(1,1)<<1.0).finished(); return L; }
inline Env boundary_right() { Env R; R.blk[{0, SU2Heisenberg::DONE, 0}] = (Mat(1,1)<<1.0).finished(); return R; }

// ----- two-site solve: build dense H_eff, diagonalize, return E and Theta ---
struct ThetaSpace {
    struct Blk { int tjl, tjc, tjr, dl, dr, off; };
    std::vector<Blk> blks;
    std::map<Key3,int> idx;     // (jl,jc,jr) -> index into blks
    int dim = 0;
    // enumerate (jl,jc,jr) with deg from bondL (jl) and bondR (jr)
    void build(const SpinBond& bondL, const SpinBond& bondR, int tphys = 1) {
        for (int li = 0; li < bondL.n(); ++li)
        for (int ri = 0; ri < bondR.n(); ++ri) {
            int tjl = bondL.tS[li], tjr = bondR.tS[ri];
            int lo = std::max(std::abs(tjl - tphys), std::abs(tjr - tphys));
            int hi = std::min(tjl + tphys, tjr + tphys);
            for (int tjc = lo; tjc <= hi; tjc += 2) {
                if (!tri_ok(tjl, tphys, tjc) || !tri_ok(tjc, tphys, tjr)) continue;
                idx[{tjl, tjc, tjr}] = (int)blks.size();
                blks.push_back(Blk{tjl, tjc, tjr, bondL.deg[li], bondR.deg[ri], dim});
                dim += bondL.deg[li] * bondR.deg[ri];
            }
        }
    }
    const Blk* find(int tjl, int tjc, int tjr) const {
        auto it = idx.find({tjl, tjc, tjr});
        return it == idx.end() ? nullptr : &blks[it->second];
    }
    Theta to_theta(const Eigen::VectorXd& v) const {
        Theta th;
        for (auto& b : blks) {
            Mat m(b.dl, b.dr);
            for (int a = 0; a < b.dl; ++a) for (int c = 0; c < b.dr; ++c) m(a,c) = v(b.off + a*b.dr + c);
            th[{b.tjl, b.tjc, b.tjr}] = m;
        }
        return th;
    }
    Eigen::VectorXd to_vec(const Theta& th) const {
        Eigen::VectorXd v = Eigen::VectorXd::Zero(dim);
        for (auto& b : blks) {
            auto it = th.find({b.tjl, b.tjc, b.tjr});
            if (it == th.end()) continue;
            for (int a = 0; a < b.dl; ++a) for (int c = 0; c < b.dr; ++c) v(b.off + a*b.dr + c) = it->second(a,c);
        }
        return v;
    }
};

// contract two adjacent site tensors into a two-site Theta (warm-start guess)
inline Theta form_theta(const Site& Ai, const Site& Aj) {
    Theta th;
    for (auto& [ka, Ma] : Ai.blk) {
        int jl = ka.first, jc = ka.second;
        for (auto& [kb, Mb] : Aj.blk) {
            if (kb.first != jc) continue;
            Key3 k{jl, jc, kb.second};
            Mat prod = Ma * Mb;
            auto it = th.find(k);
            if (it == th.end()) th[k] = prod; else it->second += prod;
        }
    }
    return th;
}

// ----- precomputed two-site H_eff contraction plan -------------------------
// Recoupling computed once at build (cached), reused across every Lanczos
// matvec. The matvec is a flat list of tiny block products replayed via a
// PARALLEL REDUCTION over terms: each thread accumulates a chunk of terms into
// its own private output buffer (load-balanced regardless of how the work
// clusters into output blocks), then the buffers are summed. Per-thread
// reusable scratch + noalias keep the hot loop allocation-free (heap allocs
// inside make many threads serialize on the malloc lock -> catastrophic). Eigen
// GEMM is pinned to 1 thread; the parallelism is ours.
//
// MEASURED FINDING (spin-1/2 Heisenberg): this matvec does NOT scale with
// threads. Multiplicity is spread thin across many multiplets, so the work is
// thousands of tiny ~10x10 GEMMs -- bandwidth/latency-bound, low arithmetic
// intensity -- and warm-started Lanczos needs only ~7 matvecs/solve, too few to
// amortize any heavier precompute (a dense per-block-pair "batched GEMM"
// operator costs more to build than it saves here). Kept because it is neutral
// (serial path when 1 thread) and helps regimes with larger blocks (e.g. higher
// spin / fermionic models). The SVD (below) is what actually parallelizes.
struct Heff2Plan {
    struct Op { const Mat* L; const Mat* R; double coeff;
                int in_off, in_dl, in_dr, out_off, out_dl, out_dr; };
    std::vector<Op> ops;
    int dim = 0;
    mutable std::vector<Eigen::VectorXd> tls;   // per-thread accumulators
    using RMap  = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>;
    using cRMap = Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>;
    static void apply_op(const Op& o, const double* xd, double* yd, Mat& scratch) {
        cRMap Th(xd + o.in_off, o.in_dl, o.in_dr);
        RMap  Y (yd + o.out_off, o.out_dl, o.out_dr);
        scratch.noalias() = Th * (*o.R);                        // (in_dl x out_dr)
        Y.noalias() += o.coeff * (o.L->transpose() * scratch);
    }
    void matvec(const Eigen::VectorXd& x, Eigen::VectorXd& y) const {
        const double* xd = x.data();
        const int no = (int)ops.size();
#ifdef _OPENMP
        if (no > 256 && omp_get_max_threads() > 1) {
            int nt = omp_get_max_threads();
            if ((int)tls.size() != nt) tls.assign(nt, Eigen::VectorXd());
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                Eigen::VectorXd& yl = tls[tid];
                if (yl.size() != dim) yl = Eigen::VectorXd::Zero(dim); else yl.setZero();
                Mat scratch;
                double* yld = yl.data();
                #pragma omp for schedule(dynamic, 64) nowait
                for (int i = 0; i < no; ++i) apply_op(ops[i], xd, yld, scratch);
            }
            y = tls[0];
            for (int t = 1; t < nt; ++t) y += tls[t];
            return;
        }
#endif
        y.setZero(dim);
        double* yd = y.data();
        Mat scratch;
        for (const Op& o : ops) apply_op(o, xd, yd, scratch);
    }
};

inline Heff2Plan build_plan(const ThetaSpace& sp, const Env& L, const Env& R,
                            const std::vector<SU2Heisenberg::Trans>& trans,
                            int tphys = 1) {
    Heff2Plan p; p.dim = sp.dim;
    for (auto& inb : sp.blks) {
        int tjl = inb.tjl, tjc = inb.tjc, tjr = inb.tjr;
        for (auto& [kL, Lb] : L.blk) {
            if (std::get<0>(kL) != tjl) continue;
            int wl = std::get<1>(kL), tjlp = std::get<2>(kL);
            int twl = SU2Heisenberg::bond_tS(wl);
            for (auto& [kR, Rb] : R.blk) {
                if (std::get<0>(kR) != tjr) continue;
                int wr = std::get<1>(kR), tjrp = std::get<2>(kR);
                int twr = SU2Heisenberg::bond_tS(wr);
                for (auto& t1 : trans) {
                    if (t1.a != wl) continue;
                    int wm = t1.b, twm = SU2Heisenberg::bond_tS(wm);
                    for (auto& t2 : trans) {
                        if (t2.a != wm || t2.b != wr) continue;
                        int lo = std::max(std::abs(tjlp - tphys), std::abs(tjrp - tphys));
                        int hi = std::min(tjlp + tphys, tjrp + tphys);
                        for (int tjcp = lo; tjcp <= hi; tjcp += 2) {
                            const ThetaSpace::Blk* outb = sp.find(tjlp, tjcp, tjrp);
                            if (!outb) continue;
                            double Rc = recoup_2site(tjl, tjc, tjr, tjlp, tjcp, tjrp,
                                                     twl, twm, twr, t1.tk, t2.tk, tphys);
                            if (Rc == 0.0) continue;
                            p.ops.push_back({&Lb, &Rb, Rc * t1.r * t2.r,
                                             inb.off, inb.dl, inb.dr,
                                             outb->off, outb->dl, outb->dr});
                        }
                    }
                }
            }
        }
    }
    return p;
}

// Lanczos for the lowest eigenpair of a symmetric operator (full reorthog).
inline double lanczos_lowest(const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& applyH,
                             Eigen::VectorXd v, Eigen::VectorXd& ground) {
    int dim = (int)v.size();
    if (v.norm() < 1e-300) v.setRandom();
    v /= v.norm();
    std::vector<Eigen::VectorXd> V; V.push_back(v);
    std::vector<double> alpha, beta;
    Eigen::VectorXd w = applyH(v);
    alpha.push_back(w.dot(v));
    w -= alpha[0] * v;
    double Eprev = alpha[0];
    int maxit = std::min(dim, 200);
    for (int it = 1; it < maxit; ++it) {
        for (auto& vv : V) w -= w.dot(vv) * vv;   // full reorthogonalization
        double b = w.norm();
        if (b < 1e-12) break;
        beta.push_back(b);
        v = w / b; V.push_back(v);
        w = applyH(v);
        alpha.push_back(w.dot(v));
        w -= alpha.back() * v + b * V[V.size()-2];
        int m = (int)alpha.size();
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
        for (int i = 0; i < m; ++i) { T(i,i) = alpha[i];
            if (i+1 < m) { T(i,i+1) = beta[i]; T(i+1,i) = beta[i]; } }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
        double E0 = es.eigenvalues()(0);
        if (it >= 2 && std::abs(E0 - Eprev) < 1e-12 * std::max(1.0, std::abs(E0))) {
            Eigen::VectorXd z = es.eigenvectors().col(0);
            ground = Eigen::VectorXd::Zero(dim);
            for (int k = 0; k < m; ++k) ground += z(k) * V[k];
            return E0;
        }
        Eprev = E0;
    }
    int m = (int)alpha.size();
    Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
    for (int i = 0; i < m; ++i) { T(i,i) = alpha[i];
        if (i+1 < m) { T(i,i+1) = beta[i]; T(i+1,i) = beta[i]; } }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
    Eigen::VectorXd z = es.eigenvectors().col(0);
    ground = Eigen::VectorXd::Zero(dim);
    for (int k = 0; k < m; ++k) ground += z(k) * V[k];
    return es.eigenvalues()(0);
}

// solve lowest eigenpair of H_eff via Lanczos on the plan; warm-started from
// the current MPS Theta if provided. Works in the (2jr+1)-rescaled basis where
// H_eff is plain-symmetric.
inline double solve_2site(const ThetaSpace& sp, const Env& L, const Env& R,
                          const std::vector<SU2Heisenberg::Trans>& trans,
                          Theta& theta_out, const Theta* warm = nullptr, int tphys = 1) {
    int n = sp.dim;
    if (n == 0) { theta_out = Theta{}; return 0.0; }
    double t0 = su2_ms();
    Heff2Plan plan = build_plan(sp, L, R, trans, tphys);
    su2_prof().build += su2_ms() - t0;
    Eigen::VectorXd sw(n);                 // sqrt weight per coordinate, w = 2jr+1
    for (auto& b : sp.blks) {
        double s = std::sqrt((double)(b.tjr + 1));
        for (int k = 0; k < b.dl*b.dr; ++k) sw(b.off + k) = s;
    }
    auto applyH = [&](const Eigen::VectorXd& u) -> Eigen::VectorXd {
        Eigen::VectorXd x = u.cwiseQuotient(sw), y;
        plan.matvec(x, y);
        return y.cwiseProduct(sw);
    };
    Eigen::VectorXd v0;
    if (warm) v0 = sp.to_vec(*warm).cwiseProduct(sw);
    else { v0 = Eigen::VectorXd::Random(n); }
    Eigen::VectorXd ground;
    double t1 = su2_ms();
    double E = lanczos_lowest(applyH, v0, ground);
    su2_prof().lanczos += su2_ms() - t1;
    theta_out = sp.to_theta(ground.cwiseQuotient(sw));
    return E;
}

// ----- SVD split of Theta at jc, with SU(2) canonical weights + truncation --
// dir = +1 : left->right (A[i]=U left-canonical, center -> A[i+1])
// dir = -1 : right->left (A[i+1]=V right-canonical, center -> A[i])
inline void svd_split(const Theta& th, SpinBond bondL, SpinBond bondR,
                      int dir, int max_mult, double tol,
                      Site& Aleft, Site& Aright, SpinBond& bondMid) {
    // bondL/bondR are taken BY VALUE: Aleft/Aright may alias A[i]/A[i+1] whose
    // .left/.right we are passed, and we clear them below.
    // gather per-jc matrix M^{jc}[(jl,al),(jr,ar)] = sqrt((2jr+1)/(2jc+1)) Theta
    struct RowIdx { int tjl, al; };
    struct ColIdx { int tjr, ar; };
    std::map<int, std::vector<RowIdx>> rows;   // jc -> rows
    std::map<int, std::vector<ColIdx>> cols;
    std::map<int, Mat> M;
    std::map<int, std::map<std::pair<int,int>, int>> rowpos, colpos; // jc -> (tj,a)->index
    for (auto& [k, blk] : th) {
        int tjl = std::get<0>(k), tjc = std::get<1>(k), tjr = std::get<2>(k);
        for (int a = 0; a < blk.rows(); ++a) {
            auto key = std::make_pair(tjl, a);
            if (!rowpos[tjc].count(key)) { rowpos[tjc][key] = (int)rows[tjc].size(); rows[tjc].push_back({tjl,a}); }
        }
        for (int c = 0; c < blk.cols(); ++c) {
            auto key = std::make_pair(tjr, c);
            if (!colpos[tjc].count(key)) { colpos[tjc][key] = (int)cols[tjc].size(); cols[tjc].push_back({tjr,c}); }
        }
    }
    for (auto& [tjc, rl] : rows) M[tjc] = Mat::Zero((int)rl.size(), (int)cols[tjc].size());
    for (auto& [k, blk] : th) {
        int tjl = std::get<0>(k), tjc = std::get<1>(k), tjr = std::get<2>(k);
        double scale = std::sqrt((double)(tjr + 1) / (double)(tjc + 1));
        for (int a = 0; a < blk.rows(); ++a)
        for (int c = 0; c < blk.cols(); ++c)
            M[tjc](rowpos[tjc][{tjl,a}], colpos[tjc][{tjr,c}]) = scale * blk(a,c);
    }

    // SVD each jc (independent sectors -> parallel). BDCSVD (divide & conquer)
    // is much faster than JacobiSVD for the larger blocks.
    struct SV { int tjc; int idx; double s; };
    std::vector<int> jcs; std::vector<Mat*> Mptr;
    for (auto& [tjc, m] : M) { jcs.push_back(tjc); Mptr.push_back(&m); }
    int nj = (int)jcs.size();
    std::vector<Mat> Uv(nj), Vv(nj);
    std::vector<Eigen::VectorXd> Sv(nj);
    #pragma omp parallel for schedule(dynamic) if(nj > 2)
    for (int j = 0; j < nj; ++j) {
        Eigen::BDCSVD<Mat> svd(*Mptr[j], Eigen::ComputeThinU | Eigen::ComputeThinV);
        Uv[j] = svd.matrixU();
        Vv[j] = svd.matrixV();
        Sv[j] = svd.singularValues();
    }
    std::map<int, Mat> Umap, Vmap;
    std::map<int, Eigen::VectorXd> Smap;
    std::vector<SV> all;
    for (int j = 0; j < nj; ++j) {
        int tjc = jcs[j];
        for (int i = 0; i < Sv[j].size(); ++i)
            if (Sv[j](i) > tol) all.push_back({tjc, i, Sv[j](i)});
        Umap[tjc] = std::move(Uv[j]);
        Vmap[tjc] = std::move(Vv[j]);
        Smap[tjc] = std::move(Sv[j]);
    }
    // rank by (2jc+1) s^2 descending (discarded-weight optimal); cap multiplets
    std::sort(all.begin(), all.end(), [](const SV&a, const SV&b){
        return (a.tjc+1)*a.s*a.s > (b.tjc+1)*b.s*b.s; });
    if (max_mult > 0 && (int)all.size() > max_mult) all.resize(max_mult);
    std::map<int,std::vector<int>> kept_idx;
    for (auto& sv : all) kept_idx[sv.tjc].push_back(sv.idx);
    for (auto& [tjc, v] : kept_idx) std::sort(v.begin(), v.end());

    // build new middle bond + tensors
    bondMid = SpinBond{};
    Aleft = Site{}; Aright = Site{};
    Aleft.left = bondL;  Aright.right = bondR;
    for (auto& [tjc, idxs] : kept_idx) {
        int nk = (int)idxs.size();
        if (nk == 0) continue;
        bond_add(bondMid, tjc, nk);
        const Mat& U = Umap[tjc];
        const Mat& V = Vmap[tjc];
        const Eigen::VectorXd& s = Smap[tjc];
        // U: rows[tjc] x rank ; V: cols[tjc] x rank
        // A_left[jl,jc][al,sigma] = U(row,sigma)      (dir +1, left-canonical)
        // A_right[jc,jr][sigma,ar] = s * sqrt((2jc+1)/(2jr+1)) V^T   (dir +1)
        // dir -1: A_right = sqrt((2jc+1)/(2jr+1)) V^T (right-canonical),
        //         A_left  = U * s
        // group rows by tjl, cols by tjr
        std::map<int,std::vector<int>> rowsByJ, colsByJ;
        for (int r = 0; r < (int)rows[tjc].size(); ++r) rowsByJ[rows[tjc][r].tjl].push_back(r);
        for (int c = 0; c < (int)cols[tjc].size(); ++c) colsByJ[cols[tjc][c].tjr].push_back(c);
        for (auto& [tjl, rlist] : rowsByJ) {
            Mat blk((int)rlist.size(), nk);
            for (int rr = 0; rr < (int)rlist.size(); ++rr)
                for (int kk = 0; kk < nk; ++kk) {
                    double u = U(rlist[rr], idxs[kk]);
                    blk(rr, kk) = (dir > 0) ? u : u * s(idxs[kk]);
                }
            Aleft.blk[{tjl, tjc}] = blk;
        }
        for (auto& [tjr, clist] : colsByJ) {
            double wfac = std::sqrt((double)(tjc + 1) / (double)(tjr + 1));
            Mat blk(nk, (int)clist.size());
            for (int kk = 0; kk < nk; ++kk)
                for (int cc = 0; cc < (int)clist.size(); ++cc) {
                    double v = V(clist[cc], idxs[kk]);   // V^T(sigma,col)=V(col,sigma)
                    double val = wfac * v;
                    blk(kk, cc) = (dir > 0) ? s(idxs[kk]) * val : val;
                }
            Aright.blk[{tjc, tjr}] = blk;
        }
    }
    Aleft.right = bondMid;
    Aright.left = bondMid;
}

// ----- full sweep driver ---------------------------------------------------
// tS_phys is the doubled physical spin (1 = spin-1/2 [Pauli sigma.sigma, matches
// the U(1) path], 2 = spin-1 [S.S, Haldane chain], ...). Targets total spin 0.
inline double run_su2_dmrg(int N, double J, int max_mult, int n_sweeps,
                           unsigned seed = 1, int tS_phys = 1) {
    Eigen::setNbThreads(1);   // we parallelize at the group/sector level; avoid nested GEMM threads
    const int tp = tS_phys;
    auto trans = SU2Heisenberg::transitions(J, tp);
    auto bonds = exact_bonds(N, tp);
    for (auto& b : bonds) b = cap_bond(b, max_mult);   // bound the initial MPS size
    std::mt19937 rng(seed);
    auto A = init_mps(bonds, rng, tp);

    // build right environments
    std::vector<Env> Lenv(N + 1), Renv(N + 1);
    Lenv[0] = boundary_left();
    Renv[N] = boundary_right();
    for (int i = N - 1; i >= 1; --i) Renv[i] = grow_right(Renv[i+1], A[i], trans, tp);

    double E = 0;
    double tt;
    for (int sweep = 0; sweep < n_sweeps; ++sweep) {
        // left -> right
        for (int i = 0; i < N - 1; ++i) {
            ThetaSpace sp; sp.build(A[i].left, A[i+1].right, tp);
            tt = su2_ms(); Theta warm = form_theta(A[i], A[i+1]), th; su2_prof().warm += su2_ms()-tt;
            E = solve_2site(sp, Lenv[i], Renv[i+2], trans, th, &warm, tp);
            SpinBond mid;
            tt = su2_ms();
            svd_split(th, A[i].left, A[i+1].right, +1,
                      (i==0||i==N-2)?0:max_mult, 1e-12, A[i], A[i+1], mid);
            su2_prof().svd += su2_ms()-tt;
            tt = su2_ms(); Lenv[i+1] = grow_left(Lenv[i], A[i], trans, tp); su2_prof().grow += su2_ms()-tt;
        }
        // right -> left
        for (int i = N - 2; i >= 0; --i) {
            ThetaSpace sp; sp.build(A[i].left, A[i+1].right, tp);
            tt = su2_ms(); Theta warm = form_theta(A[i], A[i+1]), th; su2_prof().warm += su2_ms()-tt;
            E = solve_2site(sp, Lenv[i], Renv[i+2], trans, th, &warm, tp);
            SpinBond mid;
            tt = su2_ms();
            svd_split(th, A[i].left, A[i+1].right, -1,
                      (i==0||i==N-2)?0:max_mult, 1e-12, A[i], A[i+1], mid);
            su2_prof().svd += su2_ms()-tt;
            tt = su2_ms(); Renv[i+1] = grow_right(Renv[i+2], A[i+1], trans, tp); su2_prof().grow += su2_ms()-tt;
        }
    }
    if (std::getenv("QX_SU2_PROFILE")) {
        auto& p = su2_prof();
        fprintf(stderr, "  [prof N=%d] build=%.0f lanczos=%.0f svd=%.0f grow=%.0f warm=%.0f ms\n",
                N, p.build, p.lanczos, p.svd, p.grow, p.warm);
        p = SU2Prof{};
    }
    return E;
}

} // namespace qx_su2
