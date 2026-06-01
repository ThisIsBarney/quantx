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

namespace qx_su2 {

// number of fusion paths from spin 0 to each multiplet after k spin-1/2's
inline std::vector<std::map<int,long long>> left_paths(int N) {
    std::vector<std::map<int,long long>> P(N + 1);
    P[0][0] = 1;
    for (int k = 1; k <= N; ++k)
        for (auto& [j, c] : P[k-1]) {
            if (j - 1 >= 0) P[k][j-1] += c;
            P[k][j+1] += c;
        }
    return P;
}

// exact bond multiplet structure for a length-N chain, total spin 0
inline std::vector<SpinBond> exact_bonds(int N) {
    auto P = left_paths(N);
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

// random initial MPS with the given bond structure
inline std::vector<Site> init_mps(const std::vector<SpinBond>& bonds, std::mt19937& rng) {
    int N = (int)bonds.size() - 1;
    std::uniform_real_distribution<double> d(-1, 1);
    std::vector<Site> A(N);
    for (int i = 0; i < N; ++i) {
        A[i].left = bonds[i];
        A[i].right = bonds[i+1];
        for (int li = 0; li < bonds[i].n(); ++li)
        for (int ri = 0; ri < bonds[i+1].n(); ++ri) {
            int tjl = bonds[i].tS[li], tjr = bonds[i+1].tS[ri];
            if (!tri_ok(tjl, 1, tjr)) continue;        // jl (x) 1/2 -> jr
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
    int dim = 0;
    // enumerate (jl,jc,jr) with deg from bondL (jl) and bondR (jr)
    void build(const SpinBond& bondL, const SpinBond& bondR) {
        for (int li = 0; li < bondL.n(); ++li)
        for (int ri = 0; ri < bondR.n(); ++ri) {
            int tjl = bondL.tS[li], tjr = bondR.tS[ri];
            int lo = std::max(std::abs(tjl - 1), std::abs(tjr - 1));
            int hi = std::min(tjl + 1, tjr + 1);
            for (int tjc = lo; tjc <= hi; tjc += 2) {
                if (!tri_ok(tjl, 1, tjc) || !tri_ok(tjc, 1, tjr)) continue;
                Blk b{tjl, tjc, tjr, bondL.deg[li], bondR.deg[ri], dim};
                blks.push_back(b);
                dim += b.dl * b.dr;
            }
        }
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

// solve lowest eigenpair of H_eff; returns energy, fills theta_out
inline double solve_2site(const ThetaSpace& sp, const Env& L, const Env& R,
                          const std::vector<SU2Heisenberg::Trans>& trans,
                          Theta& theta_out) {
    int n = sp.dim;
    Eigen::VectorXd w(n);                 // sqrt weight per coordinate, w=2jr+1
    for (auto& b : sp.blks) {
        double wv = std::sqrt((double)(b.tjr + 1));
        for (int k = 0; k < b.dl*b.dr; ++k) w(b.off + k) = wv;
    }
    Eigen::MatrixXd H(n, n);
    for (int J = 0; J < n; ++J) {
        Eigen::VectorXd e = Eigen::VectorXd::Zero(n); e(J) = 1.0;
        Theta col = apply_heff_2site(sp.to_theta(e), L, R, trans);
        H.col(J) = sp.to_vec(col);
    }
    // symmetrize in the weighted inner product: S = W^{1/2} H W^{-1/2}
    Eigen::MatrixXd S(n, n);
    for (int I = 0; I < n; ++I) for (int J = 0; J < n; ++J)
        S(I,J) = w(I) * H(I,J) / w(J);
    Eigen::MatrixXd Ssym = 0.5 * (S + S.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Ssym);
    double E = es.eigenvalues()(0);
    Eigen::VectorXd u = es.eigenvectors().col(0);
    Eigen::VectorXd th = u.array() / w.array();   // Theta = W^{-1/2} u
    theta_out = sp.to_theta(th);
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

    // SVD each jc; store U,V,s in plain matrices; collect singular values
    struct SV { int tjc; int idx; double s; };
    std::map<int, Mat> Umap, Vmap;
    std::map<int, Eigen::VectorXd> Smap;
    std::vector<SV> all;
    for (auto& [tjc, m] : M) {
        Eigen::JacobiSVD<Mat> svd(m, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Umap[tjc] = svd.matrixU();
        Vmap[tjc] = svd.matrixV();
        Smap[tjc] = svd.singularValues();
        for (int i = 0; i < svd.singularValues().size(); ++i) {
            double sv = svd.singularValues()(i);
            if (sv > tol) all.push_back({tjc, i, sv});
        }
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
inline double run_su2_dmrg(int N, double J, int max_mult, int n_sweeps,
                           unsigned seed = 1) {
    auto trans = SU2Heisenberg::transitions(J);
    auto bonds = exact_bonds(N);
    std::mt19937 rng(seed);
    auto A = init_mps(bonds, rng);

    // build right environments
    std::vector<Env> Lenv(N + 1), Renv(N + 1);
    Lenv[0] = boundary_left();
    Renv[N] = boundary_right();
    for (int i = N - 1; i >= 1; --i) Renv[i] = grow_right(Renv[i+1], A[i], trans);

    double E = 0;
    for (int sweep = 0; sweep < n_sweeps; ++sweep) {
        // left -> right
        for (int i = 0; i < N - 1; ++i) {
            ThetaSpace sp; sp.build(A[i].left, A[i+1].right);
            Theta th;
            E = solve_2site(sp, Lenv[i], Renv[i+2], trans, th);
            SpinBond mid;
            svd_split(th, A[i].left, A[i+1].right, +1,
                      (i==0||i==N-2)?0:max_mult, 1e-12, A[i], A[i+1], mid);
            Lenv[i+1] = grow_left(Lenv[i], A[i], trans);
        }
        // right -> left
        for (int i = N - 2; i >= 0; --i) {
            ThetaSpace sp; sp.build(A[i].left, A[i+1].right);
            Theta th;
            E = solve_2site(sp, Lenv[i], Renv[i+2], trans, th);
            SpinBond mid;
            svd_split(th, A[i].left, A[i+1].right, -1,
                      (i==0||i==N-2)?0:max_mult, 1e-12, A[i], A[i+1], mid);
            Renv[i+1] = grow_right(Renv[i+2], A[i+1], trans);
        }
    }
    return E;
}

} // namespace qx_su2
