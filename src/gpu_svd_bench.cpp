// Phase A kernel benchmark (docs/gpu_acceleration_plan.md, benchmark step "isolate
// the GPU speedup"): the per-block Gram-trick thin SVD, CPU vs GPU, as a function
// of block dimension d. This is exactly the O(d^3) kernel that dominates a TEBD
// step at large chi (the dominant central-Sz block is ~0.6*chi wide), so its
// CPU->GPU scaling is the cleanest evidence that the QuantX advantage grows with
// chi. The CPU baseline replicates thin_svd_gram (Eigen SelfAdjointEigenSolver on
// the Gram matrix) bit-for-bit.
#include "gpu_svd.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <complex>
#include <cstdio>
#include <random>
#include <vector>

using Cx = std::complex<double>;
using Cm = Eigen::MatrixXcd;
using Rv = Eigen::VectorXd;
using clk = std::chrono::steady_clock;

// CPU Gram-trick thin SVD -- identical algorithm to thin_svd_gram in dmrg_blocks.cpp.
static void cpu_gram_svd(const Cm& M, Cm& U, Cm& V, Rv& sv) {
    const int m = (int)M.rows(), nc = (int)M.cols();
    if (m <= nc) {
        Eigen::SelfAdjointEigenSolver<Cm> es(M * M.adjoint());
        const Rv& ev = es.eigenvalues(); const Cm& Q = es.eigenvectors();
        U.resize(m, m); sv.resize(m);
        for (int j = 0; j < m; ++j) { double l = ev(m-1-j); sv(j) = std::sqrt(l>0?l:0.0); U.col(j) = Q.col(m-1-j); }
        Cm MhU = M.adjoint() * U; V.resize(nc, m);
        double smax = sv(0) > 0 ? sv(0) : 1.0;
        for (int j = 0; j < m; ++j) { if (sv(j) > 1e-14*smax) V.col(j) = MhU.col(j)/sv(j); else V.col(j).setZero(); }
    } else {
        Eigen::SelfAdjointEigenSolver<Cm> es(M.adjoint() * M);
        const Rv& ev = es.eigenvalues(); const Cm& Q = es.eigenvectors();
        V.resize(nc, nc); sv.resize(nc);
        for (int j = 0; j < nc; ++j) { double l = ev(nc-1-j); sv(j) = std::sqrt(l>0?l:0.0); V.col(j) = Q.col(nc-1-j); }
        Cm MV = M * V; U.resize(m, nc);
        double smax = sv(0) > 0 ? sv(0) : 1.0;
        for (int j = 0; j < nc; ++j) { if (sv(j) > 1e-14*smax) U.col(j) = MV.col(j)/sv(j); else U.col(j).setZero(); }
    }
}

int main() {
    const bool gpu = qx_gpu::gpu_available();
    std::printf("Gram-SVD kernel: CPU (Eigen) vs GPU (cuSOLVER), complex double, square d x d blocks\n");
    std::printf("GPU available: %s\n\n", gpu ? "yes" : "NO (CPU only)");
    std::printf("%6s | %12s | %12s | %8s | %10s\n", "d", "CPU ms", "GPU ms", "speedup", "CPU GFLOP/s");
    std::printf("-------+--------------+--------------+----------+-----------\n");

    std::mt19937_64 rng(7);
    std::normal_distribution<double> g(0.0, 1.0);
    const std::vector<int> ds = {32,64,96,128,192,256,384,512,768,1024,1536,2048};

    for (int d : ds) {
        Cm M(d, d);
        for (int j = 0; j < d; ++j) for (int i = 0; i < d; ++i) M(i,j) = Cx(g(rng), g(rng));
        // fewer reps as d grows to keep the sweep quick but the timing stable
        int reps = d <= 128 ? 20 : (d <= 512 ? 6 : (d <= 1024 ? 3 : 2));

        Cm U, V; Rv sv;
        // CPU (best of reps)
        double cpu_ms = 1e30;
        for (int r = 0; r < reps; ++r) {
            auto t0 = clk::now(); cpu_gram_svd(M, U, V, sv);
            cpu_ms = std::min(cpu_ms, std::chrono::duration<double,std::milli>(clk::now()-t0).count());
        }
        // GPU (warm up once, then best of reps)
        double gpu_ms = 0; bool gok = false;
        if (gpu) {
            Cm Ug(d,d), Vg(d,d); Rv svg(d);
            gok = qx_gpu::thin_svd_gram_gpu_z(reinterpret_cast<const Cx*>(M.data()), d, d,
                       reinterpret_cast<Cx*>(Ug.data()), reinterpret_cast<Cx*>(Vg.data()), svg.data());
            gpu_ms = 1e30;
            for (int r = 0; r < reps && gok; ++r) {
                auto t0 = clk::now();
                gok = qx_gpu::thin_svd_gram_gpu_z(reinterpret_cast<const Cx*>(M.data()), d, d,
                          reinterpret_cast<Cx*>(Ug.data()), reinterpret_cast<Cx*>(Vg.data()), svg.data());
                gpu_ms = std::min(gpu_ms, std::chrono::duration<double,std::milli>(clk::now()-t0).count());
            }
        }
        // Gram-trick FLOPs: M Mᴴ (~4 d^3 real-mul-adds*8) + eigensolve (~ a few d^3) +
        // Mᴴ U (~4 d^3). Use the rough 8*d^3*~3 complex-mul model only as a sanity rate.
        double gflops = (8.0 * 3.0 * (double)d*d*d) / (cpu_ms * 1e6);
        if (gpu && gok)
            std::printf("%6d | %12.3f | %12.3f | %7.2fx | %10.1f\n", d, cpu_ms, gpu_ms, cpu_ms/gpu_ms, gflops);
        else
            std::printf("%6d | %12.3f | %12s | %8s | %10.1f\n", d, cpu_ms, "-", "-", gflops);
    }
    return 0;
}
