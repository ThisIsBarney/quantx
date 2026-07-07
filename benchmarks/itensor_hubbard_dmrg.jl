# ITensor Hubbard ground-state DMRG single point, for benchmarks/hubbard_dmrg_vs_itensor.py.
# Same problem as the QuantX/TeNPy GS benchmark: H = -t sum(c^dag c + h.c.) + U sum n_up n_dn,
# open chain, half filling, conserved N AND Sz ("Electron" sites, conserve_qns=true),
# fixed maxdim = chi, svd cutoff 1e-12, sweep until |dE| < 1e-10 (DMRGObserver).
#
# JIT fairness: a small warm-up run (N=8, chi=32) compiles every method first; the timed
# run then measures algorithm time only (the same convention as excluding Python import
# time for TeNPy). BLAS threads are a CLI arg so the driver can probe ITensor's optimum.
#
# Usage: julia itensor_hubbard_dmrg.jl [N] [U] [chi] [blas_threads]
# Prints: RESULT <wall_seconds> <energy> <sweeps_done>
using LinearAlgebra
using ITensors, ITensorMPS

function run_dmrg_hubbard(N::Int, U::Float64, chi::Int; nsweeps::Int=40)
    sites = siteinds("Electron", N; conserve_qns=true)
    os = OpSum()
    for i in 1:(N-1)
        os += -1.0, "Cdagup", i, "Cup", i+1
        os += -1.0, "Cdagup", i+1, "Cup", i
        os += -1.0, "Cdagdn", i, "Cdn", i+1
        os += -1.0, "Cdagdn", i+1, "Cdn", i
    end
    for i in 1:N
        os += U, "Nupdn", i
    end
    H = MPO(os, sites)
    state = [isodd(i) ? "Up" : "Dn" for i in 1:N]   # SDW half-filling start
    psi0 = MPS(sites, state)
    obs = DMRGObserver(; energy_tol=1e-10)
    t0 = time()
    energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=chi, cutoff=1e-12,
                       observer=obs, outputlevel=0)
    wall = time() - t0
    return wall, energy, length(ITensorMPS.energies(obs))
end

function main()
    N   = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 64
    U   = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 4.0
    chi = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 128
    nth = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1
    BLAS.set_num_threads(nth)
    run_dmrg_hubbard(8, U, 32; nsweeps=3)            # JIT warm-up (compiles everything)
    wall, energy, sweeps = run_dmrg_hubbard(N, U, chi)
    println("RESULT $wall $energy $sweeps")
end

main()
