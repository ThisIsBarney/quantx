"""Tests for the quantum chemistry module."""
from __future__ import annotations

import math
import pytest
from quantx_engine.chemistry import (
    PauliTerm, PauliHamiltonian, jordan_wigner, MolecularVQE, ChemistryError,
)


# ── PauliTerm ─────────────────────────────────────────────────────────────────

def test_pauli_term_n_qubits():
    t = PauliTerm(coeff=1.0+0j, ops={0: "X", 2: "Z"})
    assert t.n_qubits() == 3


def test_pauli_term_empty_ops():
    t = PauliTerm(coeff=0.5+0j, ops={})
    assert t.n_qubits() == 0


def test_pauli_term_to_dict():
    t = PauliTerm(coeff=1.5+0j, ops={1: "Y"})
    d = t.to_dict()
    assert d["coeff_re"] == pytest.approx(1.5)
    assert d["coeff_im"] == pytest.approx(0.0)
    assert d["ops"] == {"1": "Y"}


# ── PauliHamiltonian ──────────────────────────────────────────────────────────

def test_hamiltonian_add_and_qubits():
    h = PauliHamiltonian()
    h.add(PauliTerm(coeff=1.0+0j, ops={0: "X", 1: "Z"}))
    h.add(PauliTerm(coeff=0.5+0j, ops={}))
    assert h.n_qubits == 2
    assert len(h.terms) == 2


def test_hamiltonian_constant():
    h = PauliHamiltonian()
    h.add(PauliTerm(coeff=complex(0.7151043), ops={}))
    h.add(PauliTerm(coeff=complex(-0.5), ops={0: "Z"}))
    assert abs(h.constant() - 0.7151043) < 1e-9


def test_hamiltonian_to_dict():
    h = PauliHamiltonian()
    h.add(PauliTerm(coeff=1.0+0j, ops={0: "X"}))
    d = h.to_dict()
    assert "n_qubits" in d
    assert "terms" in d
    assert len(d["terms"]) == 1


# ── Jordan-Wigner transform ───────────────────────────────────────────────────

def test_jw_number_operator():
    """n_p = a†_p a_p → (I - Z_p)/2"""
    ham = jordan_wigner(
        one_body=[(0, 0, 1.0)],
        two_body=[],
        n_orbitals=1,
        constant=0.0,
    )
    # Should have identity term (0.5) and Z_0 term (-0.5)
    coeffs = {
        tuple(sorted(t.ops.items())): t.coeff.real
        for t in ham.terms
    }
    assert abs(coeffs.get((), 0.0) - 0.5) < 1e-10
    assert abs(coeffs.get(((0, "Z"),), 0.0) - (-0.5)) < 1e-10


def test_jw_hopping_term():
    """a†_0 a_1 + h.c. → X0 X1 + Y0 Y1 with coefficient 1.0 each (not cancelling)."""
    ham = jordan_wigner(
        one_body=[(0, 1, 1.0)],
        two_body=[],
        n_orbitals=2,
        constant=0.0,
    )
    xx_coeff = sum(t.coeff.real for t in ham.terms if t.ops.get(0) == "X" and t.ops.get(1) == "X")
    yy_coeff = sum(t.coeff.real for t in ham.terms if t.ops.get(0) == "Y" and t.ops.get(1) == "Y")
    # Hermitian sum h_pq a†_p a_q + h_qp a†_q a_p with h=1 → coefficients of 1.0 each
    assert abs(xx_coeff - 1.0) < 1e-10, f"XX coeff should be 1.0, got {xx_coeff}"
    assert abs(yy_coeff - 1.0) < 1e-10, f"YY coeff should be 1.0, got {yy_coeff}"


def test_jw_constant_term():
    ham = jordan_wigner([], [], n_orbitals=2, constant=0.7151043)
    const = sum(t.coeff.real for t in ham.terms if not t.ops)
    assert abs(const - 0.7151043) < 1e-9


def test_jw_two_body_diagonal():
    """n_p n_q → ¼(I - Z_p - Z_q + Z_p Z_q)"""
    ham = jordan_wigner(
        one_body=[],
        two_body=[(0, 1, 1, 0, 1.0)],
        n_orbitals=2,
        constant=0.0,
    )
    # Should produce 4 terms
    assert len(ham.terms) == 4


# ── MolecularVQE ─────────────────────────────────────────────────────────────

def test_h2_default():
    m = MolecularVQE.h2()
    assert m.name == "H2"
    assert m.n_electrons == 2
    assert abs(m.bond_length - 0.735) < 1e-9
    assert m.hamiltonian.n_qubits == 4
    assert len(m.hamiltonian.terms) > 0


def test_h2_fci_interpolation():
    m = MolecularVQE.h2(bond_length=0.735)
    assert m.fci_energy is not None
    assert abs(m.fci_energy - (-1.1373)) < 0.01  # within 10 mHa of tabulated value


def test_lih_default():
    m = MolecularVQE.lih()
    assert m.name == "LiH"
    assert m.hamiltonian.n_qubits == 10
    assert m.n_electrons == 4


def test_beh2_default():
    m = MolecularVQE.beh2()
    assert m.name == "BeH2"
    assert m.hamiltonian.n_qubits == 12
    assert m.n_electrons == 4


def test_to_vqe_options_keys():
    m = MolecularVQE.h2()
    opts = m.to_vqe_options()
    required_keys = {"molecule", "bond_length", "n_qubits", "n_electrons",
                     "cost_hamiltonian", "ansatz_type", "shots", "max_iterations", "optimizer"}
    assert required_keys.issubset(opts.keys())
    assert opts["n_qubits"] == 4
    assert opts["ansatz_type"] == "uccsd"
    # cost_hamiltonian must be a list of {label, coefficient} dicts
    assert isinstance(opts["cost_hamiltonian"], list)
    assert all({"label", "coefficient"}.issubset(t.keys()) for t in opts["cost_hamiltonian"])
    assert all(isinstance(t["label"], str) for t in opts["cost_hamiltonian"])
    assert all(isinstance(t["coefficient"], float) for t in opts["cost_hamiltonian"])


def test_to_vqe_options_custom():
    m = MolecularVQE.h2()
    opts = m.to_vqe_options(ansatz="hardware_efficient", shots=512, max_iterations=100)
    assert opts["ansatz_type"] == "hardware_efficient"
    assert opts["shots"] == 512
    assert opts["max_iterations"] == 100


def test_fci_energy_boundary():
    """Bond lengths outside the tabulated range return the nearest endpoint."""
    m_short = MolecularVQE.h2(bond_length=0.1)
    m_long = MolecularVQE.h2(bond_length=10.0)
    assert m_short.fci_energy is not None
    assert m_long.fci_energy is not None


def test_hamiltonian_json_serialisable():
    import json
    for mol in [MolecularVQE.h2(), MolecularVQE.lih(), MolecularVQE.beh2()]:
        d = mol.hamiltonian.to_dict()
        json.dumps(d)  # must not raise


def test_vqe_options_json_serialisable():
    import json
    opts = MolecularVQE.h2().to_vqe_options()
    json.dumps(opts)  # must not raise
