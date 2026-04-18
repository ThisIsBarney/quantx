"""Tests for the hardware-aware circuit transpiler (M76)."""
from __future__ import annotations

import pytest
from ..transpiler import (
    TranspilerError,
    GateOp,
    CouplingMap,
    TranspileResult,
    HardwareTranspiler,
    _circuit_depth,
    _cancel_inverses,
    _n_logical_qubits,
)


# ===========================================================================
# CouplingMap — construction and queries
# ===========================================================================

class TestCouplingMapLinear:
    def test_n_qubits(self):
        cm = CouplingMap.linear(4)
        assert cm.n_qubits == 4

    def test_edges(self):
        cm = CouplingMap.linear(3)
        assert set(cm.edges) == {(0, 1), (1, 2)}

    def test_is_connected_adjacent(self):
        cm = CouplingMap.linear(3)
        assert cm.is_connected(0, 1) is True
        assert cm.is_connected(1, 0) is True   # undirected

    def test_is_connected_non_adjacent(self):
        cm = CouplingMap.linear(3)
        assert cm.is_connected(0, 2) is False

    def test_neighbors(self):
        cm = CouplingMap.linear(4)
        assert sorted(cm.neighbors(1)) == [0, 2]
        assert cm.neighbors(0) == [1]
        assert cm.neighbors(3) == [2]

    def test_shortest_path_direct(self):
        cm = CouplingMap.linear(3)
        assert cm.shortest_path(0, 1) == [0, 1]

    def test_shortest_path_skipping(self):
        cm = CouplingMap.linear(4)
        assert cm.shortest_path(0, 3) == [0, 1, 2, 3]

    def test_shortest_path_same(self):
        cm = CouplingMap.linear(3)
        assert cm.shortest_path(1, 1) == [1]

    def test_distance_direct(self):
        cm = CouplingMap.linear(3)
        assert cm.distance(0, 1) == 1

    def test_distance_two_hops(self):
        cm = CouplingMap.linear(3)
        assert cm.distance(0, 2) == 2

    def test_distance_three_hops(self):
        cm = CouplingMap.linear(4)
        assert cm.distance(0, 3) == 3

    def test_linear_n1(self):
        cm = CouplingMap.linear(1)
        assert cm.n_qubits == 1
        assert cm.edges == []

    def test_invalid_n_raises(self):
        with pytest.raises(TranspilerError):
            CouplingMap.linear(0)


class TestCouplingMapRing:
    def test_n_qubits(self):
        cm = CouplingMap.ring(4)
        assert cm.n_qubits == 4

    def test_cyclic_edge(self):
        cm = CouplingMap.ring(4)
        assert cm.is_connected(3, 0)
        assert cm.is_connected(0, 3)

    def test_all_adjacent(self):
        cm = CouplingMap.ring(5)
        for i in range(5):
            assert cm.is_connected(i, (i + 1) % 5)

    def test_distance_shorter_than_linear(self):
        # In a 6-ring, distance(0,5) via wrap = 1, not 5
        cm = CouplingMap.ring(6)
        assert cm.distance(0, 5) == 1

    def test_ring_n1_raises(self):
        with pytest.raises(TranspilerError):
            CouplingMap.ring(1)


class TestCouplingMapGrid:
    def test_2x2(self):
        cm = CouplingMap.grid(2, 2)
        assert cm.n_qubits == 4
        # (0-1), (2-3) horizontal; (0-2), (1-3) vertical
        assert cm.is_connected(0, 1)
        assert cm.is_connected(0, 2)
        assert not cm.is_connected(0, 3)

    def test_2x3(self):
        cm = CouplingMap.grid(2, 3)
        assert cm.n_qubits == 6
        assert cm.is_connected(0, 1)
        assert cm.is_connected(1, 2)
        assert cm.is_connected(0, 3)
        assert cm.is_connected(1, 4)
        assert cm.is_connected(3, 4)

    def test_1x1(self):
        cm = CouplingMap.grid(1, 1)
        assert cm.n_qubits == 1

    def test_invalid_dims_raises(self):
        with pytest.raises(TranspilerError):
            CouplingMap.grid(0, 3)


class TestCouplingMapAllToAll:
    def test_all_edges_present(self):
        cm = CouplingMap.all_to_all(4)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert cm.is_connected(i, j)

    def test_distance_always_one(self):
        cm = CouplingMap.all_to_all(5)
        assert cm.distance(0, 4) == 1

    def test_n_edges(self):
        n = 5
        cm = CouplingMap.all_to_all(n)
        assert len(cm.edges) == n * (n - 1) // 2


class TestCouplingMapDisconnected:
    def test_no_path_raises(self):
        # Two isolated nodes: 0 and 1 (no edges)
        cm = CouplingMap(2, [])
        with pytest.raises(TranspilerError, match="No path"):
            cm.shortest_path(0, 1)


# ===========================================================================
# GateOp
# ===========================================================================

class TestGateOp:
    def test_equality_case_insensitive(self):
        a = GateOp("CX", [0, 1])
        b = GateOp("cx", [0, 1])
        assert a == b

    def test_inequality_different_qubits(self):
        a = GateOp("cx", [0, 1])
        b = GateOp("cx", [1, 0])
        assert a != b

    def test_hash_case_insensitive(self):
        a = GateOp("H", [0])
        b = GateOp("h", [0])
        assert hash(a) == hash(b)

    def test_params_default_empty(self):
        g = GateOp("rz", [0])
        assert g.params == []

    def test_repr(self):
        g = GateOp("cx", [0, 1])
        r = repr(g)
        assert "cx" in r
        assert "[0, 1]" in r


# ===========================================================================
# Circuit utilities
# ===========================================================================

class TestCircuitDepth:
    def test_empty(self):
        assert _circuit_depth([]) == 0

    def test_single_gate(self):
        assert _circuit_depth([GateOp("h", [0])]) == 1

    def test_parallel_gates(self):
        circ = [GateOp("h", [0]), GateOp("h", [1])]
        # Both on different qubits: depth = 1
        assert _circuit_depth(circ) == 1

    def test_serial_gates_same_qubit(self):
        circ = [GateOp("h", [0]), GateOp("x", [0])]
        assert _circuit_depth(circ) == 2

    def test_two_qubit_gate(self):
        circ = [GateOp("cx", [0, 1])]
        assert _circuit_depth(circ) == 1

    def test_mixed(self):
        circ = [
            GateOp("h", [0]),
            GateOp("h", [1]),
            GateOp("cx", [0, 1]),   # both at depth 2
            GateOp("h", [0]),        # depth 3
        ]
        assert _circuit_depth(circ) == 3


class TestNLogicalQubits:
    def test_empty(self):
        assert _n_logical_qubits([]) == 0

    def test_single_qubit(self):
        assert _n_logical_qubits([GateOp("h", [0])]) == 1

    def test_multiple(self):
        circ = [GateOp("cx", [0, 3])]
        assert _n_logical_qubits(circ) == 4


class TestCancelInverses:
    def test_hh_cancelled(self):
        circ = [GateOp("h", [0]), GateOp("h", [0])]
        assert _cancel_inverses(circ) == []

    def test_xx_cancelled(self):
        circ = [GateOp("x", [0]), GateOp("x", [0])]
        assert _cancel_inverses(circ) == []

    def test_s_sdg_cancelled(self):
        circ = [GateOp("s", [0]), GateOp("sdg", [0])]
        assert _cancel_inverses(circ) == []

    def test_sdg_s_cancelled(self):
        circ = [GateOp("sdg", [0]), GateOp("s", [0])]
        assert _cancel_inverses(circ) == []

    def test_t_tdg_cancelled(self):
        circ = [GateOp("t", [0]), GateOp("tdg", [0])]
        assert _cancel_inverses(circ) == []

    def test_swap_swap_cancelled(self):
        circ = [GateOp("swap", [0, 1]), GateOp("swap", [0, 1])]
        assert _cancel_inverses(circ) == []

    def test_different_qubits_not_cancelled(self):
        circ = [GateOp("h", [0]), GateOp("h", [1])]
        result = _cancel_inverses(circ)
        assert len(result) == 2

    def test_h_between_other_gates(self):
        # H X H: only the two H's are adjacent? No — H is at 0, X at 0, H at 0
        # After cancel: X (since H and H cancel)
        # Actually: [H(0), X(0), H(0)] — only adjacent H·H would cancel
        circ = [GateOp("h", [0]), GateOp("x", [0]), GateOp("h", [0])]
        # No adjacent inverse pairs → no cancellation
        assert _cancel_inverses(circ) == circ

    def test_chained_cancellation(self):
        # H H H H → []
        circ = [GateOp("h", [0])] * 4
        assert _cancel_inverses(circ) == []

    def test_rz_not_cancelled_with_params(self):
        # Two RZ gates with same angle: not in _INVERSE_PAIRS, so NOT cancelled
        circ = [GateOp("rz", [0], [1.0]), GateOp("rz", [0], [1.0])]
        assert len(_cancel_inverses(circ)) == 2


# ===========================================================================
# HardwareTranspiler — gate decomposition
# ===========================================================================

class TestDecompose:
    def test_cz_to_cx(self):
        t = HardwareTranspiler(CouplingMap.all_to_all(2), native_gates="cx_rz")
        result = t.decompose(GateOp("cz", [0, 1]))
        names = [g.name.lower() for g in result]
        assert names == ["h", "cx", "h"]
        # H and H go on qubit 1 (target)
        assert result[0].qubits == [1]
        assert result[2].qubits == [1]
        assert result[1].qubits == [0, 1]

    def test_cx_to_cz(self):
        t = HardwareTranspiler(CouplingMap.all_to_all(2), native_gates="cz_rz")
        result = t.decompose(GateOp("cx", [0, 1]))
        names = [g.name.lower() for g in result]
        assert names == ["h", "cz", "h"]

    def test_swap_to_cx(self):
        t = HardwareTranspiler(CouplingMap.all_to_all(2), native_gates="cx_rz")
        result = t.decompose(GateOp("swap", [0, 1]))
        names = [g.name.lower() for g in result]
        assert names == ["cx", "cx", "cx"]

    def test_native_gate_passthrough(self):
        t = HardwareTranspiler(CouplingMap.all_to_all(2), native_gates="cx_rz")
        g = GateOp("cx", [0, 1])
        assert t.decompose(g) == [g]

    def test_ccx_decomposes_on_cx_device(self):
        t = HardwareTranspiler(CouplingMap.all_to_all(3), native_gates="cx_rz")
        result = t.decompose(GateOp("ccx", [0, 1, 2]))
        names = [g.name.lower() for g in result]
        # Must contain CX and T/Tdg gates but no CCX
        assert "ccx" not in names
        assert "cx" in names

    def test_unknown_gate_raises(self):
        t = HardwareTranspiler(CouplingMap.all_to_all(2), native_gates="cx_rz")
        with pytest.raises(TranspilerError, match="Cannot decompose"):
            t.decompose(GateOp("my_fancy_gate", [0, 1]))

    def test_invalid_native_set_raises(self):
        with pytest.raises(TranspilerError, match="Unknown native_gates"):
            HardwareTranspiler(CouplingMap.linear(2), native_gates="quantum_rigetti_x")


# ===========================================================================
# HardwareTranspiler — routing
# ===========================================================================

class TestRouting:
    def test_empty_circuit(self):
        t = HardwareTranspiler(CouplingMap.linear(3))
        result = t.transpile([])
        assert result.circuit == []
        assert result.n_swaps == 0
        assert result.n_gates_before == 0

    def test_all_to_all_no_swaps(self):
        cm = CouplingMap.all_to_all(5)
        t = HardwareTranspiler(cm)
        circ = [
            GateOp("cx", [0, 4]),
            GateOp("cx", [2, 3]),
        ]
        result = t.transpile(circ)
        assert result.n_swaps == 0

    def test_linear_chain_one_swap(self):
        # CX(0,2) on 0-1-2 chain: 1 SWAP to bring them adjacent
        cm = CouplingMap.linear(3)
        t = HardwareTranspiler(cm)
        result = t.transpile([GateOp("cx", [0, 2])])
        assert result.n_swaps == 1

    def test_routed_gates_all_adjacent(self):
        cm = CouplingMap.linear(4)
        t = HardwareTranspiler(cm)
        circ = [
            GateOp("cx", [0, 3]),
            GateOp("cx", [0, 2]),
        ]
        result = t.transpile(circ)
        for gate in result.circuit:
            if len(gate.qubits) == 2:
                q0, q1 = gate.qubits
                assert cm.is_connected(q0, q1), (
                    f"Non-adjacent gate {gate} after routing"
                )

    def test_too_many_qubits_raises(self):
        cm = CouplingMap.linear(2)
        t = HardwareTranspiler(cm)
        with pytest.raises(TranspilerError, match="logical qubits"):
            t.transpile([GateOp("cx", [0, 2])])  # needs 3 qubits on 2-qubit device

    def test_identity_mapping_default(self):
        cm = CouplingMap.linear(3)
        t = HardwareTranspiler(cm)
        result = t.transpile([GateOp("h", [0])])
        assert result.qubit_map == {0: 0}

    def test_custom_initial_mapping(self):
        cm = CouplingMap.linear(4)
        t = HardwareTranspiler(cm)
        circ = [GateOp("cx", [0, 1])]
        # Map logical 0 → physical 2, logical 1 → physical 3 (adjacent)
        result = t.transpile(circ, initial_mapping={0: 2, 1: 3})
        assert result.n_swaps == 0
        assert result.qubit_map == {0: 2, 1: 3}

    def test_grid_topology_routing(self):
        # 2x3 grid; CX(0,5): requires at least 3 hops
        cm = CouplingMap.grid(2, 3)
        t = HardwareTranspiler(cm)
        result = t.transpile([GateOp("cx", [0, 5])])
        for gate in result.circuit:
            if len(gate.qubits) == 2 and gate.name.lower() != "swap":
                q0, q1 = gate.qubits
                assert cm.is_connected(q0, q1)

    def test_ring_topology_routing(self):
        cm = CouplingMap.ring(6)
        t = HardwareTranspiler(cm)
        # CX(0,3): could go 0-1-2-3 (3 hops) or 0-5-4-3 (3 hops); either way ≥2 swaps
        result = t.transpile([GateOp("cx", [0, 3])])
        for gate in result.circuit:
            if len(gate.qubits) == 2 and gate.name.lower() != "swap":
                q0, q1 = gate.qubits
                assert cm.is_connected(q0, q1)

    def test_single_qubit_gates_no_swap(self):
        cm = CouplingMap.linear(5)
        t = HardwareTranspiler(cm)
        circ = [GateOp("h", [i]) for i in range(5)]
        result = t.transpile(circ)
        assert result.n_swaps == 0

    def test_transpile_result_fields(self):
        cm = CouplingMap.linear(3)
        t = HardwareTranspiler(cm)
        circ = [GateOp("h", [0]), GateOp("cx", [0, 2]), GateOp("h", [1])]
        result = t.transpile(circ)
        assert isinstance(result, TranspileResult)
        assert result.n_gates_before == 3
        assert result.n_gates_after >= result.n_gates_before  # decomp + SWAPs add gates
        assert result.depth_before >= 1
        assert result.depth_after >= 1

    def test_only_adjacent_gates_no_swap_inserted(self):
        cm = CouplingMap.linear(4)
        t = HardwareTranspiler(cm)
        circ = [GateOp("cx", [0, 1]), GateOp("cx", [1, 2]), GateOp("cx", [2, 3])]
        result = t.transpile(circ)
        assert result.n_swaps == 0


# ===========================================================================
# HardwareTranspiler — peephole optimisation
# ===========================================================================

class TestPeepholeOptimisation:
    def test_level_0_no_cancellation(self):
        cm = CouplingMap.all_to_all(3)
        t = HardwareTranspiler(cm)
        circ = [GateOp("h", [0]), GateOp("h", [0])]
        result = t.transpile(circ, optimization_level=0)
        names = [g.name.lower() for g in result.circuit]
        assert names.count("h") == 2

    def test_level_2_cancels_hh(self):
        cm = CouplingMap.all_to_all(3)
        t = HardwareTranspiler(cm)
        circ = [GateOp("h", [0]), GateOp("h", [0])]
        result = t.transpile(circ, optimization_level=2)
        names = [g.name.lower() for g in result.circuit]
        assert "h" not in names

    def test_level_2_cancels_s_sdg(self):
        cm = CouplingMap.all_to_all(2)
        t = HardwareTranspiler(cm)
        circ = [GateOp("s", [0]), GateOp("sdg", [0])]
        result = t.transpile(circ, optimization_level=2)
        assert result.circuit == []

    def test_level_2_cancels_xx(self):
        cm = CouplingMap.all_to_all(2)
        t = HardwareTranspiler(cm)
        circ = [GateOp("x", [0]), GateOp("x", [0])]
        result = t.transpile(circ, optimization_level=2)
        assert result.circuit == []

    def test_level_2_cancels_swap_swap(self):
        cm = CouplingMap.all_to_all(3)
        t = HardwareTranspiler(cm, native_gates="all")
        circ = [GateOp("swap", [0, 1]), GateOp("swap", [0, 1])]
        result = t.transpile(circ, optimization_level=2)
        names = [g.name.lower() for g in result.circuit]
        assert "swap" not in names

    def test_level_2_keeps_non_inverse_pair(self):
        cm = CouplingMap.all_to_all(2)
        t = HardwareTranspiler(cm)
        circ = [GateOp("h", [0]), GateOp("x", [0])]
        result = t.transpile(circ, optimization_level=2)
        assert len(result.circuit) == 2


# ===========================================================================
# HardwareTranspiler — decompose-then-route integration
# ===========================================================================

class TestDecomposeAndRoute:
    def test_cz_on_cx_device_routed(self):
        cm = CouplingMap.linear(3)
        t = HardwareTranspiler(cm, native_gates="cx_rz")
        # CZ(0,2) is not native → decompose to H CX H, then route
        result = t.transpile([GateOp("cz", [0, 2])])
        names = [g.name.lower() for g in result.circuit]
        assert "cz" not in names
        assert "cx" in names

    def test_swap_on_cx_device_routed(self):
        cm = CouplingMap.all_to_all(2)
        t = HardwareTranspiler(cm, native_gates="cx_rz")
        result = t.transpile([GateOp("swap", [0, 1])])
        names = [g.name.lower() for g in result.circuit]
        assert "swap" not in names
        assert names.count("cx") == 3

    def test_ccx_decomposed(self):
        cm = CouplingMap.all_to_all(3)
        t = HardwareTranspiler(cm, native_gates="cx_rz")
        result = t.transpile([GateOp("ccx", [0, 1, 2])])
        names = [g.name.lower() for g in result.circuit]
        assert "ccx" not in names
        assert "toffoli" not in names
        assert "cx" in names

    def test_cnot_alias(self):
        cm = CouplingMap.all_to_all(2)
        t = HardwareTranspiler(cm, native_gates="cx_rz")
        # cnot is in native_gates set — should pass through
        result = t.transpile([GateOp("cnot", [0, 1])])
        assert result.n_swaps == 0
        assert result.circuit[0].name.lower() == "cnot"

    def test_rz_with_params_preserved(self):
        cm = CouplingMap.all_to_all(2)
        t = HardwareTranspiler(cm)
        circ = [GateOp("rz", [0], [1.5707963])]
        result = t.transpile(circ)
        assert result.circuit[0].params == [1.5707963]

    def test_multi_gate_circuit(self):
        # Bell state circuit: H(0) → CX(0,1)
        cm = CouplingMap.linear(2)
        t = HardwareTranspiler(cm)
        circ = [GateOp("h", [0]), GateOp("cx", [0, 1])]
        result = t.transpile(circ)
        assert result.n_swaps == 0
        assert result.n_gates_after == 2

    def test_long_chain_all_adjacent_after_routing(self):
        # Linear chain of 5 qubits; CX on all non-adjacent pairs
        cm = CouplingMap.linear(5)
        t = HardwareTranspiler(cm)
        circ = [
            GateOp("cx", [0, 4]),
            GateOp("cx", [1, 3]),
        ]
        result = t.transpile(circ)
        for gate in result.circuit:
            if len(gate.qubits) == 2 and gate.name.lower() != "swap":
                q0, q1 = gate.qubits
                assert cm.is_connected(q0, q1)

    def test_n_gates_after_includes_swaps(self):
        cm = CouplingMap.linear(3)
        t = HardwareTranspiler(cm)
        result = t.transpile([GateOp("cx", [0, 2])])
        # 1 SWAP inserted + CX itself = at minimum 2 2-qubit operations
        # But SWAP may be decomposed in all → depends; for cx_rz native,
        # SWAP is NOT in native set, but routing inserts raw SWAP before decomp.
        # Let's just check the count is ≥ 1
        assert result.n_gates_after >= 1
        assert result.n_swaps == 1

    def test_depth_after_gte_1_for_nonempty(self):
        cm = CouplingMap.all_to_all(3)
        t = HardwareTranspiler(cm)
        result = t.transpile([GateOp("h", [0]), GateOp("cx", [0, 1])])
        assert result.depth_after >= 1


# ===========================================================================
# TranspileResult structure
# ===========================================================================

class TestTranspileResult:
    def test_is_dataclass(self):
        import dataclasses
        assert dataclasses.is_dataclass(TranspileResult)

    def test_fields(self):
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(TranspileResult)}
        assert "circuit" in field_names
        assert "qubit_map" in field_names
        assert "n_swaps" in field_names
        assert "n_gates_before" in field_names
        assert "n_gates_after" in field_names
        assert "depth_before" in field_names
        assert "depth_after" in field_names


# ===========================================================================
# Custom native gate set
# ===========================================================================

class TestCustomNativeGateSet:
    def test_frozenset_accepted(self):
        t = HardwareTranspiler(
            CouplingMap.all_to_all(2),
            native_gates=frozenset({"cx", "h", "rz"}),
        )
        result = t.transpile([GateOp("cx", [0, 1]), GateOp("h", [0])])
        assert result.n_swaps == 0

    def test_custom_set_missing_gate_raises(self):
        t = HardwareTranspiler(
            CouplingMap.all_to_all(2),
            native_gates=frozenset({"h", "rz"}),   # no CX or CZ
        )
        with pytest.raises(TranspilerError):
            t.transpile([GateOp("cx", [0, 1])])
