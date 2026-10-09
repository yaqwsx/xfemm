from pathlib import Path

import numpy as np
import pytest

from xfemm import FemmSession, SessionStateError, XfemmError


MODEL = Path(__file__).parents[2] / "cfemm" / "fsolver" / "test" / "Temp.fem"
UNIFORM_MODEL = Path(__file__).parent / "data" / "uniform.fem"


def test_session_lifecycle_and_numpy_results():
    with FemmSession(MODEL) as session:
        count = session.mesh()
        assert count > 0
        status = session.solve()
        assert status["success"] is True
        assert status["elementCount"] == count

        result = session.result()
        assert result["A"].shape == (status["nodeCount"],)
        assert result["x"].dtype == np.float64
        session.accept()

    with pytest.raises(SessionStateError, match="closed"):
        session.mesh()


def test_result_requires_solve():
    with FemmSession(MODEL) as session:
        with pytest.raises(SessionStateError, match="solve first"):
            session.result()


def test_invalid_material_curve_is_reported_at_load(tmp_path):
    model = tmp_path / "invalid-curve.fem"
    text = UNIFORM_MODEL.read_text()
    text = text.replace("<BHPoints> = 0", "<BHPoints> = 1\n0 0", 1)
    model.write_text(text)

    with pytest.raises(XfemmError, match="zero points or at least two"):
        FemmSession(model)


def test_uniform_field_solve_is_in_memory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with FemmSession(UNIFORM_MODEL) as session:
        status = session.solve()
        assert status["nodeCount"] == session.result()["A"].size
        x = np.array([0.25, 0.5, 0.75])
        np.testing.assert_allclose(session.geta(x, 0.5).real, 0.25 * x, atol=1e-7)
        np.testing.assert_allclose(session.getb([[0.5, 0.5]]).real,
                                   [[0.0, -0.25]], atol=1e-7)

    # Solving must not use a hidden legacy solution round trip.
    assert list(tmp_path.iterdir()) == []


MAGNETS_COIL = Path(__file__).parent / "data" / "magnets_coil.fem"


def test_force_per_selection_and_circuit_props():
    # Two repelling NdFeB magnets (groups 1, 2) and a one-turn copper coil
    # "coil" (groups 3, 4: go/return sides, 8 x 5 mm, depth 100 mm) at 2 A.
    with FemmSession(MAGNETS_COIL) as session:
        session.solve()
        session.groupselectblock(1, clearselected=True)
        f1 = session.blockintegral(18).real
        session.groupselectblock(2, clearselected=True)
        f2 = session.blockintegral(18).real
        assert f1 < -10 and f2 > 10           # mask rebuilt per selection
        assert abs(f1 + f2) < 0.02 * abs(f1)

        current, voltage, flux = session.getcircuitprops("coil")
        assert current.real == pytest.approx(2.0)  # user circuit, not Amps*Turns
        # one-turn (solid) conductor: R = 2 * l / (sigma * A)
        resistance = 2 * 0.1 / (58e6 * 40e-6)
        assert voltage.real == pytest.approx(2.0 * resistance, rel=1e-6)
        session.groupselectblock([3, 4], clearselected=True)
        assert flux.real > 0
        assert flux.real == pytest.approx(session.blockintegral(0).real / 2.0, rel=1e-9)


def test_stress_tensor_rejects_selection_touching_non_air():
    # NdFeB magnet (group 1) glued to a steel block (group 2), periodic box.
    with FemmSession(Path(__file__).parent / "data" / "magnet_on_steel.fem") as session:
        session.solve()
        session.groupselectblock(1, clearselected=True)
        with pytest.raises(RuntimeError, match="touches a non-air region"):
            session.blockintegral(18)
        session.groupselectblock([1, 2], clearselected=True)
        assert abs(session.blockintegral(18).real) < 0.5  # whole body: ~0 (mesh noise)
