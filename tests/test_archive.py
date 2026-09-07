import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from kasner_scattering.propagation.contracts import EvolutionResult, PacketSpec, MatchingData
from kasner_scattering.propagation.inputs import ROOT
from kasner_scattering.propagation import io
from kasner_scattering.verification import digest, manifest_check, response_checks


def test_missing_platform_is_not_a_scattering_measurement():
    result = EvolutionResult("FINITE_TIME_LOCAL_RESPONSE_NO_PLATFORM_CLAIM", "two_Killing", {}).to_dict()
    assert result["p_plus"] is None and result["scattering_error"] is None
    assert result["nonlinear_horizon_embedding"] is False
    failed = EvolutionResult("NUMERICAL_FAILURE", "two_Killing", {"failure": True}).to_dict()
    assert failed["p_plus"] is None
    with pytest.raises(ValueError):
        PacketSpec(float("nan"))


def test_output_cannot_target_frozen_evidence():
    for path in (ROOT / "data", ROOT / "data/new", ROOT / "build", ROOT.parent / "other"):
        with pytest.raises(ValueError):
            io.configure(path)
    with pytest.raises(ValueError):
        io.save(ROOT / "data", "unexpected", {}, {})


def test_matching_interface_roundtrip():
    records = json.loads((ROOT / "data/inputs/matching_data.json").read_text())["records"]
    for record in records:
        value = MatchingData(**record)
        assert not value.full_horizon_second_order_ready
        assert value.source_hashes == {}


def test_manifest_rejects_modified_bytes(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text('{"value":1}\n')
    manifest = {"files": [{"path": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}]}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    assert manifest_check(tmp_path) == 1
    path.write_text('{"value":2}\n')
    with pytest.raises(ValueError, match="integrity mismatch"):
        manifest_check(tmp_path)


def test_zero_packet_field_subspace_is_invariant():
    from kasner_scattering.propagation.response_model import background_initial, reference_record, response_rhs
    bg, q = background_initial(reference_record())
    _, response = response_rhs(bg, np.zeros((28, 7)), .4, q)
    assert np.array_equal(response, np.zeros((28, 7)))


def test_displayed_observables_are_reconstructible():
    rows = response_checks()
    assert len(rows) == 5


def test_quadratic_comparison_rejects_modified_values(tmp_path):
    folder = tmp_path / "data/response"
    shutil.copytree(ROOT / "data/response/reference", folder / "reference")
    path = folder / "reference/weak.npz"
    with np.load(path) as source:
        arrays = dict(source)
    arrays["coherent"][0] += 1e-10
    np.savez_compressed(path, **arrays)
    with pytest.raises(AssertionError):
        response_checks(tmp_path)


def test_reported_residual_ceiling_is_not_relaxed(tmp_path):
    from kasner_scattering.comparison import directed_diagnostics
    for label in ("weak", "strong"):
        shutil.copyfile(ROOT / f"data/native/directed/independent_{label}_N512.json",
                        tmp_path / f"independent_{label}_N512.json")
    assert directed_diagnostics(tmp_path)["status"] == "PASS"
    path = tmp_path / "independent_weak_N512.json"
    record = json.loads(path.read_text())
    record["four_dimensional_residuals"][0]["scalar_over_expansion_squared"] = 1e-12
    path.write_text(json.dumps(record))
    assert directed_diagnostics(tmp_path)["status"] == "REVIEW_REQUIRED"
