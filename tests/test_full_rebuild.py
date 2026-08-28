from pathlib import Path

import pytest

from kasner_scattering.pipeline import rebuild


@pytest.mark.slow
def test_full_rebuild_matches_frozen_results(tmp_path: Path) -> None:
    assert rebuild(tmp_path)["status"] == "PASS"

