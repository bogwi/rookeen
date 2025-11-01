from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def fixture_texts() -> dict[str, str]:
    data_dir = ROOT / "tests" / "test_data"
    return {
        "en_short": (data_dir / "en_short.txt").read_text(encoding="utf-8"),
        "en_medium": (data_dir / "en_medium.txt").read_text(encoding="utf-8"),
        "de_short": (data_dir / "de_short.txt").read_text(encoding="utf-8"),
        "mixed_short": (data_dir / "mixed_short.txt").read_text(encoding="utf-8"),
    }
