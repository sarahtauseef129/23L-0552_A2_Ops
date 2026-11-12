
import importlib.util
from pathlib import Path


def test_prediction_is_saved_and_retrieved(
    tmp_path, monkeypatch
):
    # Use a temporary SQLite database for this test.
    # Do not touch the real prediction history.
    db_file = tmp_path / "test_visionguard.db"

    monkeypatch.delenv("DB_HOST", raising=False)
    monkeypatch.setenv(
        "DATABASE_URL",
        f"sqlite:///{db_file.as_posix()}"
    )

    source = Path(__file__).resolve().parents[1] / "database.py"

    spec = importlib.util.spec_from_file_location(
        "visionguard_test_database",
        source
    )

    database = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(database)

    database.init_db()

    result = {
        "label": "AI-generated",
        "confidence": 85.5,
        "ai_probability": 85.5,
        "model_name": "MobileNetV2_Tuned"
    }

    database.save_prediction("sample.png", result)

    records = database.get_history()

    assert len(records) == 1
    assert records[0]["filename"] == "sample.png"
    assert records[0]["prediction"] == "AI-generated"
    assert records[0]["confidence"] == 85.5
    assert records[0]["model_name"] == "MobileNetV2_Tuned"
    assert records[0]["created_at"]

    assert database.check_database() is True

    database.engine.dispose()
