
import sqlite3
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///instance/visionguard.db"
)

if not DATABASE_URL.startswith("sqlite:///"):
    raise ValueError("Expected a SQLite database URL")

DB_PATH = Path(
    DATABASE_URL.removeprefix("sqlite:///")
)

if not DB_PATH.exists():
    raise FileNotFoundError(
        f"Database not found: {DB_PATH}"
    )

print("========== VISIONGUARD DATABASE ==========")
print("Database:", DB_PATH.resolve())

with sqlite3.connect(DB_PATH) as connection:

    # Check table structure
    structure = connection.execute(
        "PRAGMA table_info(predictions)"
    ).fetchall()

    print("\nDatabase columns:")

    for column in structure:
        print(column[1], "-", column[2])

    # Count predictions
    count = connection.execute(
        "SELECT COUNT(*) FROM predictions"
    ).fetchone()[0]

    print("\nTotal saved predictions:", count)

    # Display recent predictions
    records = connection.execute("""
        SELECT
            id,
            filename,
            prediction,
            confidence,
            model_name,
            created_at
        FROM predictions
        ORDER BY id DESC
        LIMIT 10
    """).fetchall()

    print("\nRecent predictions:")

    for record in records:
        print("-" * 45)
        print("ID:", record[0])
        print("Filename:", record[1])
        print("Prediction:", record[2])
        print("Confidence:", record[3], "%")
        print("Model:", record[4])
        print("Timestamp:", record[5])

print("\nDatabase verification completed!")
