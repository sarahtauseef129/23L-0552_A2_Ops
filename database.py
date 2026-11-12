
import os
from pathlib import Path
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Float,
    Text,
    select,
    text
)
from sqlalchemy.engine import URL

load_dotenv()

# PostgreSQL is selected inside Docker Compose.
# SQLite remains available for local development.
if os.getenv("DB_HOST"):
    database_url = URL.create(
        drivername="postgresql+psycopg",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["DB_HOST"],
        port=5432,
        database=os.environ["POSTGRES_DB"]
    )
else:
    database_url = os.getenv(
        "DATABASE_URL",
        "sqlite:///instance/visionguard.db"
    )

    if not database_url.startswith("sqlite:///"):
        raise ValueError("Expected a SQLite database URL")

    db_path = Path(database_url.removeprefix("sqlite:///"))
    db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    database_url,
    pool_pre_ping=True
)

metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("filename", Text, nullable=False),
    Column("prediction", String(30), nullable=False),
    Column("confidence", Float, nullable=False),
    Column("ai_probability", Float, nullable=False),
    Column("model_name", String(100), nullable=False),
    Column("created_at", Text, nullable=False)
)


def init_db():
    metadata.create_all(engine)


def save_prediction(filename, result):
    with engine.begin() as connection:
        connection.execute(
            predictions.insert().values(
                filename=filename,
                prediction=result["label"],
                confidence=result["confidence"],
                ai_probability=result["ai_probability"],
                model_name=result["model_name"],
                created_at=datetime.now(
                    timezone.utc
                ).isoformat()
            )
        )


def get_history():
    with engine.connect() as connection:
        rows = connection.execute(
            select(predictions).order_by(
                predictions.c.id.desc()
            )
        ).mappings().all()

    return [dict(row) for row in rows]


def check_database():
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
    return True
