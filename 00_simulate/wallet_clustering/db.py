"""Funzioni di accesso al database usate dalla pipeline ufficiale."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


# La root del progetto serve per cercare il file .env anche quando lo script
# viene lanciato dalla directory principale del repository.
ROOT = Path(__file__).resolve().parents[2]


def database_parameters(database: str | None) -> dict[str, object]:
    """Legge dal .env i parametri necessari a collegarsi a PostgreSQL."""
    load_dotenv(find_dotenv() or ROOT / ".env")
    selected_database = database or os.getenv("DB_NAME")
    required = ("DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT")
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"variabili mancanti nel .env: {', '.join(missing)}")
    if not selected_database:
        raise RuntimeError("specificare --database oppure configurare DB_NAME")
    return {
        "dbname": selected_database,
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": int(os.getenv("DB_PORT")),
    }
