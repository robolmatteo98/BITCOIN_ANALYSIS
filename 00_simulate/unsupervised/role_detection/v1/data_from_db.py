"""Caricamento dal database dei dati necessari alla role detection v1."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import find_dotenv, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[4]
REPORTS_ROOT = PROJECT_ROOT / "07_reports"
WALLET_REPORTS_ROOT = REPORTS_ROOT / "wallet_clustering" / "entity_clustering"
PAGERANK_REPORTS_ROOT = REPORTS_ROOT / "pre-processing" / "pagerank"
WALLET_MANIFEST_PATH = WALLET_REPORTS_ROOT / "manifest.json"


def _database_connection(database: str | None = None):
    load_dotenv(find_dotenv() or PROJECT_ROOT / ".env")
    required = ("DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"variabili mancanti nel .env: {', '.join(missing)}")

    selected_database = database or os.getenv("DB_NAME")
    if not selected_database:
        raise RuntimeError("specificare database oppure configurare DB_NAME")

    return psycopg2.connect(
        database=selected_database,
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )


def _derived_file(reports_root: Path, category: str, filename: str, split: str | None) -> Path:
    """Restituisce il file derivato della partizione richiesta."""
    if split is None:
        return reports_root / "pre-processing" / category / filename
    return reports_root / "pre-processing" / category / split / filename


def _attach_derived_features(
    flows: pd.DataFrame,
    address_features: pd.DataFrame,
    split: str | None = None,
    include_wallet_features: bool = True,
    reports_root: Path = REPORTS_ROOT,
) -> pd.DataFrame:
    """Unisce feature SQL e preprocessing esterni agli address del campione.

    Il grafo v1 resta a livello address. Per questo il PageRank calcolato sui
    wallet viene riportato a ogni address tramite il mapping address -> wallet.
    L'id del wallet non viene usato come numero: è un identificativo nominale,
    non una misura ordinabile. Usiamo invece dimensione/confidenza del cluster.
    """
    coinjoin_path = _derived_file(
        reports_root, "coinjoin", "coinjoin_scores.csv", split
    )
    geolocation_path = _derived_file(
        reports_root, "geolocation", "address_geolocation.csv", split
    )
    missing = [
        str(path)
        for path in (coinjoin_path, geolocation_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "mancano i file di preprocessing richiesti: " + ", ".join(missing)
        )

    enriched = address_features.copy()
    enriched["address"] = enriched["address"].astype(str)

    # Il CoinJoin è calcolato per transazione: lo riportiamo sugli address
    # coinvolti nei flussi e poi lo aggreghiamo per address.
    coinjoin = pd.read_csv(coinjoin_path)
    coinjoin["transaction_id"] = pd.to_numeric(
        coinjoin["transaction_id"], errors="coerce"
    )
    coinjoin = coinjoin.dropna(subset=["transaction_id"])
    coinjoin["transaction_id"] = coinjoin["transaction_id"].astype(int)
    coinjoin["coinjoin_score"] = pd.to_numeric(
        coinjoin["coinjoin_score"], errors="coerce"
    ).fillna(0.0)
    coinjoin["candidate"] = (
        coinjoin["candidate"].astype(str).str.lower() == "true"
    )

    flow_addresses = pd.concat(
        [
            flows[["transaction_id", "from_address"]].rename(
                columns={"from_address": "address"}
            ),
            flows[["transaction_id", "to_address"]].rename(
                columns={"to_address": "address"}
            ),
        ],
        ignore_index=True,
    ).dropna(subset=["address"])
    flow_addresses["address"] = flow_addresses["address"].astype(str)
    address_coinjoin = flow_addresses.merge(
        coinjoin[["transaction_id", "coinjoin_score", "candidate"]],
        on="transaction_id", how="inner",
    )
    if address_coinjoin.empty:
        coinjoin_features = pd.DataFrame(
            columns=[
                "address", "coinjoin_score_sum", "coinjoin_score_avg",
                "coinjoin_candidate_count",
            ]
        )
    else:
        coinjoin_features = address_coinjoin.groupby("address", as_index=False).agg(
            coinjoin_score_sum=("coinjoin_score", "sum"),
            coinjoin_score_avg=("coinjoin_score", "mean"),
            coinjoin_candidate_count=("candidate", "sum"),
        )

    # La geolocation è già calcolata per address; manteniamo solo le colonne
    # numeriche utilizzabili dall'encoder.
    geolocation = pd.read_csv(geolocation_path)
    geolocation_features = geolocation[
        ["address", "estimated_utc_offset", "activity_score"]
    ].copy()
    geolocation_features["address"] = geolocation_features["address"].astype(str)
    geolocation_features = geolocation_features.groupby(
        "address", as_index=False
    ).mean(numeric_only=True)
    geolocation_features = geolocation_features.rename(
        columns={
            "estimated_utc_offset": "geolocation_utc_offset",
            "activity_score": "geolocation_activity_score",
        }
    )

    if not include_wallet_features:
        return (
            enriched
            .merge(coinjoin_features, on="address", how="left")
            .merge(geolocation_features, on="address", how="left")
            .fillna(
                {
                    "coinjoin_score_sum": 0.0,
                    "coinjoin_score_avg": 0.0,
                    "coinjoin_candidate_count": 0.0,
                    "geolocation_utc_offset": 0.0,
                    "geolocation_activity_score": 0.0,
                }
            )
        )

    # Il wallet clustering fornisce un mapping per split. Oltre alla confidenza
    # del mapping, calcoliamo la dimensione del wallet: è una caratteristica
    # numerica interpretabile e non dipende dal valore arbitrario del wallet_id.
    wallet_path = WALLET_REPORTS_ROOT / (split or "analysis") / "address_to_wallet.csv"
    if not wallet_path.exists():
        raise FileNotFoundError(f"manca il mapping wallet: {wallet_path}")
    wallet_mapping = pd.read_csv(wallet_path)
    wallet_mapping["address"] = wallet_mapping["address"].astype(str)
    wallet_mapping["wallet_id"] = pd.to_numeric(
        wallet_mapping["wallet_id"], errors="coerce"
    )
    if "confidence" not in wallet_mapping:
        # Alcuni report storici del clustering non riportano la confidenza:
        # l'assegnazione è comunque esplicita, quindi la trattiamo come 1.
        wallet_mapping["confidence"] = 1.0
    else:
        wallet_mapping["confidence"] = pd.to_numeric(
            wallet_mapping["confidence"], errors="coerce"
        ).fillna(0.0)
    wallet_mapping = wallet_mapping.dropna(subset=["wallet_id"])
    wallet_sizes = wallet_mapping.groupby("wallet_id").size().rename(
        "wallet_cluster_size"
    )
    wallet_features = wallet_mapping[["address", "wallet_id", "confidence"]].copy()
    wallet_features = wallet_features.merge(
        wallet_sizes, on="wallet_id", how="left"
    ).rename(columns={"confidence": "wallet_confidence"})
    wallet_features["wallet_is_singleton"] = (
        wallet_features["wallet_cluster_size"] == 1
    ).astype(float)

    # Il file unificato del PageRank contiene i tre algoritmi. Vengono prima
    # letti i punteggi wallet e poi, se presenti, quelli address. In questo
    # modo la pipeline può usare entrambe le granularità senza cambiare codice.
    def read_pagerank(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame(columns=[
                "node", "pagerank_basic", "pagerank_combined", "pagerank_paper"
            ])
        values = pd.read_csv(path)
        for column in ("pagerank_basic", "pagerank_combined", "pagerank_paper"):
            values[column] = pd.to_numeric(values.get(column, 0.0), errors="coerce").fillna(0.0)
        return values[["node", "pagerank_basic", "pagerank_combined", "pagerank_paper"]]

    wallet_pagerank = read_pagerank(
        PAGERANK_REPORTS_ROOT / (split or "analysis") / "pagerank_scores.csv"
    ).rename(columns={
        "pagerank_basic": "wallet_pagerank_basic",
        "pagerank_combined": "wallet_pagerank_combined",
        "pagerank_paper": "wallet_pagerank_paper",
    })
    if not wallet_pagerank.empty:
        wallet_pagerank["wallet_id"] = wallet_pagerank["node"].str.replace(
            "wallet_", "", regex=False
        )
        wallet_pagerank["wallet_id"] = pd.to_numeric(
            wallet_pagerank["wallet_id"], errors="coerce"
        )
        wallet_pagerank = wallet_pagerank.dropna(subset=["wallet_id"]).drop(
            columns=["node"]
        )
        wallet_features = wallet_features.merge(
            wallet_pagerank, on="wallet_id", how="left"
        )

    address_pagerank = read_pagerank(
        PAGERANK_REPORTS_ROOT / "address" / (split or "analysis") / "pagerank_scores.csv"
    ).rename(columns={
        "pagerank_basic": "address_pagerank_basic",
        "pagerank_combined": "address_pagerank_combined",
        "pagerank_paper": "address_pagerank_paper",
        "node": "address",
    })
    if not address_pagerank.empty:
        address_pagerank["address"] = address_pagerank["address"].astype(str)

    enriched = (
        enriched
        .merge(wallet_features, on="address", how="left")
        .merge(address_pagerank, on="address", how="left")
    )

    return (
        enriched
        .merge(coinjoin_features, on="address", how="left")
        .merge(geolocation_features, on="address", how="left")
        .infer_objects(copy=False)
        .fillna(
            {
                "coinjoin_score_sum": 0.0,
                "coinjoin_score_avg": 0.0,
                "coinjoin_candidate_count": 0.0,
                "geolocation_utc_offset": 0.0,
                "geolocation_activity_score": 0.0,
                "wallet_cluster_size": 0.0,
                "wallet_confidence": 0.0,
                "wallet_is_singleton": 0.0,
                "wallet_pagerank_basic": 0.0,
                "wallet_pagerank_combined": 0.0,
                "wallet_pagerank_paper": 0.0,
                "address_pagerank_basic": 0.0,
                "address_pagerank_combined": 0.0,
                "address_pagerank_paper": 0.0,
            }
        )
    )


def load_wallet_manifest() -> dict[str, object]:
    """Carica e valida la finestra prodotta dal wallet clustering."""
    if not WALLET_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "manca il manifest del wallet clustering: "
            f"{WALLET_MANIFEST_PATH}. Eseguire prima wallet_clustering/main.py."
        )
    manifest = json.loads(WALLET_MANIFEST_PATH.read_text(encoding="utf-8"))
    transaction_ids = manifest.get("transaction_ids")
    history_ids = manifest.get("history_transaction_ids")
    analysis_ids = manifest.get("analysis_transaction_ids")
    if not isinstance(transaction_ids, list) or not isinstance(history_ids, list) or not isinstance(analysis_ids, list):
        raise ValueError("manifest wallet non valido: mancano gli elenchi di transaction_id")
    if len(transaction_ids) != len(history_ids) + len(analysis_ids):
        raise ValueError("manifest wallet non valido: history e analysis non ricostruiscono la finestra")
    return manifest


def load_v1_dataset(
    database: str | None = None,
    limit: int = 1000,
    history_limit: int = 0,
    include_features: bool = True,
    include_wallet_features: bool = True,
    transaction_ids: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carica il dataset della v1 usando una finestra coerente.

    Quando ``transaction_ids`` è valorizzato, la selezione arriva dal manifest
    del wallet clustering e non viene ricostruita da una seconda query SQL.
    """
    if limit < 1 or history_limit < 0 or history_limit > limit:
        raise ValueError(
            "limit deve essere positivo e history_limit deve stare tra 0 e limit"
        )

    if transaction_ids is not None:
        transaction_ids = [int(value) for value in transaction_ids]
        if len(transaction_ids) != limit:
            raise ValueError(
                "il manifest contiene un numero di transazioni diverso da --limit"
            )

    if transaction_ids is None:
        selected_transactions_sql = """
            SELECT DISTINCT t.id AS transaction_id
            FROM transaction t
            JOIN tx_output output
              ON output.fk_transaction_id = t.id
            JOIN address output_address
              ON output_address.id = output.fk_address_id
            WHERE t.n_inputs = 0
               OR EXISTS (
                    SELECT 1
                    FROM tx_input input_row
                    JOIN tx_output previous_output
                      ON previous_output.fk_transaction_id = input_row.prev_transaction_id
                     AND previous_output.n = input_row.prev_vout
                    JOIN address input_address
                      ON input_address.id = previous_output.fk_address_id
                    WHERE input_row.fk_transaction_id = t.id
               )
            ORDER BY t.id
            LIMIT %s
        """
        selected_transactions_params = (limit,)
    else:
        # Il manifest è la sorgente di verità: gli ID sono già stati scelti
        # dalla pipeline wallet sulla stessa finestra history/analysis.
        selected_transactions_sql = "SELECT unnest(%s::bigint[]) AS transaction_id"
        selected_transactions_params = (transaction_ids,)

    # Seleziono contesto + finestra in ordine di transaction.id. La selezione
    # include anche le coinbase, purché abbiano almeno un output decodificato.
    # Nessun dato successivo al cutoff entra nel DB.
    flows_query = f"""
        WITH selected_transactions AS ({selected_transactions_sql})
        SELECT f.from_address, f.to_address, f.flow_amount, f.time,
               t.id AS transaction_id,
               (t.n_inputs = 0) AS is_coinbase
        FROM flows_view f
        JOIN transaction t
          ON t.id = f.txid
        JOIN selected_transactions s
          ON t.id = s.transaction_id
        WHERE f.to_address IS NOT NULL
        ORDER BY t.id, f.from_address, f.to_address
    """

    with _database_connection(database) as connection:
        with connection.cursor() as cursor:
            cursor.execute(flows_query, selected_transactions_params)
            flow_values = cursor.fetchall()
            flow_columns = [description[0] for description in cursor.description]

        flows = pd.DataFrame(flow_values, columns=flow_columns)
        if flows.empty:
            return flows, pd.DataFrame()
        if not include_features:
            return flows, pd.DataFrame()

        # Le feature vengono calcolate dalla funzione SQL sullo stesso insieme di txid, evitando di leggere la address_feature_summary globale.
        tx_ids = flows["transaction_id"].dropna().astype(int).unique().tolist()
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM address_feature_summary_for_transactions(%s::bigint[])
                """,
                (tx_ids,),
            )
            feature_values = cursor.fetchall()
            feature_columns = [description[0] for description in cursor.description]

    # flows = una riga per ogni flusso address → address
    # features = una riga per ogni address
    features = pd.DataFrame(feature_values, columns=feature_columns)
    return flows, _attach_derived_features(
        flows, features, include_wallet_features=include_wallet_features
    )


def load_scoped_features(
    database: str | None,
    transaction_ids: list[int],
    flows: pd.DataFrame | None = None,
    split: str | None = None,
    include_wallet_features: bool = True,
) -> pd.DataFrame:
    """Carica le feature SQL limitandole a un insieme di transazioni."""
    if not transaction_ids:
        return pd.DataFrame()

    with _database_connection(database) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM address_feature_summary_for_transactions(%s::bigint[])
                """,
                (transaction_ids,),
            )
            feature_values = cursor.fetchall()
            feature_columns = [description[0] for description in cursor.description]

    features = pd.DataFrame(feature_values, columns=feature_columns)
    if flows is None:
        raise ValueError("flows è richiesto per aggregare le feature CoinJoin per address")
    return _attach_derived_features(
        flows,
        features,
        split=split,
        include_wallet_features=include_wallet_features,
    )


def aggregate_wallet_graph_inputs(
    flows: pd.DataFrame,
    address_features: pd.DataFrame,
    split: str = "analysis",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggrega flussi e feature address in un grafo wallet -> wallet.

    La funzione mantiene separati tre passaggi: risoluzione del mapping,
    aggregazione dei flussi e aggregazione delle feature. I trasferimenti
    interni vengono esclusi dagli archi, ma diventano feature del wallet.
    """
    mapping_path = WALLET_REPORTS_ROOT / split / "address_to_wallet.csv"
    if not mapping_path.exists():
        raise FileNotFoundError(f"manca il mapping wallet: {mapping_path}")

    mapping = pd.read_csv(mapping_path)
    mapping["address"] = mapping["address"].astype(str)
    mapping["wallet_id"] = pd.to_numeric(mapping["wallet_id"], errors="coerce")
    mapping = mapping.dropna(subset=["wallet_id"])
    address_to_wallet = {
        row.address: f"wallet_{int(row.wallet_id)}"
        for row in mapping.itertuples(index=False)
    }

    def wallet_for(address: object) -> str | None:
        if pd.isna(address):
            return None
        address = str(address)
        # Unmapped address: singleton per address, così il nodo resta distinto.
        return address_to_wallet.get(address, f"wallet_singleton_{address}")

    wallet_flows = flows.copy()
    wallet_flows["from_address"] = wallet_flows["from_address"].map(wallet_for)
    wallet_flows["to_address"] = wallet_flows["to_address"].map(wallet_for)
    wallet_flows = wallet_flows.dropna(subset=["from_address", "to_address"])

    # Prima sommiamo gli output della stessa transazione e coppia wallet.
    # Questo evita di moltiplicare artificialmente l'importo quando una
    # transazione contiene più input appartenenti allo stesso wallet.
    transaction_group = ["transaction_id", "from_address", "to_address"]
    transaction_aggregation = {"flow_amount": "sum"}
    if "time" in wallet_flows.columns:
        transaction_aggregation["time"] = "min"
    if "is_coinbase" in wallet_flows.columns:
        transaction_aggregation["is_coinbase"] = "max"
    wallet_flows = wallet_flows.groupby(
        transaction_group, as_index=False, dropna=False
    ).agg(transaction_aggregation)

    # Gli archi interni descrivono attività del wallet, non relazioni tra
    # wallet. Li misuriamo prima di rimuoverli dal grafo esterno.
    internal_mask = wallet_flows["from_address"] == wallet_flows["to_address"]
    internal_flows = wallet_flows[internal_mask].copy()
    external_flows = wallet_flows[~internal_mask].copy()

    # Ora accorpiamo tutte le transazioni dello stesso wallet-pair.
    pair_aggregation = {"flow_amount": "sum"}
    if "transaction_id" in external_flows.columns:
        pair_aggregation["transaction_id"] = "nunique"
    if "time" in external_flows.columns:
        pair_aggregation["time"] = "min"
    if "is_coinbase" in external_flows.columns:
        pair_aggregation["is_coinbase"] = "max"
    wallet_flows = external_flows.groupby(
        ["from_address", "to_address"], as_index=False, dropna=False
    ).agg(pair_aggregation)
    if "transaction_id" in wallet_flows.columns:
        wallet_flows = wallet_flows.rename(columns={"transaction_id": "transaction_count"})

    # Le feature SQL sono address-level. Per ogni wallet sommiamo conteggi e
    # importi, prendiamo min/max per il periodo di attività e la media per le
    # misure già normalizzate o condivise dal wallet.
    scoped = address_features.copy()
    scoped["address"] = scoped["address"].astype(str)
    scoped["wallet_node"] = scoped["address"].map(wallet_for)
    scoped = scoped.dropna(subset=["wallet_node"])

    sum_columns = {
        "received_output_count", "coinbase_received_count",
        "coinbase_received_amount", "spent_output_count", "unspent_output_count",
        "total_received", "total_sent", "net_flow", "estimated_balance",
        "activity_count", "active_days",
        "coinjoin_score_sum", "coinjoin_candidate_count",
    }
    min_columns = {"first_seen"}
    max_columns = {"last_seen"}
    ignored_columns = {
        "address", "wallet_node", "wallet_id",
        "wallet_cluster_size", "wallet_confidence", "wallet_is_singleton",
        "wallet_internal_transaction_count", "wallet_internal_amount",
        "wallet_internal_activity_ratio",
    }
    numeric_columns = [
        column for column in scoped.columns
        if column not in ignored_columns
    ]

    aggregate_spec = {}
    for column in numeric_columns:
        scoped[column] = pd.to_numeric(scoped[column], errors="coerce")
        if column in sum_columns:
            aggregate_spec[column] = "sum"
        elif column in min_columns:
            aggregate_spec[column] = "min"
        elif column in max_columns:
            aggregate_spec[column] = "max"
        else:
            aggregate_spec[column] = "mean"

    wallet_features = scoped.groupby("wallet_node", as_index=False).agg(aggregate_spec)
    wallet_features = wallet_features.rename(columns={"wallet_node": "address"})

    # La durata di attività del wallet è la durata dell'unione temporale dei
    # suoi address, non la somma delle durate dei singoli address.
    if {"first_seen", "last_seen"}.issubset(wallet_features.columns):
        wallet_features["active_span"] = (
            wallet_features["last_seen"] - wallet_features["first_seen"]
        ).clip(lower=0.0)

    # La dimensione viene calcolata sugli address effettivamente presenti nel
    # campione. Così anche i singleton creati come fallback vengono contati.
    mapping_confidence = mapping[["address", "wallet_id"]].copy()
    if "confidence" in mapping:
        mapping_confidence["confidence"] = pd.to_numeric(
            mapping["confidence"], errors="coerce"
        ).fillna(0.0)
    else:
        mapping_confidence["confidence"] = 1.0
    mapping_confidence["wallet_node"] = mapping_confidence["wallet_id"].map(
        lambda value: f"wallet_{int(value)}"
    )
    scoped = scoped.merge(
        mapping_confidence[["address", "confidence"]],
        on="address", how="left",
    )
    scoped["confidence"] = scoped["confidence"].fillna(0.0)
    wallet_sizes = scoped.groupby("wallet_node")["address"].nunique()
    wallet_confidences = scoped.groupby("wallet_node")["confidence"].mean()
    wallet_features["wallet_cluster_size"] = wallet_features["address"].map(wallet_sizes).fillna(1.0)
    wallet_features["wallet_confidence"] = wallet_features["address"].map(wallet_confidences).fillna(0.0)
    wallet_features["wallet_is_singleton"] = (
        wallet_features["wallet_cluster_size"] == 1
    ).astype(float)

    # Le feature interne vengono aggiunte dopo l'aggregazione address-level.
    # Un wallet senza trasferimenti interni riceve zero esplicito.
    if internal_flows.empty:
        internal_features = pd.DataFrame(
            columns=["address", "wallet_internal_transaction_count", "wallet_internal_amount"]
        )
    else:
        internal_features = internal_flows.groupby(
            "from_address", as_index=False
        ).agg(
            wallet_internal_transaction_count=("transaction_id", "nunique"),
            wallet_internal_amount=("flow_amount", "sum"),
        ).rename(columns={"from_address": "address"})
    wallet_features = wallet_features.merge(internal_features, on="address", how="left")
    wallet_features["wallet_internal_transaction_count"] = wallet_features[
        "wallet_internal_transaction_count"
    ].fillna(0.0)
    wallet_features["wallet_internal_amount"] = wallet_features[
        "wallet_internal_amount"
    ].fillna(0.0)
    activity_total = wallet_features.get("activity_count", pd.Series(0.0, index=wallet_features.index))
    wallet_features["wallet_internal_activity_ratio"] = (
        wallet_features["wallet_internal_transaction_count"]
        / activity_total.clip(lower=1.0)
    )
    return wallet_flows, wallet_features


def summarize_wallet_graph(
    wallet_flows: pd.DataFrame,
    wallet_features: pd.DataFrame,
) -> dict[str, int | float]:
    """Crea una diagnostica semplice del grafo wallet aggregato."""
    internal_transactions = wallet_features.get(
        "wallet_internal_transaction_count", pd.Series(dtype=float)
    )
    internal_amount = wallet_features.get(
        "wallet_internal_amount", pd.Series(dtype=float)
    )
    singleton_count = wallet_features.get(
        "wallet_is_singleton", pd.Series(dtype=float)
    )

    # PostgreSQL può restituire gli importi come Decimal. Convertiamo tutti i
    # valori prima della somma, così la diagnostica resta indipendente dal tipo
    # numerico usato dal driver del database.
    internal_amount = pd.to_numeric(internal_amount, errors="coerce").fillna(0.0)
    return {
        "wallet_node_count": int(len(wallet_features)),
        "external_edge_count": int(len(wallet_flows)),
        "internal_transaction_count": int(internal_transactions.sum()),
        "internal_amount": float(internal_amount.sum()),
        "singleton_wallet_count": int(singleton_count.sum()),
        "multi_address_wallet_count": int(
            (wallet_features.get("wallet_cluster_size", pd.Series(dtype=float)) > 1).sum()
        ),
    }
