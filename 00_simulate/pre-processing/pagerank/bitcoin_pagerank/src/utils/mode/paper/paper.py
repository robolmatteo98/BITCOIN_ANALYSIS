import math

# TIME
def normalize_timestamp(timestamp, time_bounds):
    """
        Porta il timestamp grezzo nell'intervallo [0, 1].
        In questo modo la logistic lavora su valori piccoli e non va in overflow.
    """
    if timestamp is None or time_bounds is None:
        return None

    min_ts, max_ts = time_bounds
    if min_ts is None or max_ts is None:
        return None

    min_ts = float(min_ts)
    max_ts = float(max_ts)
    if max_ts <= min_ts:
        return 0.5

    return float(timestamp) / max_ts

# a più grande, premia le transazioni più recenti
def get_Tt(t, time_bounds, a=10.0):
    """
        T(t) = 1 / (1 + e^-a(t - t0))
    """

    if t is None:
        return 0.0

    normalized_t = normalize_timestamp(t, time_bounds)
    if normalized_t is None:
        return 0.0

    t0 = 0.5

    # controllo overflow
    exponent = -float(a) * (float(normalized_t) - float(t0))
    if exponent > 700:
        return 0.0
    if exponent < -700:
        return 1.0

    return 1.0 / (1.0 + math.exp(exponent))

def time_weight_paper(edge, time_bounds):
    """
        Xi = SUM_{j = 1, i != j in N} aij * ( MAX ( T(t) ))
    """
    
    transactions = edge.get("transactions") or []
    if not transactions:
        return 1.0

    numeric_values = [
        float(transaction["timestamp"])
        for transaction in transactions
        if transaction.get("timestamp") is not None
    ]
    if not numeric_values:
        return 1.0

    # siccome è una funzione crescente, applico prima il max, invece di eseguire T_t su tutti i timestamp e poi applicare il max
    MAX_T_t = get_Tt(max(numeric_values), time_bounds)
    if MAX_T_t is None:
        return 1.0

    # DUE VARIANTI:
    # 1. aij viene approssimato a 1.
    # 2. aij viene approssimato con il numero di transazioni tra i e j.
    aij = float(1.0)
    return aij * MAX_T_t

# VALUE
def get_Vz(z, b):
    """
        V(z) = 1 / 1 + e^(-bz)
    """

    z = max(float(z or 0.0), 0.0)
    return 1.0 / (1.0 + math.exp(-b * z))

def value_weight_paper(edge, time_bounds, b=0.75):
    """
        Yij = SUM_{t in H, z in Q} T(t) * V(z)
    """
    transactions = edge.get("transactions") or []
    if not transactions:
        return 0.0

    score = 0.0
    for transaction in transactions:
        # score = T(t) * V(z)
        score += get_Tt(transaction.get("timestamp"), time_bounds) * get_Vz(transaction.get("value", 0.0), b)
    return score

def compute_time_scores(edges_rdd, time_bounds):
    return (
        edges_rdd
        .map(lambda edge: (edge["src"], time_weight_paper(edge, time_bounds)))
        .reduceByKey(lambda a, b: a + b)
        .cache()
    )

def normalize_time_scores(nodes, list_Xi, SUM_Xi, num_nodes):
    """
        1a parte della formula = Xi / SUM_{i=1 a N} Xi
    """
    if SUM_Xi > 0:
        return (
            nodes
            .map(lambda node: (node, 0.0))
            .leftOuterJoin(list_Xi) # per nodi che non hanno archi uscenti, posso avere: (2, (0.0, None))
            .mapValues(lambda pair: float(pair[1] or 0.0) / float(SUM_Xi)) # (2, 0.0)
            .cache()
        )
    return nodes.map(lambda node: (node, 1.0 / num_nodes)).cache()

def build_paper_personalization(edges_rdd, nodes, num_nodes, time_bounds):
    list_Xi = compute_time_scores(edges_rdd, time_bounds)
    SUM_Xi = list_Xi.values().sum()
    return normalize_time_scores(nodes, list_Xi, SUM_Xi, num_nodes)