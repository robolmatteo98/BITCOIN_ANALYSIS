from utils.mode.paper.paper import build_paper_personalization, value_weight_paper

# 2a parte della formula
def compute_contribs(item):
    _, (neighbors, rank) = item
    total_weight = sum(weight for _, weight in neighbors)
    if total_weight <= 0:
        return []
    return [(dst, rank * (weight / total_weight)) for dst, weight in neighbors]

def build_links(edges_rdd, weight_function):
    """
        [(src_1, [(dst_1, value_1), ... (dst_k, value_k)]), ... (src_n, [(...)])]
    """
    
    return (
        edges_rdd
        .map(lambda edge: (edge["src"], (edge["dst"], weight_function(edge))))
        .groupByKey()
        .mapValues(list)
        .cache()
    )

def add_sink_nodes(links, nodes):
    source_nodes = links.keys() # [nodo_1, ... nodo_n] dai links (sottoinsieme di nodes)
    sink_only_nodes = nodes.subtract(source_nodes) # [nodo_2, ... nodo_n-1] sottrae l'insieme dei nodi con archi uscenti all'insieme dei nodi totali
    sink_nodes = sink_only_nodes.map(lambda node: (node, [])) # costruisce [(nodo_2, []), ...]
    return links.union(sink_nodes).cache() # [(nodo_1, [(dst_1, value_1), ...]), (dst_2, []), ...]

def run_pagerank(edges_rdd, iterations, damping, time_bounds=None):
    nodes = edges_rdd.flatMap(lambda edge: [edge["src"], edge["dst"]]).distinct().cache()
    num_nodes = nodes.count()
    # [nodo_1, ... nodo_n]

    ranks = nodes.map(lambda node: (node, 1.0 / num_nodes)).cache()
    # [(nodo_1, rank_1), ... (nodo_n, rank_n)]

    links = build_links(edges_rdd, lambda edge: value_weight_paper(edge, time_bounds))
    links = add_sink_nodes(links, nodes)
    # [(src_1, [(dst_11, value_11), ... (dst_1k, value_1k)]), (src_n, [(dst_n1, value_n1), ... (dst_nk, value_nk)])]

    prima_parte_formula = build_paper_personalization(edges_rdd, nodes, num_nodes, time_bounds)
    # [(src_1, pers_1), ... (src_n, pers_n)]

    for _ in range(iterations):

        joined = links.join(ranks).cache()
        # [(src_1, ([(dst_1, value_1), ...], rank_1), ...]

        contribs = joined.flatMap(compute_contribs)
        # [(dst_1, contribs_1), (dst_1, contribs_2), ... (dst_n, constribs_k)]
        
        seconda_parte_formula = contribs.reduceByKey(lambda a, b: a + b)
        # [(dst_1, sum_contribs_1), ... (dst_n, sum_contribs_n)]

        dangling_mass = (
            joined
            .filter(lambda item: len(item[1][0]) == 0)
            .map(lambda item: item[1][1])
            .sum()
        )
        # VALUE

        base_term = (1.0 - damping) + damping * dangling_mass
        # VALUE
        
        ranks = (
            prima_parte_formula
            .leftOuterJoin(seconda_parte_formula)
            .mapValues(lambda formula: base_term * formula[0] + damping * (formula[1] or 0.0))
        )
        # [(nodo_1, rank_1), ... (nodo_n, rank_n)]

    return ranks