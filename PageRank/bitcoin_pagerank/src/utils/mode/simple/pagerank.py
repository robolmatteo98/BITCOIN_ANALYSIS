from utils.mode.simple.simple import edge_weight

def compute_contribs(item):
    _, (neighbors, rank) = item
    total_weight = sum(weight for _, weight in neighbors)
    if total_weight <= 0:
        return []
    return [(dst, rank * (weight / total_weight)) for dst, weight in neighbors]

def build_links(edges_rdd, weight_function):
    return (
        edges_rdd
        .map(lambda edge: (edge["src"], (edge["dst"], weight_function(edge))))
        .groupByKey()
        .mapValues(list)
        .cache()
    )

def add_sink_nodes(links, nodes):
    source_nodes = links.keys()
    sink_only_nodes = nodes.subtract(source_nodes)
    sink_nodes = sink_only_nodes.map(lambda node: (node, []))
    return links.union(sink_nodes).cache()

def run_pagerank(edges_rdd, mode, iterations, damping, time_bounds=None):
    nodes = edges_rdd.flatMap(lambda edge: [edge["src"], edge["dst"]]).distinct().cache()
    num_nodes = nodes.count()

    ranks = nodes.map(lambda node: (node, 1.0 / num_nodes)).cache()

    links = build_links(edges_rdd, lambda edge: edge_weight(edge, mode, time_bounds))
    links = add_sink_nodes(links, nodes)

    personalization = nodes.map(lambda node: (node, 1.0 / num_nodes)).cache()

    for _ in range(iterations):
        joined = links.join(ranks).cache()
        contribs = joined.flatMap(compute_contribs)
        summed = contribs.reduceByKey(lambda a, b: a + b)

        dangling_mass = (
            joined
            .filter(lambda item: len(item[1][0]) == 0)
            .map(lambda item: item[1][1])
            .sum()
        )

        base_term = (1.0 - damping) + damping * dangling_mass
        ranks = (
            personalization
            .leftOuterJoin(summed)
            .mapValues(lambda pair: base_term * pair[0] + damping * (pair[1] or 0.0))
        )

    return ranks