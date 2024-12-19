# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import warnings
from typing import Optional, Tuple, Union

import networkx as nx
import numpy as np


def random_permute_adj(adj_mat: np.array) -> np.array:
    """
    Randomly permute given adjacency matrix
    :param adj_mat: Adjacency matrix of graph
    :return: Random permutation of that adjacency matrix
    """
    num_nodes = adj_mat.shape[0]
    perm_mat = np.random.permutation(np.eye(num_nodes))
    return perm_mat.T @ adj_mat @ perm_mat


def simulate_dag_erdos_renyi(
    num_nodes: int,
    prob_arcs: Optional[float] = None,
    d_arcs: Optional[float] = None,
    degree: Optional[Union[float, int]] = None,
    seed: Optional[int] = None,
) -> nx.DiGraph:
    """
    Function to generate a random DAG under the Erdos-Renyi model.
    If any of the given sparsity parameters {prob_arcs, d_arcs, degree} leads to an edge probability > 1. we raise a warining.

    :param num_nodes: Number of nodes of the DAG
    :param prob_arcs: Probability of each edge
    :param d_arcs: Expected number of edges d_arcs * num_nodes (e.g. if d_args=2 and num_nodes=10 we expect 20 edges in the DAG).
    :param degree: Expected average degree. This is equal to 2 * d_arcs
    :param seed: Random seed (only used for networkx. One could also set numpy random seed and random.seed instead).
    :return: DAG generated under the Erdos-Renyi model.
    """
    if prob_arcs is None and d_arcs is None and degree is None:
        raise ValueError("Neither prob_arcs, nor d_arcs provided")
    elif d_arcs is not None:
        prob_arcs = (2 * d_arcs) / (num_nodes - 1)
    elif degree is not None:
        prob_arcs = degree / (num_nodes - 1)

    if prob_arcs > 1.0:
        raise Warning(f"Edge probability is {prob_arcs} > 1.")

    graph = nx.erdos_renyi_graph(num_nodes, prob_arcs, directed=True, seed=seed)
    adj_matrix = nx.to_numpy_array(graph)
    adj_matrix_dag = np.tril(random_permute_adj(adj_matrix), k=-1)
    dag = nx.DiGraph(adj_matrix_dag)
    return nx.relabel_nodes(dag, {i: f"X{i}" for i in dag})


def domain_expert_nodes(
    dag: nx.DiGraph,
    knowledge_fraction: Optional[float] = None,
    knowledge_set: Optional[list] = None,
    accidentally_correct_after_shuffling_flag: bool = True,
    max_tries: int = 10000,
    return_knowledge_set: bool = False,
) -> Union[Tuple[nx.DiGraph, list], nx.DiGraph, None]:
    """
    A function to simulate a domain expert (DE) that has node-specific knowledge about the underlying system.
    Specifically, if the DE has knowledge about n nodes, we assume that he knows all direct causal relations between those
    n nodes but nothing about the remaining causal relations except for the sparsity of the systems DAG and the fact
    that it is acyclic. If further `shuffle_avoid_true=True` we will assume that the DE will never set correct edges
    between nodes not set in the knowledge set (by chance).
    Note that for certain input parameters we will not be able to simulate a domain expert (especially due to the
    acyclicity constraint). Therefore, the argument `max_tries` sets the maximum number of retries after which we return
    None.

    :param dag: Ground truth DAG
    :param knowledge_fraction: Fraction of nodes about which the domain expert has full knowledge.
    :param knowledge_set: Set of nodes over which the domain expert has full knowledge.
    :param accidentally_correct_after_shuffling_flag: If set to true allow the node expert to place correct edges
    between nodes where at least one is not in the knowledge by chance.
    :param max_tries: Maximum number of tries after which the function returns None
    :param return_knowledge_set: Whether to return the knowledge set
    :return: Either a tuple (DAG, knowledege_set) if `return_K`=True or a DAG.
    """

    if knowledge_fraction is None and knowledge_set is None:
        raise ValueError("Either knowledge set or knowledge fraction must be provided!")

    A = nx.to_numpy_array(dag)
    if not np.all((A == 0.0) | (A == 1.0)):
        warnings.warn(
            f"Input graph has weights != 1. Note, that weights are not preseved in the returned domain expert"
            f"graph!"
        )

    for tries in range(max_tries):
        adj = nx.to_numpy_array(dag, weight=None)

        if knowledge_set:
            # Get node ids from given knowledge set K
            knowledge_nodes = np.array(
                [i for i, node in enumerate(dag.nodes) if node in knowledge_set]
            )
        elif knowledge_fraction:
            # Get random fraction of variables of which we have knowledge
            knowledge_nodes = np.random.choice(
                adj.shape[0],
                size=round(knowledge_fraction * adj.shape[0]),
                replace=False,
            )
        else:
            raise ValueError(f"Must provide either knowledge_fraction or K!")

        # For all node pairs of which we have knowledge:
        #   - if there is an edge between i and j: adj[i, j] = 3
        #   - if there is no edge between i and j: adj[i, j] = 2
        for i in knowledge_nodes:
            for j in knowledge_nodes:
                adj[i, j] += 2

        # adj[i, j] = 1 are the edges which we will reshuffle
        num_edges_shuffle = len(np.where(adj == 1)[0])

        # If accidentally_correct_after_shuffling_flag=False, we never place a true arc (adj==1) among the shuffled
        # ones (only create wrong arcs)
        if accidentally_correct_after_shuffling_flag:
            i_s, j_s = np.where((adj < 2))
        else:
            i_s, j_s = np.where((adj == 0))

        adj[adj == 1] = 0
        idxs = np.random.choice(len(i_s), num_edges_shuffle, replace=False)
        for idx in idxs:
            adj[i_s[idx], j_s[idx]] = 1

        # Revert the operations above. adj[i, j] = 2: No edge in the true dag, adj[i,j] = 3: Edge in the true dag that
        # is kept because we have domain knowledge for those nodes.
        adj[adj == 2] = 0
        adj[adj == 3] = 1

        # Return the hypothesis
        h_dag = nx.DiGraph(adj)
        if nx.algorithms.is_directed_acyclic_graph(h_dag):
            h_dag = nx.relabel_nodes(h_dag, dict(zip(h_dag.nodes, dag.nodes)))
            if return_knowledge_set:
                return h_dag, [
                    n for i, n in enumerate(h_dag.nodes) if i in knowledge_nodes
                ]
            return h_dag
    return None


def domain_expert_edges(
    dag: nx.DiGraph,
    num_changes: Optional[Tuple[int, int]] = None,
    shd: Optional[int] = None,
    max_tries: int = 10000,
) -> Union[nx.DiGraph, None]:
    """
    A domain expert (DE) that has edge-specific knowledge about the underlying system.
    This DE randomly adds, removes, and flips some edges. We will always remove the same amount of edges as we add, to
    ensure that the DAG has the same sparsity as the GT. The amount changes can be either controlled via
    (i) `num_changes`, which expects a tuple of integers (edges to add AND remove, edges to flip). Note that the first
    element of this tuple has to be even, otherwise we would change the sparsity of the DAG
    (ii) `shd`, which expects an integer that determines the SHD of the resulting DAG. This is possible because this DE
    has a direct relation to the SHD via SHD = #Edges in G' not in G + #Edges in G not in G' + #Edges flipped.
    Note that for certain input parameters we will not be able to simulate a domain expert (especially due to the
    acyclicity constraint). Therefore the argument `max_tries` sets the maximum number of retries after which we return
    None.

    :param dag: Ground truth DAG
    :param num_changes: Tuple of integers: (edges to add AND remove, edges to flip)
    :param shd: Desired SHD between the resulting hypothesis and the GT DAG
    :param max_tries: Maximum number of tries after which the function returns None
    :return: DAG
    """
    if num_changes is None and shd is None:
        raise ValueError("Must provide either fractions or shd!")
    if num_changes and sum(num_changes) > 2 * dag.number_of_edges():
        raise ValueError(
            f"Sum of num_changes must be smaller than 2* number of edges but got {num_changes}"
        )
    if num_changes and num_changes[0] % 2 != 0:
        raise ValueError(
            f"First item in num_changes is the number of edges to add + remove. Must be even "
            f"(got {num_changes[0]}), otherwise we change the sparsity of the DAG."
        )
    if shd and shd > 2 * dag.number_of_edges():
        raise ValueError(f"SHD cannot be greater than 2*number of edges but got {shd}")

    if num_changes:
        num_add_rem, num_flip = num_changes[0] // 2, num_changes[1]
    else:
        num_add_rem = np.random.randint(
            max(0, shd - dag.number_of_edges()), shd // 2 + 1
        )
        num_flip = shd - 2 * num_add_rem

    A = nx.to_numpy_array(dag)
    if not np.all((A == 0.0) | (A == 1.0)):
        warnings.warn(
            f"Input graph has weights != 1. Note, that weights are not preseved in the returned domain expert"
            f"graph!"
        )

    for tries in range(max_tries):
        adj = nx.to_numpy_array(dag, weight=None)
        # Add random edges where no edge currently exists (in either direction)
        i_s, j_s = np.where((adj == 0) & (adj.T == 0) & (np.eye(adj.shape[0]) == 0))
        idxs = np.random.choice(len(i_s), num_add_rem, replace=False)
        for idx in idxs:
            adj[i_s[idx], j_s[idx]] = 2

        # Remove random edges where an edge exists in original graph (without the ones added in previous step)
        i_s, j_s = np.where((adj == 1))
        idxs = np.random.choice(len(i_s), num_add_rem, replace=False)
        for idx in idxs:
            adj[i_s[idx], j_s[idx]] = 3

        # Flip random edges where an edge exists in original graph (which we didn't remove in previous step)
        i_s, j_s = np.where((adj == 1))
        idxs = np.random.choice(len(i_s), num_flip, replace=False)
        for idx in idxs:
            adj[i_s[idx], j_s[idx]] = 4
            adj[j_s[idx], i_s[idx]] = 5

        # Now all 2s are new edges, all 3s are removed edges, all 4s are edges that were flipped to 5s
        adj[adj == 2] = 1
        adj[adj == 3] = 0
        adj[adj == 4] = 0
        adj[adj == 5] = 1

        h_dag = nx.DiGraph(adj)
        if nx.algorithms.is_directed_acyclic_graph(h_dag):
            return nx.relabel_nodes(h_dag, dict(zip(h_dag.nodes, dag.nodes)))
    return None
