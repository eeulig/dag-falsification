# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from dowhy.gcm import InvertibleStructuralCausalModel, draw_samples
from dowhy.gcm.causal_models import PARENTS_DURING_FIT
from dowhy.graph import get_ordered_predecessors

from falsifydags.utils.synthetic_data import (
    _get_random_function,
    _get_random_root_variable_model,
    _SimpleFeedForwardNetwork,
)


def load_sachs_data() -> Tuple[pd.DataFrame, nx.DiGraph]:
    """
    We load the data from the supplemental material of [1], avaliable at [2].
    [1] K. Sachs, O. Perez, D. Pe’er, D. A. Lauffenburger, and G. P. Nolan, “Causal Protein-Signaling Networks Derived
    from Multiparameter Single-Cell Data,” Science, vol. 308, no. 5721, pp. 523–529, Apr. 2005.
    [2] https://www.science.org/doi/abs/10.1126/science.1105809
    """

    dirname = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_excel(os.path.join(dirname, "data", "sachs_data.xls"))
    g_true = nx.read_gml(os.path.join(dirname, "data", "sachs_g_true.gml"))
    return data, g_true


def load_auto_data():
    data = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original",
        delim_whitespace=True,
        header=None,
        names=[
            "mpg",
            "cylinders",
            "displacement",
            "horsepower",
            "weight",
            "acceleration",
            "model year",
            "origin",
            "car name",
        ],
    )
    data.dropna(inplace=True)
    data.drop(["car name"], axis=1, inplace=True)
    g_true = nx.DiGraph(
        [
            ("cylinders", "displacement"),
            ("displacement", "acceleration"),
            ("displacement", "horsepower"),
            ("displacement", "mpg"),
            ("displacement", "weight"),
            ("weight", "mpg"),
            ("weight", "acceleration"),
            ("horsepower", "acceleration"),
            ("horsepower", "mpg"),
        ]
    )
    return data, g_true


def load_apm_data() -> Tuple[pd.DataFrame, nx.DiGraph]:
    dirname = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(os.path.join(dirname, "data", "APM_data.csv"))
    g_true = nx.read_gml(os.path.join(dirname, "data", "APM_g_true.gml"))

    return data, g_true


def generate_random_dataset_from_dag(
    dag: nx.DiGraph,
    num_samples: int = 1000,
    return_model: bool = False,
    prob_nonlinear: float = 0.8,
    gaussian_only: bool = False,
    linear_coef_space: list = [-1.0, 1.0],
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, InvertibleStructuralCausalModel]]:
    """
    Generates a random functional causal model from a given DAG.

    :param dag: Directed Acyclic Graph for which to generate a random FCM.
    :param num_samples: Number of samples to generate.
    :param return_model: Whether to return the randomly generated model.
    :param prob_nonlinear: Probability of nonlinear functions.
    :param gaussian_only: Model root nodes as normal distributions only.
    :param linear_coef_space: Define the coefficient space of linear functions. If [a, b] we draw parameters uniformly
    from the interval [a,b]. If [[a,b], [c,d], ...], we draw parameters uniformly from [a,b] U [c,d] U ... .
    :return: Randomly generated data from the DAG and optionally the functional model used to generate that data.
    """

    invertible_scm = InvertibleStructuralCausalModel(dag)
    training_data = pd.DataFrame()

    for node in nx.topological_sort(invertible_scm.graph):
        if invertible_scm.graph.in_degree(node) == 0:
            random_stochastic_model = _get_random_root_variable_model(
                is_root_node=True, gaussian_only=gaussian_only
            )
            invertible_scm.set_causal_mechanism(node, random_stochastic_model)
            training_data[node] = random_stochastic_model.draw_samples(1000).squeeze()
        else:
            parents = get_ordered_predecessors(invertible_scm.graph, node)
            causal_model = _get_random_function(
                len(parents),
                prob_nonlinear=prob_nonlinear,
                linear_coef_space=linear_coef_space,
            )

            if isinstance(causal_model.prediction_model, _SimpleFeedForwardNetwork):
                # Only calling fit here to learn the normalization of the inputs. It does not train weights etc., i.e.,
                # the Y values doesn't matter.
                causal_model.prediction_model.fit(
                    X=training_data[parents].to_numpy(), Y=np.zeros((1000, 1))
                )
            invertible_scm.set_causal_mechanism(node, causal_model)

            training_data[node] = causal_model.draw_samples(
                training_data[parents].to_numpy()
            )

            invertible_scm.graph.add_node(node, causal_mechanism=causal_model)
            if isinstance(causal_model.prediction_model, _SimpleFeedForwardNetwork):
                causal_model.fit(
                    X=training_data[parents].to_numpy(), Y=np.zeros((1000, 1))
                )
            training_data[node] = causal_model.draw_samples(
                training_data[parents].to_numpy()
            )

        # Update local hash
        invertible_scm.graph.nodes[node][PARENTS_DURING_FIT] = get_ordered_predecessors(
            invertible_scm.graph, node
        )

    data = draw_samples(invertible_scm, num_samples=num_samples).sort_index(axis=1)

    if not return_model:
        return data
    return data, invertible_scm
