# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import sys
import warnings
from functools import partial

warnings.simplefilter(action="ignore", category=FutureWarning)
import dowhy.gcm.falsify as falsify
import networkx as nx
import numpy as np
from dowhy.gcm.independence_test.generalised_cov_measure import generalised_cov_based
from dowhy.gcm.ml import SklearnRegressionModel
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm

from falsifydags.datasets import generate_random_dataset_from_dag
from falsifydags.utils import (
    domain_expert_edges,
    domain_expert_nodes,
    load_json,
    save_obj,
    set_random_seed,
    simulate_dag_erdos_renyi,
)
from falsifydags.utils.independence_tests import correlation_based, kernel_based


def create_gradient_boost_regressor(**kwargs) -> SklearnRegressionModel:
    return SklearnRegressionModel(GradientBoostingRegressor(**kwargs))


def gcm(X, Y, Z=None):
    return generalised_cov_based(
        X,
        Y,
        Z=Z,
        prediction_model_X=create_gradient_boost_regressor,
        prediction_model_Y=create_gradient_boost_regressor,
    )


COND_IND_TESTS = {
    "kernel_based": kernel_based,
    "correlation_based": correlation_based,
    "gcm": gcm,
}

PARAMS = load_json(sys.argv[1])


def generate_hypotheses(g_true):
    if PARAMS["HYPOTHESES_TYPE"] == "domain_expert_nodes":
        knowledge_nodes_perm = list(np.random.permutation(list(g_true.nodes)))
        if isinstance(PARAMS["HYPOTHESES"][0], int):
            knowledge_nodes_sets = [
                knowledge_nodes_perm[:i] for i in PARAMS["HYPOTHESES"]
            ]
        elif isinstance(PARAMS["HYPOTHESES"][0], float):
            knowledge_nodes_sets = [
                knowledge_nodes_perm[: round(i * len(knowledge_nodes_perm))]
                for i in PARAMS["HYPOTHESES"]
            ]
        else:
            raise ValueError(
                f'Hypotheses must be array of ints (size of K) or floats (fraction of nodes). Received {type(PARAMS["HYPOTHESES"][0])} instead.'
            )
        given_dags = [
            domain_expert_nodes(
                g_true, knowledge_set=K, accidentally_correct_after_shuffling_flag=True
            )
            for K in knowledge_nodes_sets
        ]

    elif PARAMS["HYPOTHESES_TYPE"] == "domain_expert_edges":
        if isinstance(PARAMS["HYPOTHESES"][0], int):
            given_dags = [
                domain_expert_edges(g_true, shd=i) for i in PARAMS["HYPOTHESES"]
            ]
        elif isinstance(PARAMS["HYPOTHESES"][0], float):
            shds = [int(g_true.number_of_edges() * i) for i in PARAMS["HYPOTHESES"]]
            given_dags = [domain_expert_edges(g_true, shd=i) for i in shds]
    else:
        raise ValueError(
            f"{sys.argv[1]} must have parameter 'HYPOTHESES_TYPE' = {{domain_expert_nodes, "
            f"domain_expert_edges}}. Got {PARAMS['HYPOTHESES_TYPE']} instead."
        )
    return given_dags


def run_experiment(g_true, given_dags, data, savename, p_values_memory=None):
    ind_test = COND_IND_TESTS[PARAMS["validate_graph_args"]["independence_test"]]
    cond_ind_test = COND_IND_TESTS[
        PARAMS["validate_graph_args"]["conditional_independence_test"]
    ]

    RESULTS = dict(params=PARAMS, hypothesis={h: None for h in PARAMS["HYPOTHESES"]})
    if not p_values_memory:
        p_values_memory = falsify._PValuesMemory()
    for g_given, hypotheses in zip(given_dags, PARAMS["HYPOTHESES"]):
        results = {
            "G_true": falsify.run_validations(
                causal_graph=g_true,
                data=data,
                methods=(
                    partial(
                        falsify.validate_lmc,
                        p_values_memory=p_values_memory,
                        independence_test=ind_test,
                        conditional_independence_test=cond_ind_test,
                    ),
                    partial(falsify.validate_tpa, causal_graph_reference=g_true),
                ),
            ),
            "G_given": falsify.run_validations(
                causal_graph=g_given,
                data=data,
                methods=(
                    partial(
                        falsify.validate_lmc,
                        p_values_memory=p_values_memory,
                        independence_test=ind_test,
                        conditional_independence_test=cond_ind_test,
                    ),
                    partial(falsify.validate_tpa, causal_graph_reference=g_given),
                ),
            ),
            "G_given_perm": falsify._permutation_based(
                causal_graph=g_given,
                data=data,
                methods=(
                    partial(
                        falsify.validate_lmc,
                        p_values_memory=p_values_memory,
                        independence_test=ind_test,
                        conditional_independence_test=cond_ind_test,
                    ),
                    partial(falsify.validate_tpa, causal_graph_reference=g_given),
                ),
                exclude_original_order=PARAMS["baseline_args"][
                    "exclude_original_order"
                ],
                n_permutations=PARAMS["baseline_args"]["n_permutations"],
                show_progress_bar=PARAMS["baseline_args"]["show_progress_bar"],
            ),
        }
        RESULTS["hypothesis"][hypotheses] = dict(g_given=g_given, results=results)

    save_obj(RESULTS, os.path.join(PARAMS["SAVEDIR"], f"{savename}.pkl"))


if __name__ == "__main__":
    if not os.path.exists(PARAMS["SAVEDIR"]):
        os.makedirs(PARAMS["SAVEDIR"])

    if PARAMS["DATASET"]["TYPE"] == "synthetic":
        # Run experiments

        def run_single_sample(s, num_nodes, degree, seed):
            np.random.seed(seed)
            set_random_seed(seed)
            # Generate dataset, g_true, and hypotheses
            found_dag = False
            while not found_dag:
                g_true = simulate_dag_erdos_renyi(num_nodes=num_nodes, degree=degree)
                given_dags = generate_hypotheses(g_true)
                if (
                    np.all([isinstance(g_given, nx.DiGraph) for g_given in given_dags])
                    and g_true.number_of_edges() > 0
                ):
                    found_dag = True
            data = generate_random_dataset_from_dag(
                dag=g_true, **PARAMS["DATASET"]["DATASET_ARGS"]
            )
            run_experiment(
                g_true, given_dags, data, f"synthetic_n{num_nodes}__d{degree}_{s}"
            )

        params = []
        for num_nodes in PARAMS["DATASET"]["DAG_ARGS"]["num_nodes"]:
            for degree in PARAMS["DATASET"]["DAG_ARGS"]["degree"]:
                for s in range(PARAMS["DATASET"]["DAG_ARGS"]["num_samples"]):
                    seed = np.random.randint(np.iinfo(np.int32).max)
                    params.append((s, num_nodes, degree, seed))

        random_seeds = np.random.randint(
            np.iinfo(np.int32).max, size=PARAMS["DATASET"]["DAG_ARGS"]["num_samples"]
        )
        Parallel(n_jobs=-1)(delayed(run_single_sample)(*p) for p in tqdm(params))

    elif PARAMS["DATASET"]["TYPE"] in ["sachs", "auto", "apm"]:
        if PARAMS["DATASET"]["TYPE"] == "sachs":
            from falsifydags.datasets import load_sachs_data

            data, g_true = load_sachs_data()

        elif PARAMS["DATASET"]["TYPE"] == "auto":
            from falsifydags.datasets import load_auto_data

            data, g_true = load_auto_data()

        elif PARAMS["DATASET"]["TYPE"] == "apm":
            from falsifydags.datasets import load_apm_data

            data, g_true = load_apm_data()
            # Remove all columns for which we have NaNs (see A.5 in the paper).
            data.dropna(axis=1, inplace=True)

        def run_single_sample(s, seed):
            np.random.seed(seed)
            set_random_seed(seed)
            found_dag = False
            while not found_dag:
                given_dags = generate_hypotheses(g_true)
                if np.all([isinstance(g_given, nx.DiGraph) for g_given in given_dags]):
                    found_dag = True
            run_experiment(g_true, given_dags, data, f"{PARAMS['DATASET']['TYPE']}_{s}")

        random_seeds = np.random.randint(
            np.iinfo(np.int32).max, size=PARAMS["DATASET"]["DAG_ARGS"]["num_samples"]
        )
        Parallel(n_jobs=-1)(
            delayed(run_single_sample)(s, random_seed)
            for s, random_seed in tqdm(
                zip(range(PARAMS["DATASET"]["DAG_ARGS"]["num_samples"]), random_seeds),
                total=PARAMS["DATASET"]["DAG_ARGS"]["num_samples"],
            )
        )
    else:
        raise ValueError(f"Dataset type {PARAMS['DATASET']['TYPE']} not known!")
