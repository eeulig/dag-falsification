# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("experiments/paper.mplstyle")
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
from dowhy.gcm.falsify import FalsifyConst
from tqdm import tqdm

from falsifydags.utils import load_json, load_obj, save_obj

COLWIDTH = 3.25063
TEXTWIDTH = 6.75133

parser = argparse.ArgumentParser()
parser.add_argument(
    "--domain-experts",
    type=str,
    nargs="+",
    default=["de-n"],
    help="Domain experts to plot data for: Must be either de-n, de-e, or both. Default: de-n",
)
parser.add_argument(
    "--mechanisms",
    type=str,
    default="linear",
    help="Mechanisms to plot data for. Must be either linear or nonliner. Default: linear",
)
parser.add_argument(
    "--savepath", type=str, default="results/plots", help="Savepath for the plot"
)
args = parser.parse_args()

if isinstance(args.domain_experts, str):
    args.domain_experts = [args.domain_experts]

# If using both domain experts, plot DE-V on left side and DE-E on right side (as in paper)
args.domain_experts.sort(reverse=True)

if not os.path.exists(args.savepath):
    os.mkdir(args.savepath)


def gather_results(PARAMS):
    print(f"Load {PARAMS['SAVEDIR']} ...")
    if os.path.isfile(os.path.join(PARAMS["SAVEDIR"], "results.pkl")):
        results = load_obj(os.path.join(PARAMS["SAVEDIR"], "results.pkl"))
        return results

    results = {}
    for num_nodes in PARAMS["DATASET"]["DAG_ARGS"]["num_nodes"]:
        results[num_nodes] = {}
        for degree in PARAMS["DATASET"]["DAG_ARGS"]["degree"]:
            results[num_nodes][degree] = {}
            for m in [FalsifyConst.VALIDATE_LMC, FalsifyConst.VALIDATE_TPA]:
                results[num_nodes][degree][m] = {}
                for h in PARAMS["HYPOTHESES"]:
                    results[num_nodes][degree][m][h] = {
                        "p-values": [],
                        "histogram": [],
                        "f given": [],
                    }

    for num_nodes in PARAMS["DATASET"]["DAG_ARGS"]["num_nodes"]:
        for degree in PARAMS["DATASET"]["DAG_ARGS"]["degree"]:
            for s in tqdm(
                range(PARAMS["DATASET"]["DAG_ARGS"]["num_samples"]),
                desc=f"Load n={num_nodes}, d={degree}",
            ):
                result = load_obj(
                    os.path.join(
                        PARAMS["SAVEDIR"], f"synthetic_n{num_nodes}__d{degree}_{s}.pkl"
                    )
                )
                for m in [FalsifyConst.VALIDATE_LMC, FalsifyConst.VALIDATE_TPA]:
                    for h in PARAMS["HYPOTHESES"]:
                        # Get fraction of violations for given DAG
                        g_given_violations = (
                            result["hypothesis"][h]["results"]["G_given"][m][
                                FalsifyConst.N_VIOLATIONS
                            ]
                            / result["hypothesis"][h]["results"]["G_given"][m][
                                FalsifyConst.N_TESTS
                            ]
                        )
                        results[num_nodes][degree][m][h]["f given"].append(
                            g_given_violations
                        )
                        # Get fraction of violations for permuted DAGs
                        g_given_perm_violations = [
                            perm[FalsifyConst.N_VIOLATIONS] / perm[FalsifyConst.N_TESTS]
                            for perm in result["hypothesis"][h]["results"][
                                "G_given_perm"
                            ][m]
                        ]
                        results[num_nodes][degree][m][h]["histogram"].append(
                            g_given_perm_violations
                        )
                        # Compute p value
                        p_value = sum(
                            [
                                1
                                for n_perm in g_given_perm_violations
                                if n_perm <= g_given_violations
                            ]
                        ) / len(g_given_perm_violations)
                        results[num_nodes][degree][m][h]["p-values"].append(p_value)

    if not os.path.isfile(os.path.join(PARAMS["SAVEDIR"], "results.pkl")):
        save_obj(results, os.path.join(PARAMS["SAVEDIR"], "results.pkl"))

    return results


results = {}
for de in args.domain_experts:
    params = load_json(f"experiments/configs/{de}_synthetic_{args.mechanisms}.json")
    results[f"{de} params"] = params
    results[f"{de} results"] = gather_results(params)


# Plot p values
plot_args = {
    "de-n": {
        "xlabel": r"$|K| / | \mathbf{V} |$",
        "xticks": [0.2, 0.4, 0.6, 0.8, 1.0],
        "xlim": [0.15, 1.05],
    },
    "de-e": {
        "xlabel": r"$\mathrm{SHD}/|\mathcal{E}|$",
        "xticks": [0.0, 0.5, 1.0, 1.5, 2.0],
        "xlim": [-0.05, 2.05],
    },
}

fig, ax = plt.subplots(
    figsize=(COLWIDTH * 0.5 * len(args.domain_experts), COLWIDTH * 0.3),
    ncols=len(args.domain_experts),
    sharey=True,
    gridspec_kw={"wspace": 0.1},
    squeeze=False,
)
for d, de in enumerate(args.domain_experts):
    for i_col, num_nodes in enumerate(
        results[f"{de} params"]["DATASET"]["DAG_ARGS"]["num_nodes"]
    ):
        for line, degree in zip(
            ["-", "--", ":"], results[f"{de} params"]["DATASET"]["DAG_ARGS"]["degree"]
        ):
            means = np.array(
                [
                    np.mean(
                        results[f"{de} results"][num_nodes][degree][
                            FalsifyConst.VALIDATE_LMC
                        ][h]["p-values"]
                    )
                    for h in results[f"{de} params"]["HYPOTHESES"]
                ]
            )
            stds = np.array(
                [
                    np.std(
                        results[f"{de} results"][num_nodes][degree][
                            FalsifyConst.VALIDATE_LMC
                        ][h]["p-values"]
                    )
                    for h in results[f"{de} params"]["HYPOTHESES"]
                ]
            )
            ax[0][d].plot(
                results[f"{de} params"]["HYPOTHESES"],
                means,
                line,
                color=cycle[i_col],
                linewidth=1,
                label=f"ER-{num_nodes}-{degree}",
                marker=".",
            )
            ax[0][d].set_xlabel(plot_args[de]["xlabel"])
            ax[0][d].set_xticks(plot_args[de]["xticks"])
            ax[0][d].set_xlim(plot_args[de]["xlim"])
ax[0][0].set_ylim([0, 0.8])
ax[0][0].set_ylabel(r"$p_\mathrm{LMC}$")
ax[0][0].legend(
    loc="lower center",
    bbox_to_anchor=(1.0 if len(args.domain_experts) == 2 else 0.5, 1.01),
    ncol=3,
    fontsize=5.5,
)
plt.savefig(
    os.path.join(
        args.savepath,
        f"p_lmc_synthetic_{args.mechanisms}_{'_'.join(args.domain_experts)}.pdf",
    ),
    bbox_inches="tight",
)
plt.close()
