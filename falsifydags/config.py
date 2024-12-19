# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
N_VIOLATIONS = "n_violations"
F_VIOLATIONS = "f_violations"
N_TESTS = "n_tests"
P_VALUE = "p-value"
P_VALUES = "p_values"

GIVEN_VIOLATIONS = N_VIOLATIONS + " g_given"
F_GIVEN_VIOLATIONS = F_VIOLATIONS + " g_given"
PERM_VIOLATIONS = N_VIOLATIONS + " permutations"
F_PERM_VIOLATIONS = F_VIOLATIONS + " permutations"
LOCAL_VIOLATION_INSIGHT = "local violations"

# Define readable names for validation tests
VALIDATION_METHODS = {
    "validate_lmc": "LMC",
    "validate_pd": "Faithfulness",
    "validate_coll": "Colliders",
    "validate_parental_dsep": "dSep",
    "validate_causal_minimality": "Causal Minimality",
}

# Color for plotting local violations
VIOLATION_COLOR = "red"
