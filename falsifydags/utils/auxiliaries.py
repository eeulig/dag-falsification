# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import os
import pickle
import random

import numpy as np


def save_obj(obj, path):
    """Save pickle object to specified path."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    """Load pickle object from specified path."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(filepath):
    with open(filepath, "r") as f:
        obj = json.load(f)
    return obj


def set_random_seed(random_seed: int) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)
