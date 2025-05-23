# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from dowhy.gcm.config import default_n_jobs
from dowhy.gcm.independence_test.kernel_operation import apply_rbf_kernel
from dowhy.gcm.stats import merge_p_values_quantile
from dowhy.gcm.util.general import (
    apply_one_hot_encoding,
    fit_one_hot_encoders,
    set_random_seed,
    shape_into_2d,
)
from joblib import Parallel, delayed
from numpy.linalg import LinAlgError, pinv, svd
from scipy.stats import gamma, norm
from sklearn.preprocessing import scale

EPS = os.getenv("EPSILON_VALUE", np.finfo(np.float64).eps)


def correlation_based(
    X: np.ndarray, Y: np.ndarray, Z: Optional[np.ndarray] = None
) -> float:
    """
    Hypothesis test for partial correlation.
    Computes a p-value for the hypothesis that X and Y are uncorrelated given Z by computing the partial correlation.
    For normal random variables with linear relationships (we can linearly regress X on Z and Y on Z), this is
    equivalent to testing conditional independence.
    Here, the partial correlation is computed via matrix inversion. and test is inspired by [1, Section 2.2.2]
    [1] "Estimating High-Dimensional Directed Acyclic Graphs with the PC-Algorithm" by Kalisch and Buehlmann in JMLR 8 (2007) 613-636.

    :param X: samples from 1-dimensional random variable
    :param Y: samples from 1-dimensional random variable
    :param Z: samples from conditioning variable. Can be None (test correlation) or samples from a finite dimensional
    variable
    :return: p value for the hypothesis that X and Y are uncorrelated given Z
    """
    if Z is None:
        X, Y = shape_into_2d(X, Y)
        XYZ = np.transpose(np.concatenate((X, Y), axis=1))
    else:
        X, Y, Z = shape_into_2d(X, Y, Z)
        XYZ = np.transpose(np.concatenate((X, Y, Z), axis=1))

    if X.shape[1] > 1 or Y.shape[1] > 1:
        raise RuntimeError(
            "X and Y need to be 1-dimensional variables, "
            "were {}-d and {}-d".format(X.shape[1], Y.shape[1])
        )

    n = X.shape[0]  # samplesize
    k = 0 if Z is None else Z.shape[1]  # dimensionality of conditioning set
    if n - k - 3 < 0:
        raise RuntimeError("Not enough samples or too large conditioning set")

    covariance = np.cov(XYZ)
    precision = np.linalg.inv(covariance)
    p_01 = precision[0, 1] / (
        np.sqrt(precision[0, 0]) * np.sqrt(precision[1, 1])
    )  # partial correlation X, Y given Z

    if p_01 == 1.0:
        raise RuntimeError(
            "X and Y are perfectly correlated. Either they are the same or they are perfectly correlated"
            "non Gaussian."
        )

    z = 1 / 2 * np.log((1 + p_01) / (1 - p_01))  # Fisher's z transform

    p_value = 2 * (1 - norm.cdf(np.sqrt(n - k - 3) * abs(z)))

    return p_value


def kernel_based(
    X: np.ndarray,
    Y: np.ndarray,
    Z: Optional[np.ndarray] = None,
    kernel: Callable[[np.ndarray], np.ndarray] = apply_rbf_kernel,
    scale_data: bool = True,
    use_bootstrap: bool = True,
    bootstrap_num_runs: int = 20,
    bootstrap_num_samples_per_run: int = 2000,
    bootstrap_n_jobs: Optional[int] = None,
    p_value_adjust_func: Callable[
        [Union[np.ndarray, List[float]]], float
    ] = merge_p_values_quantile,
) -> float:
    """
    Prepares the data and uses kernel (conditional) independence test. Depending whether Z is given, a
    conditional or pairwise independence test is performed.

    If Z is given: Using KCI as conditional independence test.
    If Z is not given: Using HSIC as pairwise independence test.

    return: The p-value for the null hypothesis that X and Y are independent (given Z).
    """
    bootstrap_n_jobs = default_n_jobs if bootstrap_n_jobs is None else bootstrap_n_jobs

    def evaluate_kernel_test_on_samples(
        X: np.ndarray, Y: np.ndarray, Z: np.ndarray, parallel_random_seed: int
    ) -> float:
        set_random_seed(parallel_random_seed)

        X = _remove_zero_std_columns(X)
        Y = _remove_zero_std_columns(Y)

        if X.shape[1] == 0 or Y.shape[1] == 0:
            # Either X and/or Y is constant.
            return 1.0

        if Z is not None:
            Z = _remove_zero_std_columns(Z)
            if Z.shape[1] == 0:
                # If Z is empty, we are in the pairwise setting.
                Z = None

        try:
            if Z is None:
                return _hsic(X, Y, kernel=kernel, scale_data=scale_data)
            else:
                return _kci(X, Y, Z, kernel=kernel, scale_data=scale_data)
        except LinAlgError:
            return np.nan

    if use_bootstrap and X.shape[0] > bootstrap_num_samples_per_run:
        random_indices = [
            np.random.choice(
                X.shape[0],
                min(X.shape[0], bootstrap_num_samples_per_run),
                replace=False,
            )
            for run in range(bootstrap_num_runs)
        ]

        random_seeds = np.random.randint(
            np.iinfo(np.int32).max, size=len(random_indices)
        )
        p_values = Parallel(n_jobs=bootstrap_n_jobs)(
            delayed(evaluate_kernel_test_on_samples)(
                X[indices],
                Y[indices],
                Z[indices] if Z is not None else None,
                random_seed,
            )
            for indices, random_seed in zip(random_indices, random_seeds)
        )

        return p_value_adjust_func(p_values)
    else:
        return evaluate_kernel_test_on_samples(
            X, Y, Z, np.random.randint(np.iinfo(np.int32).max, size=1)[0]
        )


def _kci(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    kernel: Callable[[np.ndarray], np.ndarray],
    scale_data: bool,
    regularization_param: float = 10**-3,
) -> float:
    """
    Tests the null hypothesis that X and Y are independent given Z using the kernel conditional independence test.

    This is a corrected reimplementation of the KCI method in the CondIndTests R-package. Authors of the original R
    package: Christina Heinze-Deml, Jonas Peters, Asbjoern Marco Sinius Munk

    :return: The p-value for the null hypothesis that X and Y are independent given Z.
    """
    X, Y, Z = _convert_to_numeric(*shape_into_2d(X, Y, Z))

    if X.shape[0] != Y.shape[0] != Z.shape[0]:
        raise RuntimeError("All variables need to have the same number of samples!")

    n = X.shape[0]

    if scale_data:
        X = scale(X)
        Y = scale(Y)
        Z = scale(Z)

    k_x = kernel(X)
    k_y = kernel(Y)
    k_z = kernel(Z)

    k_xz = k_x * k_z

    k_xz = _fast_centering(k_xz)
    k_y = _fast_centering(k_y)
    k_z = _fast_centering(k_z)

    r_z = np.eye(n) - k_z @ pinv(k_z + regularization_param * np.eye(n))

    k_xz_z = r_z @ k_xz @ r_z.T
    k_y_z = r_z @ k_y @ r_z.T

    # Not dividing by n, seeing that the expectation and variance are also not divided by n and n**2, respectively.
    statistic = np.sum(k_xz_z * k_y_z.T)

    # Taking the sum, because due to numerical issues, the matrices might not be symmetric.
    eigen_vec_k_xz_z, eigen_val_k_xz_z, _ = svd((k_xz_z + k_xz_z.T) / 2)
    eigen_vec_k_y_z, eigen_val_k_y_z, _ = svd((k_y_z + k_y_z.T) / 2)

    # Filter out eigenvalues that are too small.
    eigen_val_k_xz_z, eigen_vec_k_xz_z = _filter_out_small_eigen_values_and_vectors(
        eigen_val_k_xz_z, eigen_vec_k_xz_z
    )
    eigen_val_k_y_z, eigen_vec_k_y_z = _filter_out_small_eigen_values_and_vectors(
        eigen_val_k_y_z, eigen_vec_k_y_z
    )

    if len(eigen_val_k_xz_z) == 1:
        empirical_kernel_map_xz_z = eigen_vec_k_xz_z * np.sqrt(eigen_val_k_xz_z)
    else:
        empirical_kernel_map_xz_z = (
            eigen_vec_k_xz_z
            @ (np.eye(len(eigen_val_k_xz_z)) * np.sqrt(eigen_val_k_xz_z)).T
        )

    empirical_kernel_map_xz_z = empirical_kernel_map_xz_z.squeeze()
    empirical_kernel_map_xz_z = empirical_kernel_map_xz_z.reshape(
        empirical_kernel_map_xz_z.shape[0], -1
    )

    if len(eigen_val_k_y_z) == 1:
        empirical_kernel_map_y_z = eigen_vec_k_y_z * np.sqrt(eigen_val_k_y_z)
    else:
        empirical_kernel_map_y_z = (
            eigen_vec_k_y_z
            @ (np.eye(len(eigen_val_k_y_z)) * np.sqrt(eigen_val_k_y_z)).T
        )

    empirical_kernel_map_y_z = empirical_kernel_map_y_z.squeeze()
    empirical_kernel_map_y_z = empirical_kernel_map_y_z.reshape(
        empirical_kernel_map_y_z.shape[0], -1
    )

    num_eigen_vec_xz_z = empirical_kernel_map_xz_z.shape[1]
    num_eigen_vec_y_z = empirical_kernel_map_y_z.shape[1]

    size_w = num_eigen_vec_xz_z * num_eigen_vec_y_z

    w = (
        empirical_kernel_map_y_z[:, None] * empirical_kernel_map_xz_z[..., None]
    ).reshape(empirical_kernel_map_y_z.shape[0], -1)

    if size_w > n:
        ww_prod = w @ w.T
    else:
        ww_prod = w.T @ w

    return _estimate_p_value(ww_prod, statistic)


def _hsic(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: Callable[[np.ndarray], np.ndarray],
    scale_data: bool,
    cut_off_value: float = EPS,
) -> float:
    """
    Estimates the Hilbert-Schmidt Independence Criterion score for a pairwise independence test between variables X
    and Y.

    This is a reimplementation from the original Matlab code provided by the authors.

    :return: The p-value for the null hypothesis that X and Y are independent.
    """
    X, Y = _convert_to_numeric(*shape_into_2d(X, Y))

    if X.shape[0] != Y.shape[0]:
        raise RuntimeError("All variables need to have the same number of samples!")

    if X.shape[0] < 6:
        raise RuntimeError(
            "At least 6 samples are required for the HSIC independence test. Only %d were given."
            % X.shape[0]
        )

    n = X.shape[0]

    if scale_data:
        X = scale(X)
        Y = scale(Y)

    k_mat = kernel(X)
    l_mat = kernel(Y)

    k_c = _fast_centering(k_mat)
    l_c = _fast_centering(l_mat)

    #  Test statistic is given as np.trace(K @ H @ L @ H) / n. Below computes without matrix products.
    test_statistic = (
        1
        / n
        * (
            np.sum(k_mat * l_mat)
            - 2 / n * np.sum(k_mat, axis=0) @ np.sum(l_mat, axis=1)
            + 1 / n**2 * np.sum(k_mat) * np.sum(l_mat)
        )
    )

    var_hsic = (k_c * l_c) ** 2
    var_hsic = (np.sum(var_hsic) - np.trace(var_hsic)) / n / (n - 1)
    var_hsic = var_hsic * 2 * (n - 4) * (n - 5) / n / (n - 1) / (n - 2) / (n - 3)

    k_mat = k_mat - np.diag(np.diag(k_mat))
    l_mat = l_mat - np.diag(np.diag(l_mat))

    bone = np.ones((n, 1), dtype=float)
    mu_x = (bone.T @ k_mat @ bone) / n / (n - 1)
    mu_y = (bone.T @ l_mat @ bone) / n / (n - 1)

    m_hsic = (1 + mu_x * mu_y - mu_x - mu_y) / n

    var_hsic = max(var_hsic.squeeze(), cut_off_value)
    m_hsic = max(m_hsic.squeeze(), cut_off_value)
    if test_statistic <= cut_off_value:
        test_statistic = 0

    al = m_hsic**2 / var_hsic
    bet = var_hsic * n / m_hsic

    p_value = 1 - gamma.cdf(test_statistic, al, scale=bet)

    return p_value


def _fast_centering(k: np.ndarray) -> np.ndarray:
    """
    Compute centered kernel matrix in time O(n^2). The centered kernel matrix is defined as K_c = H @ K @ H, with
    H = identity - 1/ n * ones(n,n). Computing H @ K @ H via matrix multiplication scales with n^3. The
    implementation circumvents this and runs in time n^2.
    :param k: original kernel matrix of size nxn
    :return: centered kernel matrix of size nxn
    """
    n = len(k)
    k_c = (
        k
        - 1 / n * np.outer(np.ones(n), np.sum(k, axis=0))
        - 1 / n * np.outer(np.sum(k, axis=1), np.ones(n))
        + 1 / n**2 * np.sum(k) * np.ones((n, n))
    )
    return k_c


def _filter_out_small_eigen_values_and_vectors(
    eigen_values: np.ndarray,
    eigen_vectors: np.ndarray,
    relative_tolerance: float = (10**-5),
) -> Tuple[np.ndarray, np.ndarray]:
    filtered_indices_xz_z = np.where(
        eigen_values[eigen_values > max(eigen_values) * relative_tolerance]
    )

    return eigen_values[filtered_indices_xz_z], eigen_vectors[:, filtered_indices_xz_z]


def _estimate_p_value(ww_prod: np.ndarray, statistic: np.ndarray) -> float:
    # Dividing by n not required since we do not divide the test statistical_tools by n.
    mean_approx = np.trace(ww_prod)
    variance_approx = 2 * np.trace(ww_prod @ ww_prod)

    alpha_approx = mean_approx**2 / variance_approx
    beta_approx = variance_approx / mean_approx

    return 1 - gamma.cdf(statistic, alpha_approx, scale=beta_approx)


def _estimate_column_wise_covariances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.cov(X, Y, rowvar=False)[: X.shape[1], -Y.shape[1] :]


def _convert_to_numeric(*args) -> List[np.ndarray]:
    return [apply_one_hot_encoding(X, fit_one_hot_encoders(X)) for X in args]


def _remove_zero_std_columns(X: np.ndarray) -> np.ndarray:
    X = shape_into_2d(X)
    return X[:, [np.unique(X[:, i]).shape[0] > 1 for i in range(X.shape[1])]]
