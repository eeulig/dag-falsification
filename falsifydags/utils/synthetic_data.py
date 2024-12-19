# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional

import numpy as np
from dowhy.gcm.causal_mechanisms import (
    AdditiveNoiseModel,
    FunctionalCausalModel,
    PredictionModel,
    StochasticModel,
)
from dowhy.gcm.ml.regression import SklearnRegressionModel
from dowhy.gcm.stochastic_models import ScipyDistribution
from dowhy.gcm.util.general import shape_into_2d
from scipy import stats
from sklearn.linear_model import LinearRegression


class _GaussianMixtureDistribution(StochasticModel):
    """
    Gaussian mixture distribution object.
    NOTE: This only allows an univariate distribution!

    Class attributes:
    __means: A list of means of the Gaussians distributions.
    __std_values: A list of standard deviations of the Gaussians distributions.
    __parameters: A numpy array representing the means and standard deviations.
    __weights: Weights of the gaussian distributions. Here. they are uniform.
    __normalize: Indicates whether the output should be normalized by X / (max[means] - min[means])
    """

    def __init__(
        self,
        means: List[float],
        std_values: List[float],
        normalize: Optional[bool] = False,
    ) -> None:
        """
        Initializes the GaussianMixtureDistribution.

        :param means: The means of the Gaussians.
        :param std_values: The standard deviations of the Gaussians.
        :param normalize: True if the output should be normalized in terms of (output - min(means)) / (max(means) -
        min(means)). False if the output should not be normalized. Default: False
        :return: None
        """
        self.__means = None
        self.__std_values = None

        self.__normalize = normalize

        self.__set_parameters(means, std_values)

    @property
    def means(self) -> List[float]:
        """
        :return: The means of the Gaussians.
        """
        return self.__means

    @property
    def std_values(self) -> List[float]:
        """
        :return: The standard deviations of the Gaussians.
        """
        return self.__std_values

    def __set_parameters(self, means: List[float], std_values: List[float]) -> None:
        """
        Sets the means and standard deviations of the Gaussians.

        :param means: The means of the Gaussians.
        :param std_values: The standard deviation of the Gaussians.
        :return: None
        """
        if len(means) < 2 or len(std_values) < 2:
            raise RuntimeError(
                "At least two means and standard deviations are needed! %d means and %d standard deviations "
                "were given." % (len(means), len(std_values))
            )

        self.__means = means
        self.__std_values = std_values
        self.__init_parameters()

    def fit(self, X: np.ndarray) -> None:
        pass

    def __init_parameters(self) -> None:
        """
        Initializes the parameters. Here, it zips together the means and standard deviations into tuples and
        initializes the weights of the Gaussians uniformly.

        :return: None
        """
        self.__parameters = np.array(
            [list(x) for x in zip(self.__means, self.__std_values)]
        )
        self.__weights = np.ones(self.__parameters.shape[0], dtype=np.float64) / float(
            self.__parameters.shape[0]
        )

    def draw_samples(self, num_samples: int) -> np.ndarray:
        """
        Randomly samples from the defined Gaussian mixture distribution.

        :param num_samples: The number of samples.
        :return: The generated samples.
        """
        mixture_ids = np.random.choice(
            self.__parameters.shape[0], size=num_samples, replace=True, p=self.__weights
        )

        result = np.fromiter(
            (stats.norm.rvs(*(self.__parameters[i])) for i in mixture_ids),
            dtype=np.float64,
        )

        if self.__normalize:
            result = (
                2
                * (result - np.min(self.__means))
                / (np.max(self.__means) - np.min(self.__means))
                - 1
            )

        return shape_into_2d(result)

    def clone(self):
        return _GaussianMixtureDistribution(
            self.__means, self.__std_values, self.__normalize
        )


class _SimpleFeedForwardNetwork(PredictionModel):
    """
    A simple feed forward network. This is mostly used for creating random MLPs.

    Class attributes:
    __init_data: Initial data for finding the minimum and maximum output value of the network in order to
    normalize it.
    __weights: Network weights for each layer.
    __min_val: Minimum value of the output. This is used to normalize the data on [-1, 1].
    __max_val: Maximum value of the output. This is used to normalize the data on [-1, 1].
    """

    def __init__(self, weights: Dict[int, np.ndarray]) -> None:
        self.__weights = weights
        self.__min_val = None
        self.__max_val = None

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs: Optional[Any]) -> None:
        """Only needed to find the min and max output value for normalization."""
        tmp_output = self.predict(X, normalize=False)
        self.__min_val = np.min(tmp_output)
        self.__max_val = np.max(tmp_output)

    def clone(self):
        return _SimpleFeedForwardNetwork(self.__weights)

    def predict(self, X: np.ndarray, normalize: Optional[bool] = True) -> np.ndarray:
        current_result = X
        keys = list(self.__weights.keys())
        for q in range(len(keys) - 1):
            current_result = 1 / (
                1 + np.exp(-np.dot(current_result, self.__weights[keys[q]]))
            )

        predictions = np.dot(
            current_result, self.__weights[keys[len(keys) - 1]]
        ).squeeze()
        if normalize:
            return (
                2 * (predictions - self.__min_val) / (self.__max_val - self.__min_val)
                - 1
            )
        else:
            return predictions


def _get_random_root_variable_model(
    is_root_node: bool, gaussian_only: bool = False
) -> StochasticModel:
    """
    Returns a random distribution. These distributions can be:
    - A uniform distribution.
    - A Gaussian distribution.
    - A Gaussian mixture model (only for root nodes).

    :param is_root_node: True if the distribution is meant for a root node. This returns a distribution with greater
    variance. Otherwise, the distribution has a smaller variance. NOTE: Only root nodes can have a Gaussian mixture
    model.
    :param gaussian_only: If set to true, all distributions are N(0,1)
    :return: A random distribution.
    """
    if gaussian_only:
        return ScipyDistribution(stats.norm, loc=0, scale=1)

    if is_root_node:
        rand_val = np.random.randint(0, 3)
    else:
        rand_val = np.random.randint(0, 2)

    if rand_val == 0:
        if is_root_node:
            return ScipyDistribution(stats.norm, loc=0, scale=1)
        else:
            return ScipyDistribution(stats.uniform, loc=0, scale=0.3)
    elif rand_val == 1:
        if is_root_node:
            return ScipyDistribution(stats.uniform, loc=-1, scale=2)
        else:
            val = np.random.uniform(0, 0.5)
            return ScipyDistribution(stats.uniform, loc=-val, scale=2 * val)
    elif rand_val == 2:
        num_components = np.random.choice(4, 1)[0] + 2

        means = []
        std_vals = []

        for i in range(num_components):
            means.append(np.random.uniform(-1, 1))
            std_vals.append(0.5)

        means = np.array(means)
        means = means - np.mean(means)

        return _GaussianMixtureDistribution(
            means=means, std_values=std_vals, normalize=True
        )
    else:
        raise NotImplementedError


def _get_random_function(
    num_inputs: int,
    gaussian_noise_std: float = 0.1,
    prob_nonlinear: float = 0.8,
    linear_coef_space: List = [-1, 1],
) -> FunctionalCausalModel:
    """
    Returns a random function. The functions can be:
    - With 20% chance: A linear function with random coefficients on [-1, 1].
    - With 80% chance: A non-linear function generated by a random neural network with outputs on [-1, 1]

    :param num_inputs: Number of input variables of the function.
    :param gaussian_noise_std: Set the std of the normal distribution for the additive noise model.
    :param prob_nonlinear: Probability of function being nonlinear.
    :param linear_coef_space: Define the coefficient space of linear functions. If [a, b] we draw parameters uniformly
    from the interval [a,b]. If [[a,b], [c,d], ...], we draw parameters uniformly from [a,b] U [c,d] U ...
    :return: A random function.
    """
    rand_val = np.random.uniform(0, 1)

    if rand_val < prob_nonlinear:
        layers = {0: np.random.uniform(-5, 5, (num_inputs, np.random.randint(2, 100)))}
        layers[1] = np.random.uniform(
            -5, 5, (layers[0].shape[1], np.random.randint(2, 100))
        )
        layers[2] = np.random.uniform(-5, 5, (layers[1].shape[1], 1))

        return AdditiveNoiseModel(
            _SimpleFeedForwardNetwork(weights=layers),
            noise_model=ScipyDistribution(stats.norm, loc=0, scale=gaussian_noise_std),
        )

    else:
        linear_reg = LinearRegression()
        if len(linear_coef_space) == 2 and not isinstance(linear_coef_space[0], list):
            linear_reg.coef_ = np.random.uniform(*linear_coef_space, num_inputs)
        elif isinstance(linear_coef_space[0], list):
            linear_coef = []
            lengths = np.array([b - a for (a, b) in linear_coef_space])
            subset_idxs = np.random.choice(
                len(linear_coef_space), p=lengths / np.sum(lengths), size=num_inputs
            )
            for subset_idx in subset_idxs:
                linear_coef.append(np.random.uniform(*linear_coef_space[subset_idx]))
            linear_reg.coef_ = np.array(linear_coef)
        else:
            raise ValueError(
                f"Expected linear_coef_space to be either of the form [a, b] or [[a, b], [c, d], ...]. "
                f"Got {linear_coef_space} instead."
            )
        linear_reg.intercept_ = 0

        return AdditiveNoiseModel(
            SklearnRegressionModel(linear_reg),
            noise_model=ScipyDistribution(stats.norm, loc=0, scale=gaussian_noise_std),
        )
