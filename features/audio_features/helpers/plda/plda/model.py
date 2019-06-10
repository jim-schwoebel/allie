# Copyright 2017 Ravi Sojitra. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal as gaussian
from .optimizer import get_prior_params
from .optimizer import get_posterior_params
from .optimizer import get_posterior_predictive_params
from .optimizer import optimize_maximum_likelihood
from .optimizer import calc_scatter_matrices


def get_space_walk(from_space, to_space):
    U_model_to_D = ['U_model', 'U', 'X', 'D']
    D_to_U_model = U_model_to_D[::-1]

    assert from_space in U_model_to_D and to_space in U_model_to_D

    from_idx = U_model_to_D.index(from_space)
    to_idx = U_model_to_D.index(to_space)

    if to_idx < from_idx:
        spaces = D_to_U_model

        from_idx = D_to_U_model.index(from_space)
        to_idx = D_to_U_model.index(to_space)

    else:
        spaces = U_model_to_D

    from_spaces = [x for x in spaces[from_idx: to_idx]]
    to_spaces = [x for x in spaces[from_idx + 1: to_idx + 1]]

    return zip(from_spaces, to_spaces)


def transform_D_to_X(data, pca):
    return data if pca is None else pca.transform(data)


def transform_X_to_U(data, inv_A, m):
    return np.matmul(data - m, inv_A.T)


def transform_U_to_U_model(data, relevant_U_dims):
    return data[..., relevant_U_dims]


def transform_U_model_to_U(data, relevant_U_dims, U_dimensionality):
    shape = (*data.shape[:-1], U_dimensionality)

    U = np.zeros(shape)
    U[..., relevant_U_dims] = data

    return U


def transform_U_to_X(data, A, m):
    return m + np.matmul(data, A.T)


def transform_X_to_D(data, pca):
    return data if pca is None else pca.inverse_transform(data)


class Model:
    def __init__(self, row_wise_data, labels, n_principal_components=None):
        assert len(row_wise_data.shape) == 2
        assert len(labels) == row_wise_data.shape[0]

        self.pca = None
        self.m = None
        self.A = None
        self.Psi = None
        self.relevant_U_dims = None
        self.inv_A = None

        self.prior_params = None
        self.posterior_params = None
        self.posterior_predictive_params = None

        self.fit(row_wise_data, labels, n_principal_components)

    def calc_logp_posterior(self, v_model, category):
        assert v_model.shape[-1] == self.get_dimensionality('U_model')

        mean = self.posterior_params[category]['mean']
        cov_diag = self.posterior_params[category]['cov_diag']

        return gaussian(mean, np.diag(cov_diag)).logpdf(v_model)

    def calc_logp_posterior_predictive(self, U_model, category):
        assert U_model.shape[-1] == self.get_dimensionality('U_model')

        mean = self.posterior_predictive_params[category]['mean']
        cov_diag = self.posterior_predictive_params[category]['cov_diag']

        return gaussian(mean, np.diag(cov_diag)).logpdf(U_model)

    def calc_logp_marginal_likelihood(self, U_model):
        """ Computes the log marginal likelihood on axis=-2. """
        assert U_model.shape[-1] == self.get_dimensionality('U_model')

        if len(U_model.shape) == 1:
            U_model = U_model[None, :]

        n = U_model.shape[-2]
        psi_diag = self.prior_params['cov_diag']
        n_psi_plus_eye = n * psi_diag + 1

        log_constant = -.5 * n * np.log(2 * np.pi)
        log_constant += -.5 * np.log(n_psi_plus_eye)

        sum_of_squares = np.sum(U_model ** 2, axis=-2)
        log_exponent_1 = -.5 * sum_of_squares

        mean = U_model.mean(axis=-2)
        log_exponent_2 = .5 * (n ** 2 * psi_diag * mean ** 2)
        log_exponent_2 /= n_psi_plus_eye

        logp_ml = log_constant + log_exponent_1 + log_exponent_2
        logp_ml = np.sum(logp_ml, axis=-1)

        return logp_ml

    def calc_logp_prior(self, v_model):
        assert v_model.shape[-1] == self.get_dimensionality('U_model')

        mean = self.prior_params['mean']
        cov_diag = self.prior_params['cov_diag']

        return gaussian(mean, np.diag(cov_diag)).logpdf(v_model)

    def fit(self, data, labels, n_principal_components=None):
        if n_principal_components is None:
            S_b, S_w = calc_scatter_matrices(data, labels)
            matrix_rank = np.linalg.matrix_rank(S_w)

        else:
            matrix_rank = n_principal_components

        if matrix_rank != data.shape[-1]:
            self.pca = PCA(n_components=matrix_rank)
            self.pca.fit(data)

        X = self.transform(data, from_space='D', to_space='X')

        self.m, self.A, self.Psi, self.relevant_U_dims, self.inv_A = \
            optimize_maximum_likelihood(X, labels)

        U_model = self.transform(X, from_space='X', to_space='U_model')

        self.prior_params = \
            get_prior_params(self.Psi, self.relevant_U_dims)

        self.posterior_params = \
            get_posterior_params(U_model, labels, self.prior_params)

        self.posterior_predictive_params = \
            get_posterior_predictive_params(self.posterior_params)

    def get_dimensionality(self, space):
        if space == 'U_model':
            return self.relevant_U_dims.shape[0]

        elif space == 'U':
            return self.A.shape[0]

        elif space == 'X':
            return self.A.shape[0]

        elif space == 'D':
            if self.pca is None:
                return self.m.shape[0]

            else:
                return self.pca.n_features_

        else:
            raise ValueError

    def transform(self, data, from_space, to_space):
        """ Potential_spaces: 'D' <---> 'X' <---> 'U' <---> 'U_model'.

        DESCRIPTION
         There are 6 basic transformations to move back and forth
          between the data space, 'D', and the model's space, 'U_model':

         1. From D to X.
             (i.e. from data space to preprocessed space)
            Uses the minimum number of components from
             Principal Components Analysis that
             captures 100% of the variance in the data.

         2. From X to U.
             (i.e. from preprocessed space to latent space)
             See the bottom of p.533 of Ioffe 2006.

         3. From U to U_model.
             (i.e. from latent space to the model space)
             See Fig 2 on p.537 of Ioffe 2006.

         4. From U_model to U.
             (i.e. from the model space to latent space)

         5. From U to X.
             (i.e. from the latent space to the preprocessed space)

         6. From X to D.
             (i.e. from the preprocessed space to the data space)
        """
        if len(data.shape) == 1:
            data = data[None, :]

        if from_space == 'D' and to_space == 'X':
            return transform_D_to_X(data, self.pca)

        elif from_space == 'X' and to_space == 'U':
            return transform_X_to_U(data, self.inv_A, self.m)

        elif from_space == 'U' and to_space == 'U_model':
            return transform_U_to_U_model(data, self.relevant_U_dims)

        elif from_space == 'U_model' and to_space == 'U':
            dim = self.get_dimensionality('U')

            return transform_U_model_to_U(data, self.relevant_U_dims, dim)

        elif from_space == 'U' and to_space == 'X':
            return transform_U_to_X(data, self.A, self.m)

        elif from_space == 'X' and to_space == 'D':
            return transform_X_to_D(data, self.pca)

        else:
            transformed = data

            for space_1, space_2 in get_space_walk(from_space, to_space):
                transformed = self.transform(transformed, space_1, space_2)

            return transformed
