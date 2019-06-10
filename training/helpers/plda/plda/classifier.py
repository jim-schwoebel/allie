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
from scipy.special import logsumexp
from .model import Model


class Classifier:
    def __init__(self):
        self.model = None

    def fit_model(self, X, Y, n_principal_components=None):
        self.model = Model(X, Y, n_principal_components)

    def predict(self, data, space='D', normalize_logps=False):
        """ Classifies data into categories present in the training data.

        DESCRIPTION
         Predictions are the MAP estimates,
          i.e. categories with the highest probabilities,
          following the procedure described in the first sentence
          on p.535 of Ioffe 2006.

         See the `calc_logp_pp_categories()` method for documentation on the
          actual equations.

        ARGUMENT
         data  (numpy.ndarray), shape=(..., data_dimensionality)
           - Data must follow statistics convention (row-wise).
           - The last dimension of the array corresponds to data dimension.
           - The dimensionality of the data depends on the space (see below).

        PARAMETERS
         space  (str)
           - Must be either 'D', 'X', 'U', or 'U_model',
              where 'D' is the data space,
                    'X' is the preprocessed space,
                    'U' is the latent space, and
                    'U_model' is the subspace of 'U' the model works in:

                    'D' <---> 'X' <---> 'U' <---> 'U_model'.

           - See `transform()` method in model.py
              for details on the relationship between spaces.

         normalize_logps  (bool)
           - Whether or not to normalize
              the posterior predictive probabilities before returning them.

        RETURNS
         predictions  (numpy.ndarray), shape=data.shape[:-1]

         logps  (numpy.ndarray), shape=(*data.shape[:-1], n_categories)
           - Log posterior predictive probabilities for each category,
              if normalize_logps = False.
           - Log classification probabilities for each category,
              if normalize_logps = True.
             These are just the normalized posterior predictive
              probabilities, aka model certainties.
        """
        if space != 'U_model':
            data = self.model.transform(data,
                                        from_space=space, to_space='U_model')

        logpps_k, K = self.calc_logp_pp_categories(data, normalize_logps)
        predictions = K[np.argmax(logpps_k, axis=-1)]

        return predictions, logpps_k

    def calc_logp_pp_categories(self, data, normalize_logps):
        """ Computes log posterior predictive probabilities for each category.

        DESCRIPTION
         The posterior predictive comes from p.535 of Ioffe 2006.
         The classification procedure is described in the first sentence
          of p.535,
          which clearly implies the prior on categories to be uniform.

        LATEX EQUATIONS
         Normalized posterior predictive (classification certainty):
           ```
           \begin{align}
           p(y_* = T \mid \mathbf{u}_*)
           &= \frac{
                p(\mathbf{u}_*
                \vert
                \mathbf{u}_1^T,
                \mathbf{u}_2^T,
                \dots,
                \mathbf{u}_n^T)
              }{\sum\limits_{k \in K}
                p(\mathbf{u}_*
                \vert
                \mathbf{u}_1^k,
                \mathbf{u}_2^k,
                \dots,
                \mathbf{u}_n^k)
              },
           \end{align}
           ```

         Posterior predictive
           ```
           \begin{align}
           p(\mathbf{u}_*
             \vert
             \mathbf{u}_1^k,
             \mathbf{u}_2^k,
             \dots,
             \mathbf{u}_n^k)
           &= \int \dots \int
              p(\mathbf{u}_* \vert \mathbf{v})
              p(\mathbf{v} \vert
                \mathbf{u}_1^k,
                \mathbf{u}_2^k,
                \dots,
                \mathbf{u}_n^k)
                d\mathbf{v} \\
           &= \mathcal{N}
              \left(
              \mathbf{u}_*
              \mid
              \frac{
                n \mathbf{\Psi}
              }{n \mathbf{\Psi} + \mathbf{I}
              }
              \mathbf{\bar{u}}^k,
              \mathbf{I}
              + \frac{\mathbf{\Psi}
                }{n \mathbf{\Psi} + \mathbf{I}
                }
              \right)
           \end{align}
           ```

        ARGUMENT
         See documentation for the `predict()` method.

        PARAMTER
         See documentation for the `predict()` method.
        """
        assert type(normalize_logps) == bool

        logpps_by_category = []
        K = self.get_categories()

        for k in K:
            logpps_k = self.model.calc_logp_posterior_predictive(data, k)
            logpps_by_category.append(logpps_k)

        logpps_by_category = np.stack(logpps_by_category, axis=-1)

        if normalize_logps:
            norms = logsumexp(logpps_by_category, axis=-1)
            logps = logpps_by_category - norms[..., None]
        else:
            logps = logpps_by_category

        return logps, np.asarray(K)

    def get_categories(self):
        return [k for k in self.model.posterior_params.keys()]
