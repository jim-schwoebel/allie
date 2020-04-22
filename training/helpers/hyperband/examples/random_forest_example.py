"""
==================================================================
Tuning the hyperparameters of a random forest model with hyperband
==================================================================
"""
from hyperband import HyperbandSearchCV

from scipy.stats import randint as sp_randint
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer


if __name__ == '__main__':
    model = RandomForestClassifier()
    param_dist = {
        'max_depth': [3, None],
        'max_features': sp_randint(1, 11),
        'min_samples_split': sp_randint(2, 11),
        'min_samples_leaf': sp_randint(1, 11),
        'bootstrap': [True, False],
        'criterion': ['gini', 'entropy']
    }

    digits = load_digits()
    X, y = digits.data, digits.target
    y = LabelBinarizer().fit_transform(y)

    search = HyperbandSearchCV(model, param_dist,
                               resource_param='n_estimators',
                               scoring='roc_auc',
                               n_jobs=1,
                               verbose=1)
    search.fit(X, y)
    print(search.best_params_)
    print(search.best_score_)
