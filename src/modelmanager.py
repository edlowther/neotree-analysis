import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import optuna

from src.datamanager import DataManager
from dataclasses import dataclass, field

class ModelManager():
    def __init__(self, name, y_label, data_filepath, columns_of_interest, seed, clf_function=None, n_trials=None, search_space=None, clf_params={}, scale='all', dummies=False, reduce_cardinality=True, hours_threshold=72):
        self.name = name
        self.y_label = y_label
        self.data_filepath = data_filepath
        self.columns_of_interest = columns_of_interest
        self.seed = seed
        self.clf_function = clf_function
        self.n_trials = n_trials
        self.search_space = search_space
        self.clf_params = clf_params
        self.scale = scale
        self.dummies = dummies
        self.reduce_cardinality = reduce_cardinality
        self.seed_shuffle_rows = []
        self.hours_threshold = hours_threshold
        
    def run(self):
        self.data_manager = DataManager(
            self.data_filepath, 
            scale=self.scale, 
            dummies=self.dummies, 
            reduce_cardinality=self.reduce_cardinality,
            hours_threshold=self.hours_threshold
        )
        self.data_manager.remove_duplicate_predictors(self.columns_of_interest, self.y_label)
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_manager.get_X_y(self.columns_of_interest, self.seed, y_label=self.y_label)
        if self.name != 'benchmark':
            self.optimizer = Optimizer(self.clf_function, self.search_space, self.clf_params, self.X_train, self.y_train)
            if self.search_space:
                self.study = optuna.create_study(direction='maximize')
                self.study.optimize(self.optimizer, n_trials=self.n_trials)
                self.best_clf = self.clf_function(**self.study.best_params, **self.clf_params)
            else:
                self.study = None
                self.best_clf = self.clf_function(**self.clf_params)
            self.best_clf.fit(self.X_train, self.y_train)

        
    def enhance(self):
        for seed in range(100):
            X_train, X_test, y_train, y_test = self.data_manager.get_X_y(self.columns_of_interest, seed, y_label=self.y_label)
            if self.search_space:
                clf = self.clf_function(**self.study.best_params, **self.clf_params)
            else:
                clf = self.clf_function(**self.clf_params)
            clf.fit(X_train, y_train)
            auprc = average_precision_score(y_test, clf.predict_proba(X_test)[:,-1])
            auroc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,-1])
            self.seed_shuffle_rows.append({
                'name': self.name,
                'y_label': self.y_label,
                'reduce_cardinality': self.reduce_cardinality,
                'seed': seed,
                'test_set_auprc': auprc,
                'test_set_auroc': auroc
            })

class Optimizer():
    def __init__(self, clf_function, search_space, clf_params, X_train, y_train):
        self.clf_function = clf_function
        self.search_space = search_space
        self.clf_params = clf_params
        self.X_train = X_train
        self.y_train = y_train

    def __call__(self, trial):
        params = {}
        for param in self.search_space:            
            params[param.name] = getattr(trial, param.trial_func)(param.name, *param.args, **param.kwargs)
        clf = self.clf_function(**params, **self.clf_params)
        score = cross_val_score(clf, self.X_train, self.y_train, scoring='roc_auc', cv=StratifiedKFold(n_splits=5), n_jobs=-1).mean()
        return score

@dataclass
class Param:
    name: str
    trial_func: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
