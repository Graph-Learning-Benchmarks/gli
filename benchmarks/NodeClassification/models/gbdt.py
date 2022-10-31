"""
CatBoost and Lightgbm model in GLI.

References:
https://github.com/nd7141/bgnn/blob/master/models/GBDT.py
"""

from catboost import Pool, CatBoostClassifier, CatBoostRegressor
import time
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import numpy as np
from collections import defaultdict as ddict
import lightgbm


class GBDTCatBoost:
    """GDBT CatBoost."""

    def __init__(self,
                 task="regression",
                 depth=6,
                 lr=0.1,
                 l2_leaf_reg=None,
                 max_bin=None):
        """Initiate class."""
        self.task = task
        self.depth = depth
        self.learning_rate = lr
        self.l2_leaf_reg = l2_leaf_reg
        self.max_bin = max_bin

    def init_model(self, num_epochs, patience):
        """Initiate model."""
        catboost_model_obj = CatBoostRegressor if self.task == "regression" \
            else CatBoostClassifier
        # self.loss_function = "RMSE"
        # if self.task == "regression" else "CrossEntropy"
        self.loss_function = "RMSE" if self.task == "regression" \
            else "MultiClass"
        self.custom_metrics = ["R2"] if self.task == "regression" \
            else ["Accuracy"]
        # ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC", "R2"],

        print("loss function: ", self.loss_function)
        print("metric: ", self.custom_metrics)

        self.model = catboost_model_obj(iterations=num_epochs,
                                        depth=self.depth,
                                        learning_rate=self.learning_rate,
                                        loss_function=self.loss_function,
                                        custom_metric=self.custom_metrics,
                                        random_seed=0,
                                        early_stopping_rounds=patience,
                                        l2_leaf_reg=self.l2_leaf_reg,
                                        max_bin=self.max_bin,
                                        nan_mode="Min")

    def get_metrics(self):
        """Get metrics."""
        d = self.model.evals_result_
        metrics = ddict(list)
        keys = ["learn", "validation_0", "validation_1"] \
            if "validation_0" in self.model.evals_result_ \
            else ["learn", "validation"]
        for metric_name in d[keys[0]]:
            perf = [d[key][metric_name] for key in keys]
            if metric_name == self.loss_function:
                metrics["loss"] = list(zip(*perf))
            else:
                metrics[metric_name.lower()] = list(zip(*perf))

        return metrics

    def get_test_metric(self, metrics, metric_name):
        """Get test metric."""
        if metric_name == "loss":
            val_epoch = np.argmin([acc[1] for acc in metrics[metric_name]])
        else:
            val_epoch = np.argmax([acc[1] for acc in metrics[metric_name]])
        min_metric = metrics[metric_name][val_epoch]
        return min_metric, val_epoch

    def save_metrics(self, metrics, fn):
        """Save metrics."""
        with open(fn, "w+", encoding="utf-8") as f:
            for key, value in metrics.items():
                print(key, value, file=f)

    def train_val_test_split(self, x, y, train_mask, val_mask, test_mask):
        """Get train/val/test split."""
        x_train, y_train = x[train_mask], y[train_mask]
        x_val, y_val = x[val_mask], y[val_mask]
        x_test, y_test = x[test_mask], y[test_mask]
        return x_train, y_train, x_val, y_val, x_test, y_test

    def fit(self,
            x, y, train_mask, val_mask, test_mask,
            num_epochs=1000, patience=200,
            plot=False, verbose=False,
            loss_fn="", metric_name="loss"):
        """Fit model."""
        x_train, y_train, x_val, y_val, x_test, y_test = \
            self.train_val_test_split(x, y, train_mask, val_mask, test_mask)
        self.init_model(num_epochs, patience)

        start = time.time()
        # print("type(x_train)", type(x_train))
        # print("type(y_train)", type(y_train))
        # print("cat_features", cat_features)
        pool = Pool(x_train.numpy(), y_train.numpy())
        eval_set = [(x_val.numpy(), y_val.numpy()),
                    (x_test.numpy(), y_test.numpy())]
        self.model.fit(pool, eval_set=eval_set, plot=plot, verbose=verbose)
        finish = time.time()

        num_trees = self.model.tree_count_
        print(f"Finished training. Total time: {finish - start:.2f} |\
                Number of trees: {num_trees:d} |\
                Time per tree: {(time.time() - start )/num_trees:.2f}")

        metrics = self.get_metrics()
        min_metric, min_val_epoch = self.get_test_metric(metrics, metric_name)
        if loss_fn:
            self.save_metrics(metrics, loss_fn)
        print(f"Best {metric_name} at iteration {min_val_epoch}:\
                {min_metric[0]:.3f}/{min_metric[1]:.3f}/{min_metric[2]:.3f}")
        return metrics

    def predict(self, x_test, y_test):
        """Predict."""
        pred = self.model.predict(x_test)

        metrics = {}
        metrics["rmse"] = mean_squared_error(pred, y_test) ** .5

        return metrics


class GBDTLGBM:
    """GBDT Lightgbm."""

    def __init__(self, task="regression", lr=0.1, num_leaves=31, max_bin=255,
                 lambda_l1=0., lambda_l2=0., boosting="gbdt"):
        """Initiate lightgbm."""
        self.task = task
        self.boosting = boosting
        self.learning_rate = lr
        self.num_leaves = num_leaves
        self.max_bin = max_bin
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2

    def accuracy(self, preds, train_data):
        """Calculate accuracy."""
        labels = train_data.get_label()
        preds_classes = preds.reshape((preds.shape[0]//labels.shape[0],
                                       labels.shape[0])).argmax(0)
        return "accuracy", accuracy_score(labels, preds_classes), True

    def r2(self, preds, train_data):
        """Calculate R2."""
        labels = train_data.get_label()
        return "r2", r2_score(labels, preds), True

    def init_model(self):
        """Initiate model."""
        self.parameters = {
            "objective": "regression" if self.task == "regression"
            else "multiclass",
            "metric": {"rmse"} if self.task == "regression"
            else {"multiclass"},
            "num_classes": self.num_classes,
            "boosting": self.boosting,
            "num_leaves": self.num_leaves,
            "max_bin": self.max_bin,
            "learning_rate": self.learning_rate,
            "lambda_l1": self.lambda_l1,
            "lambda_l2": self.lambda_l2,
            # "num_threads": 1,
            # "feature_fraction": 0.9,
            # "bagging_fraction": 0.8,
            # "bagging_freq": 5,
            "verbose": 1,
            # "device_type": "gpu"
        }
        self.evals_result = {}

    def get_metrics(self):
        """Get metrics."""
        d = self.evals_result
        metrics = ddict(list)
        keys = ["training", "valid_1", "valid_2"] \
            if "training" in d \
            else ["valid_0", "valid_1"]
        for metric_name in d[keys[0]]:
            perf = [d[key][metric_name] for key in keys]
            if metric_name in ["regression", "multiclass", "rmse", "l2",
                               "multi_logloss", "binary_logloss"]:
                metrics["loss"] = list(zip(*perf))
            else:
                metrics[metric_name] = list(zip(*perf))
        return metrics

    def get_test_metric(self, metrics, metric_name):
        """Get test metrics."""
        if metric_name == "loss":
            val_epoch = np.argmin([acc[1] for acc in metrics[metric_name]])
        else:
            val_epoch = np.argmax([acc[1] for acc in metrics[metric_name]])
        min_metric = metrics[metric_name][val_epoch]
        return min_metric, val_epoch

    def save_metrics(self, metrics, fn):
        """Save metrics."""
        with open(fn, "w+", encoding="utf-8") as f:
            for key, value in metrics.items():
                print(key, value, file=f)

    def train_val_test_split(self, x, y, train_mask, val_mask, test_mask):
        """Get train/val/test splits."""
        x_train, y_train = x[train_mask], y[train_mask]
        x_val, y_val = x[val_mask], y[val_mask]
        x_test, y_test = x[test_mask], y[test_mask]
        return x_train, y_train, x_val, y_val, x_test, y_test

    def fit(self,
            x, y, train_mask, val_mask, test_mask,
            num_epochs=1000, patience=200,
            loss_fn="", metric_name="loss"):
        """Fit model."""
        x_train, y_train, x_val, y_val, x_test, y_test = \
            self.train_val_test_split(x.numpy(), y.numpy(),
                                      train_mask.numpy(),
                                      val_mask.numpy(),
                                      test_mask.numpy())
        self.num_classes = None if self.task == "regression"\
            else int(y.max()+1)
        self.init_model()

        start = time.time()
        train_data = lightgbm.Dataset(x_train, label=y_train)
        val_data = lightgbm.Dataset(x_val, label=y_val)
        test_data = lightgbm.Dataset(x_test, label=y_test)
        valid_sets = [train_data, val_data, test_data]
        self.model = lightgbm.train(params=self.parameters,
                                    train_set=train_data,
                                    valid_sets=valid_sets,
                                    num_boost_round=num_epochs,
                                    early_stopping_rounds=patience,
                                    evals_result=self.evals_result,
                                    feval=self.r2 if self.task == "regression"
                                    else self.accuracy,
                                    verbose_eval=1)
        finish = time.time()
        print(f"Finished training. Total time: {finish - start:.2f}")

        metrics = self.get_metrics()
        min_metric, min_val_epoch = self.get_test_metric(metrics, metric_name)
        if loss_fn:
            self.save_metrics(metrics, loss_fn)
        print(f"Best {metric_name} at iteration {min_val_epoch}:\
                {min_metric[0]:.3f}/{min_metric[1]:.3f}/{min_metric[2]:.3f}")
        return metrics

    def predict(self, x_test, y_test):
        """Predict."""
        pred = self.model.predict(x_test)

        metrics = {}
        metrics["rmse"] = mean_squared_error(pred, y_test) ** .5

        return metrics
