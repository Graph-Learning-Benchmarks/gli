"""
Train for GBDT.

Reference:
https://github.com/nd7141/bgnn/blob/master/scripts/run.py
"""
import os
import re
import json
import time
import datetime
from collections import defaultdict as ddict

import numpy as np
from sklearn.model_selection import ParameterGrid

import gli
from utils import parse_args, set_seed, \
                  load_config_file, check_multiple_split
from gli.utils import to_dense
from models.gbdt import GBDTCatBoost, GBDTLGBM


class RunModel:
    """Model class for gbdt."""

    def __init__(self, args, model_cfg, train_cfg):
        """Initiate model."""
        self.args = args
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

    def read_input(self):
        """Read input."""
        data = gli.dataloading.get_gli_dataset(self.args.dataset,
                                               "NodeClassification")
        g = data[0]
        g = to_dense(g)
        feature_name = re.search(r".*Node/(\w+)", data.features[0]).group(1)
        label_name = re.search(r".*Node/(\w+)", data.target).group(1)
        features = g.ndata[feature_name]
        labels = g.ndata[label_name].squeeze()
        train_mask = g.ndata["train_mask"]
        val_mask = g.ndata["val_mask"]
        test_mask = g.ndata["test_mask"]
        # for multi-split dataset, choose 0-th split for now
        if check_multiple_split(self.args.dataset):
            train_mask = train_mask[:, 0]
            val_mask = val_mask[:, 0]
            test_mask = test_mask[:, 0]

        # When labels contains -1, modify masks
        if labels.min() < 0:
            train_mask = train_mask * (labels >= 0)
            val_mask = val_mask * (labels >= 0)
            test_mask = test_mask * (labels >= 0)

        self.x = features
        self.y = labels

        self.masks = {"0": {"train": train_mask,
                            "val": val_mask,
                            "test": test_mask}}

    def get_input(self):
        """Get input."""
        if self.save_folder is None:
            self.save_folder = f"grid/gdbt_results/{self.args.dataset}/\
                                {datetime.datetime.now().strftime('%d_%m')}"

        self.read_input()
        print("Save to folder:", self.save_folder)

    def run_one_model(self, config_fn, model_name):
        """Run single model."""
        print(config_fn)
        # self.config = OmegaConf.load(config_fn)
        # print(type(self.config))
        # print(self.config)
        # grid = ParameterGrid(dict(self.config.hp))

        self.config = load_config_file(config_fn)
        print(type(self.config))
        print(self.config)
        grid = ParameterGrid(self.config["hp"])

        for ps in grid:
            print("hyper params: ", ps)
            param_string = "".join([f"-{key}{ps[key]}" for key in ps])
            exp_name = f"{model_name}{param_string}"
            print(f"\nSeed {self.seed} RUNNING:{exp_name}")

            runs = []
            runs_custom = []
            times = []
            for _ in range(self.repeat_exp):
                start = time.time()
                model = self.define_model(model_name, ps)

                inputs = {"x": self.x, "y": self.y,
                          "train_mask": self.train_mask,
                          "val_mask": self.val_mask,
                          "test_mask": self.test_mask}

                metrics = model.fit(num_epochs=self.config["num_epochs"],
                                    patience=self.config["patience"],
                                    loss_fn=f"{self.seed_folder}/\
                                              {exp_name}.txt",
                                    metric_name="loss"
                                    if self.task == "regression"
                                    else "accuracy", **inputs)
                finish = time.time()
                best_loss = min(metrics["loss"], key=lambda x: x[1])
                best_custom = max(metrics["r2" if self.task == "regression"
                                  else "accuracy"],
                                  key=lambda x: x[1])
                runs.append(best_loss)
                runs_custom.append(best_custom)
                times.append(finish - start)
            self.store_results[exp_name] = (list(map(np.mean,
                                                     zip(*runs))),
                                            list(map(np.mean,
                                                     zip(*runs_custom))),
                                            np.mean(times))

    def define_model(self, model_name, ps):
        """Define model."""
        if model_name == "catboost":
            return GBDTCatBoost(self.task, **ps)
        elif model_name == "lightgbm":
            return GBDTLGBM(self.task, **ps)

    def create_save_folder(self, seed):
        """Create folder to save output."""
        self.seed_folder = f"{self.save_folder}/{seed}"
        os.makedirs(self.seed_folder, exist_ok=True)

    def split_masks(self, seed):
        """Split masks."""
        self.train_mask = self.masks[seed]["train"]
        self.val_mask = self.masks[seed]["val"]
        self.test_mask = self.masks[seed]["test"]

    def save_results(self, seed):
        """Save results."""
        self.seed_results[seed] = self.store_results
        with open(f"{self.save_folder}/seed_results.json", "w+",
                  encoding="utf-8") as f:
            json.dump(self.seed_results, f)

        self.aggregated = self.aggregate_results()
        with open(f"{self.save_folder}/aggregated_results.json", "w+",
                  encoding="utf-8") as f:
            json.dump(self.aggregated, f)

    def aggregate_results(self):
        """Aggregate results."""
        model_best_score = ddict(list)
        model_best_time = ddict(list)

        results = self.seed_results
        # print("results:", results)
        for seed_tuple in results.items():
            # print("seed_tuple", seed_tuple)
            # print("seed_tuple[1]", seed_tuple[1])
            model_results_for_seed = ddict(list)
            for _, output in seed_tuple[1].items():
                model_name = self.args.model
                if self.task == "regression":  # rmse metric
                    val_metric, test_metric = output[0][1], output[0][2]
                    cur_time = output[2]
                else:  # accuracy metric
                    val_metric, test_metric = output[1][1], output[1][2]
                    cur_time = output[2]
                model_results_for_seed[model_name].append((val_metric,
                                                           test_metric,
                                                           cur_time))

            for model_name, model_results in model_results_for_seed.items():
                if self.task == "regression":
                    best_result = min(model_results)  # rmse
                else:
                    best_result = max(model_results)  # accuracy
                model_best_score[model_name].append(best_result[1])
                model_best_time[model_name].append(best_result[2])

        aggregated = {}
        for model, scores in model_best_score.items():
            aggregated[model] = (np.mean(scores), np.std(scores),
                                 np.mean(model_best_time[model]),
                                 np.std(model_best_time[model]))
        return aggregated

    def run(self,
            save_folder: str = None,
            task: str = "NodeClassification",
            repeat_exp: int = 1,
            max_seeds: int = 5,
            ):
        """Run the model."""
        start2run = time.time()
        self.repeat_exp = repeat_exp
        self.max_seeds = max_seeds
        print(self.args.dataset, task, repeat_exp, max_seeds)

        self.task = task
        self.save_folder = save_folder
        self.get_input()

        self.seed_results = {}
        for ix, seed in enumerate(self.masks):
            print(f"{self.args.dataset} Seed {seed}")
            self.seed = seed

            self.create_save_folder(seed)
            self.split_masks(seed)

            self.store_results = {}
            self.run_one_model("configs/" + self.args.model + ".yaml",
                               self.args.model)

            self.save_results(seed)
            if ix+1 >= max_seeds:
                break

        print(f"Finished {self.args.dataset}: {time.time() - start2run} sec.")


def main():
    """Load dataset and train the model."""
    # Load cmd line args
    args = parse_args()
    print(args)
    # Load config file
    model_cfg = load_config_file(args.model_cfg)
    train_cfg = load_config_file(args.train_cfg)
    set_seed(train_cfg["seed"])

    RunModel(args, model_cfg, train_cfg).run()


if __name__ == "__main__":
    main()
