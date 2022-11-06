# GLI benchmarking on Node Classification

Available datasets: `arxiv-year`, `cornell`, `chameleon`, `citeseer`, `penn94`, `wiki`, `wisconsin`, `ogbn-arxiv`, `cora`, `snap-patents`, `pokec`, `ogbn-products`, `texas`, `ogbn-mag`, `pubmed`, `twitch-gamers`, `ogbn-proteins`, `actor`, `squirrel`, `genius`.

## Graph Neural Network (full batch)

Available models: `GCN`, `MLP`, `GAT`,`GraphSAGE`, `MoNet`, `MixHop`, `LINKX`.

For each model, run with the following: 

```bash
python train.py --dataset <dataset> --model GCN
python train.py --dataset <dataset> --model MLP
python train.py --dataset <dataset> --model GAT --model-cfg configs/GAT.yaml
python train.py --dataset <dataset> --model GraphSAGE --model-cfg configs/GraphSAGE.yaml
python train.py --dataset <dataset> --model MoNet --model-cfg configs/MoNet.yaml
python train.py --dataset <dataset> --model MixHop --model-cfg configs/MixHop.yaml
python train.py --dataset <dataset> --model LINKX --model-cfg configs/LINKX.yaml --train-cfg configs/LINKX_train.yaml
```

One can provide a `yaml` file for model configuration and training configuration. If not provided, default configurations will be used. See [model_default.yaml](https://github.com/Graph-Learning-Benchmarks/gli/blob/main/benchmarks/NodeClassification/configs/model_default.yaml) and [train_default.yaml](https://github.com/Graph-Learning-Benchmarks/gli/blob/main/benchmarks/NodeClassification/configs/train_default.yaml)

For example, to train a GCN model on `cora`:

```bash
python train.py --dataset cora --model GCN
```

---

## Graph Neural Network (mini batch)

```bash
python train_minibatch.py --dataset <dataset> --model GCN_minibatch
```

## Gradient Boosting Decision Tree (GBDT)

Available models: `lightgbm`, `catboost`.

```bash
python train_gbdt.py --dataset <dataset>  --model lightgbm
python train_gbdt.py --dataset <dataset>  --model catboost
```
