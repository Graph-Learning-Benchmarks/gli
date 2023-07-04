# GLI Benchmarking on `NodeClassification` Task

The code in this folder can be used to benchmark some popular models on `NodeClassification` task.

## How to run

Example commands to run the code:

```bash
# full batch
python train.py --dataset <dataset> --model GCN
python train.py --dataset <dataset> --model MLP
python train.py --dataset <dataset> --model GAT --model-cfg configs/GAT.yaml
python train.py --dataset <dataset> --model GraphSAGE --model-cfg configs/GraphSAGE.yaml
python train.py --dataset <dataset> --model MoNet --model-cfg configs/MoNet.yaml
python train.py --dataset <dataset> --model MixHop --model-cfg configs/MixHop.yaml
python train.py --dataset <dataset> --model LINKX --model-cfg configs/LINKX.yaml --train-cfg configs/LINKX_train.yaml
python train.py --dataset <dataset> --model TAGCN --model-cfg configs/TAGCN.yaml
python train.py --dataset <dataset> --model GATv2 --model-cfg configs/GATv2.yaml
python train.py --dataset <dataset> --model SGC --model-cfg configs/SGC.yaml
python train.py --dataset <dataset> --model APPNP --model-cfg configs/APPNP.yaml
python train.py --dataset <dataset> --model GCNII --model-cfg configs/GCNII.yaml


# mini batch
python train_minibatch.py --dataset <dataset> --model GCN_minibatch
python train_minibatch.py --dataset <dataset> --model GraphSAGE_minibatch

# GBDT
python train_gbdt.py --dataset <dataset>  --model lightgbm
python train_gbdt.py --dataset <dataset>  --model catboost
```

One can provide a `yaml` file to arguments `--model-cfg` or `--train-cfg` respectively for model configuration or training configuration. If not provided, default configurations (see [model_default.yaml](https://github.com/Graph-Learning-Benchmarks/gli/blob/main/benchmarks/NodeClassification/configs/model_default.yaml) and [train_default.yaml](https://github.com/Graph-Learning-Benchmarks/gli/blob/main/benchmarks/NodeClassification/configs/train_default.yaml)) will be used. 

Note that some models may have unique hyperparameters not included in the default configuration files. In this case, one should pass the model-specific coniguration files to `train.py`.

## Supported models

The following list of models are supported by this benchmark.

### Full batch

- `GCN`
- `MLP`
- `GAT`
- `GraphSAGE`
- `MoNet`
- `MixHop`
- `LINKX`
- `APPNP`
- `GCNII`

### Mini batch

- `GCN_minibatch`
- `GraphSage_minibatch`

### Gradient Boosting Decision Tree (GBDT)

- `catboost`
- `lightgbm`

To add a new model, one should add the model implementation under the `models` folder, and add model specific confgurations under the `configs` folder when needed. We have tried to implement `train.py` in a generic way so one may only need to make minimal modifications to `train.py` and `utils.py`.

Contributions of new models are welcome through pull requests.

## Supported datasets

This benchmark should work for most datasets with a `NodeClassification` task associated. The following datasets have been tested for this code.

- `arxiv-year`
- `cornell`
- `chameleon`
- `citeseer`
- `penn94`
- `wiki`
- `wisconsin`
- `ogbn-arxiv`
- `cora`
- `snap-patents`
- `pokec`
- `ogbn-products`
- `texas`
- `ogbn-mag`
- `pubmed`
- `twitch-gamers`
- `ogbn-proteins`
- `actor`
- `squirrel`
- `genius`
