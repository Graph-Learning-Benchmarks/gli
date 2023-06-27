# GLI Benchmarking on `GraphClassification` Task

The code in this folder can be used to benchmark some popular models on `GraphClassification` task.

## How to run

Example commands to run the code:

```bash
python train.py --dataset <dataset> --model GCN 
python train.py --dataset <dataset> --model DGN --model-cfg configs/DGN.yaml
python train.py --dataset <dataset> --model ChebNet --model-cfg configs/ChebNet.yaml
python train.py --dataset <dataset> --model GIN --model-cfg configs/GIN.yaml
```

One can provide a `yaml` file to arguments `--model-cfg` or `--train-cfg` respectively for model configuration or training configuration. If not provided, default configurations (see [model_default.yaml](https://github.com/Graph-Learning-Benchmarks/gli/blob/main/benchmarks/GraphClassification/configs/model_default.yaml) and [train_default.yaml](https://github.com/Graph-Learning-Benchmarks/gli/blob/main/benchmarks/GraphClassification/configs/train_default.yaml)) will be used. 

Note that some models may have unique hyperparameters not included in the default configuration files. In this case, one should pass the model-specific coniguration files to `train.py`.

## Supported models

The following list of models are supported by this benchmark. 

- `GCN`
- `DGN`
- `ChebNet`
- `GIN`

To add a new model, one should add the model implementation under the `models` folder, and add model specific confgurations under the `configs` folder when needed. We have tried to implement `train.py` in a generic way so one may only need to make minimal modifications to `train.py` and `utils.py`.

Contributions of new models are welcome through pull requests. 

## Supported datasets

This benchmark should work for most datasets with a `GraphClassification` task associated. The following datasets have been tested for this code. 

- `mnist`
- `ogbg-molpcba`
- `ogbg-molhiv`
- `ogbg-molsider`
- `ogbg-molbace`
- `ogbg-molmuv`
- `cifar`
- `ogbg-molclintox`
