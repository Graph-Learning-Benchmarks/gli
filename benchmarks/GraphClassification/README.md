# GLI benchmarking on Graph Classification

Available datasets: `mnist`, `ogbg-molpcba`, `ogbg-molhiv`, `ogbg-molsider`, `ogbg-molbace`, `ogbg-molmuv`, `cifar`, `ogbg-molclintox`

## How to run

For each model, run with the following: 

```bash
python train.py --dataset <dataset> --model GCN 
python train.py --dataset <dataset> --model DGN --model-cfg configs/DGN.yaml
python train.py --dataset <dataset> --model ChebNet --model-cfg configs/ChebNet.yaml
```

One can provide a `yaml` file for model configuration and training configuration. If not provided, default configurations will be used. See [model_default.yaml](https://github.com/Graph-Learning-Benchmarks/gli/blob/main/benchmarks/GraphClassification/configs/model_default.yaml) and [train_default.yaml](https://github.com/Graph-Learning-Benchmarks/gli/blob/main/benchmarks/GraphClassification/configs/train_default.yaml)

For example, to train a GCN model on `ogbg-molhiv`:

```
python train.py --dataset ogbg-molhiv --model GCN 
```

