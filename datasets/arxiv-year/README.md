# arXiv-year

## Dataset Description
arXiv-year is the ogbn-arXiv network with different labels. This dataset was originally created by Wang<sup>[1](#myfootnote1)</sup> and transformed by Hu<sup>[2](#myfootnote2)</sup> as well as Lim<sup>[3](#myfootnote3)</sup>.
 The difference between this dataset and ogbn-arXiv is that the class labels are set to be the year that the paper is posted, instead of paper subject area.  The nodes are arXiv papers, and directed edges connect a paper to other papers that it cites. The node features are averaged word2vec token features of both the title and abstract of the paper. The five classes are chosen by partitioning the posting dates so that class ratios are approximately balanced.

Statistics:
- Nodes: 169343
- Edges: 1166243
- Number of Classes: 5

#### Citation
<a name="myfootnote1">[1]</a>: Kuansan Wang, Zhihong Shen, Chiyuan Huang, Chieh-Han Wu, Yuxiao Dong, and Anshul Kanakia. Microsoft academic graph: When experts are not enough. Quantitative Science Studies, 1(1):396–413, 2020.

<a name="myfootnote2">[2]</a>: Hu, Weihua, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. "Open graph benchmark: Datasets for machine learning on graphs." Advances in neural information processing systems 33 (2020): 22118-22133.

<a name="myfootnote3">[3]</a>: [3] Lim, Derek, Felix Hohne, Xiuyu Li, Sijia Linda Huang, Vaishnavi Gupta, Omkar Bhalerao, and Ser Nam Lim. "Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods." Advances in Neural Information Processing Systems 34 (2021): 20887-20902.

```
@article{lim2021large,
  title={Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods},
  author={Lim, Derek and Hohne, Felix and Li, Xiuyu and Huang, Sijia Linda and Gupta, Vaishnavi and Bhalerao, Omkar and Lim, Ser Nam},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={20887--20902},
  year={2021}
}
```
## Available Tasks

### arXiv-year

- Task type: `NodeClassification`


#### Citation

```
@article{lim2021large,
  title={Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods},
  author={Lim, Derek and Hohne, Felix and Li, Xiuyu and Huang, Sijia Linda and Gupta, Vaishnavi and Bhalerao, Omkar and Lim, Ser Nam},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={20887--20902},
  year={2021}
}
```

## Preprocessing
The data file in GLB format is transformed from the [CUAI](https://github.com/CUAI/Non-Homophily-Large-Scale). Check [Non-homo-datasets](https://github.com/GreatSnoopyMe/Non-homo-datasets) for the preprocessing.


### Requirements

The preprocessing code requires the following packages.

```
dataset==1.5.2
numpy==1.22.3
scipy==1.7.3
torch==1.11.0
```
