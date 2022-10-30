# arXiv-year

## Dataset Description
arXiv-year is the ogbn-arXiv network with different labels. This dataset was originally created by Wang<sup>[1](#myfootnote1)</sup> and transformed by Hu<sup>[2](#myfootnote2)</sup> as well as Lim<sup>[3](#myfootnote3)</sup>. The difference between this dataset and ogbn-arXiv is that the class labels are set to be the year that the paper is posted, instead of paper subject area.  The nodes are arXiv papers, and directed edges connect a paper to other papers that it cites. The node features are averaged word2vec token features of both the title and abstract of the paper. The five classes are chosen by partitioning the posting dates so that class ratios are approximately balanced.

Statistics:
- Nodes: 169343
- Edges: 1166243
- Number of Classes: 5

<a name="myfootnote1">[1]</a>: Kuansan Wang, Zhihong Shen, Chiyuan Huang, Chieh-Han Wu, Yuxiao Dong, and Anshul Kanakia. Microsoft academic graph: When experts are not enough. Quantitative Science Studies, 1(1):396–413, 2020.

<a name="myfootnote2">[2]</a>: Hu, Weihua, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, and Jure Leskovec. "Open graph benchmark: Datasets for machine learning on graphs." Advances in neural information processing systems 33 (2020): 22118-22133.

<a name="myfootnote3">[3]</a>: Lim, Derek, Felix Hohne, Xiuyu Li, Sijia Linda Huang, Vaishnavi Gupta, Omkar Bhalerao, and Ser Nam Lim. "Large scale learning on non-homophilous graphs: New benchmarks and strong simple methods." Advances in Neural Information Processing Systems 34 (2021): 20887-20902.
#### Citation
- Original Source
  + [Website](https://www.microsoft.com/en-us/research/project/microsoft-academic-graph/)
  + LICENSE: missing
  ```
  @article{10.1162/qss_a_00021,
    author = {Wang, Kuansan and Shen, Zhihong and Huang, Chiyuan and Wu, Chieh-Han and Dong, Yuxiao and Kanakia, Anshul},
    title = "{Microsoft Academic Graph: When experts are not enough}",
    journal = {Quantitative Science Studies},
    volume = {1},
    number = {1},
    pages = {396-413},
    year = {2020},
    month = {02},
    issn = {2641-3337},
    doi = {10.1162/qss_a_00021},
    url = {https://doi.org/10.1162/qss\_a\_00021},
    eprint = {https://direct.mit.edu/qss/article-pdf/1/1/396/1760880/qss\_a\_00021.pdf},
  }
  ```
  ```
  @inproceedings{sinha2015overview,
    title={An overview of microsoft academic service (mas) and applications},
    author={Sinha, Arnab and Shen, Zhihong and Song, Yang and Ma, Hao and Eide, Darrin and Hsu, Bo-June and Wang, Kuansan},
    booktitle={Proceedings of the 24th international conference on world wide web},
    pages={243--246},
    year={2015}
  }
  ```

- Current Version
  + [Website](https://github.com/CUAI/Non-Homophily-Large-Scale)
  + LICENSE: missing
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

- Previous Version
  + [Website](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)
  + LICENSE: [ODC-BY](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv)
  ```
  @article{hu2020open,
    title={Open graph benchmark: Datasets for machine learning on graphs},
    author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
    journal={Advances in neural information processing systems},
    volume={33},
    pages={22118--22133},
    year={2020}
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
The data file in GLI format is transformed from the [CUAI](https://github.com/CUAI/Non-Homophily-Large-Scale). Check [Non-homo-datasets](https://github.com/GreatSnoopyMe/Non-homo-datasets) for the preprocessing.


### Requirements

The preprocessing code requires the following packages.

```
dataset==1.5.2
numpy==1.22.3
scipy==1.7.3
torch==1.11.0
```
