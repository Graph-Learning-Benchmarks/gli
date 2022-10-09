# Twitch-ENGB

## Dataset Description
These datasets used for node classification and transfer learning are Twitch user-user networks of gamers who stream in a certain language. Nodes are the users themselves and the links are mutual friendships between them. Vertex features are extracted based on the games played and liked, location and streaming habits. Datasets share the same set of node features, this makes transfer learning across networks possible. These social networks were collected in May 2018. The supervised task related to these networks is binary node classification - one has to predict whether a streamer uses explicit language.

Statistics:
- Nodes: 7126
- Edges: 35324
- Number of Classes: 2

#### Citation
- Original Source
  ```
  @misc{rozemberczki2019multiscale,    
       title = {Multi-scale Attributed Node Embedding},   
       author = {Benedek Rozemberczki and Carl Allen and Rik Sarkar},   
       year = {2019},   
       eprint = {1909.13021},  
       archivePrefix = {arXiv},  
       primaryClass = {cs.LG}   
       }
  ```

## Available Tasks


- Task type: `NodeClassification`, `LinkPrediction`



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
The data files in GLI format are transformed from https://github.com/benedekrozemberczki/MUSAE. Check preprocess.py for the preprocessing.



### Requirements

The preprocessing code requires the following packages.

```
numpy==1.22.3
scipy==1.7.3
torch==1.11.0
networkx==2.6.3
csv==1.0
json==2.0.9
```
