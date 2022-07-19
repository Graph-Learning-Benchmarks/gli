# VesselGraph / ogbl-BALBc1

## Dataset Description
BALBc1, or BALBc_no1, is a dataset of *VesselGraph*. It is a large, spatial and structured graph of whole-brain vessels, where bifurcation points are nodes, and blood vessels are edges. Node data include the xyz coordinates and a flag indicating whether the bifurcation point is at sample border. Edge data include properties regarding the radius and length of the blood vessel.

Currently, VesselGraph is not officially included in the OGB dataset yet. The original data needs to be loaded using a custom version of `ogb`. Please see the [jocpae/VesselGraph](https://github.com/jocpae/VesselGraph) repo for details.

Statistics:
- nodes 3,538,495
- edges 5,345,897

### Citation
```bibtex
@misc{paetzold2021brain,
      title={Whole Brain Vessel Graphs: A Dataset and Benchmark for Graph Learning and Neuroscience (VesselGraph)}, 
      author={Johannes C. Paetzold and Julian McGinnis and Suprosanna Shit and Ivan Ezhov and Paul Büschl and Chinmay Prabhakar and Mihail I. Todorov and Anjany Sekuboyina and Georgios Kaissis and Ali Ertürk and Stephan Günnemann and Bjoern H. Menze},
      year={2021},
      eprint={2108.13233},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Available Tasks
OGB
- Task type: `RelationLinkPrediction`

## Requirements
```
numpy
ogb
```