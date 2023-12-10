# MUTAG

## Dataset Description

MUTAG is a collection of nitroaromatic compounds. In this dataset, a single large heterogenous graph is used to represent various chemical compounds where Vertices represent atoms and edges represent bonds between corresponding atoms.

The goal is to predict the mutagenicity of a given compound on Salmonella Typhimurium, which is denoted by the attribute 'label' on nodes of type 'd'.

## Statistics 
- Nodes: 27163
- Edges: 148100 (including reverse edges)
- Target Category: d
- Number of Classes: 2


## Citation

`@article{Debnath1991StructureactivityRO,
  title={Structure-activity relationship of mutagenic aromatic and heteroaromatic nitro compounds. Correlation with molecular orbital energies and hydrophobicity.},
  author={Asim Kumar Debnath and R L Compadre and Gargi Debnath and Alan J. Shusterman and Corwin Hansch},
  journal={Journal of medicinal chemistry},
  year={1991},
  volume={34 2},
  pages={
          786-97
        },
  url={https://api.semanticscholar.org/CorpusID:19990980}
}` 

## Preprocessing
The data files and task config file in GLI format are transformed from the [DGL implementation](https://docs.dgl.ai/generated/dgl.data.MUTAGDataset.html#dgl.data.MUTAGDataset) Check MUTAG.ipynb for the preprocessing.

