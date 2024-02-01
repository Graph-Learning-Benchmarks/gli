# ARXIV-2023

## Dataset Description

arxiv-2023 is collected to be compared with ogbn-arxiv. Both datasets represent directed citation networks where each node corresponds to a paper published on arXiv and each edge indicates one paper citing another.

Statistics:
- Nodes: 33868
- Edges: 305672
- Number of Classes: 40

#### Citation

- Original Source
	+ [Website](https://github.com/TRAIS-Lab/LLM-Structured-Data)
	+ LICENSE: [<license type>](<URL to license>)



```
@misc{huang2023llms,
      title={Can LLMs Effectively Leverage Graph Structural Information: When and Why}, 
      author={Jin Huang and Xingjian Zhang and Qiaozhu Mei and Jiaqi Ma},
      year={2023},
      eprint={2309.16595},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

- Current Version
	+ [Website](https://github.com/TRAIS-Lab/LLM-Structured-Data)
	+ LICENSE: [<license type>](<URL to license>)



```
@misc{huang2023llms,
      title={Can LLMs Effectively Leverage Graph Structural Information: When and Why}, 
      author={Jin Huang and Xingjian Zhang and Qiaozhu Mei and Jiaqi Ma},
      year={2023},
      eprint={2309.16595},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

- Previous Version
	+ [Website](<URL to website>)
	+ LICENSE: [<license type>](<URL to license>)


## Available Tasks

### <Task Name>



- Task type: `NodeClassification`


#### Citation

```
@misc{huang2023llms,
      title={Can LLMs Effectively Leverage Graph Structural Information: When and Why}, 
      author={Jin Huang and Xingjian Zhang and Qiaozhu Mei and Jiaqi Ma},
      year={2023},
      eprint={2309.16595},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

<!-- Insert the BibTeX citation into the above code block. -->

## Preprocessing

The data files and task config file in GLI format are transformed in arxiv-2023.ipynb file. Raw data aquried in TRAIS-Lab/LLM-Structured-Data folder.

### Requirements

```
openai
pytorch
PyG
ogb
```


