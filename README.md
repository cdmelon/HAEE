# HAEE
Code and datasets for the ISWC 2023 paper "HAEE: Low-Resource Event Detection with Hierarchy-Aware Event Graph Embeddings"

## Data File Structure
The structure of data files is as follows: 

```
pythonProject
 |-- ModelData  # data
 |    |-- event_dict_train_data.json  #  OntoEvent training set
 |    |-- event_dict_valid_data.json  #  OntoEvent validation set
 |    |-- event_dict_test_data.json  #  OntoEvent testing set
 |    |-- event_relation.json  #  OntoEvent relation set
 |    |-- maven_train.json  #  MAVEN-Few training set
 |    |-- maven_test.json  #  MAVEN-Few testing set
 |    |-- relation.json  #  MAVEN-Few relation set
 |-- TestData  # cache data
 |    |-- event_map
 |    |-- rel_event_ids
 |    |-- rel_example_ids
 |-- data_utils.py  # process data
 |-- model.py  # HAEE model
 |-- run_model.py
```

## Requirements

- python==3.6.9

- scikit-learn==0.20.2

- torch==1.8.0

- transformers==2.8.0

## Usage

**Running Model**:

Code is currently in development mode, hyperparameters need to be modified in the code.

If keep the current configuration, HAEE will run on the OntoEvent in overall evaluation.

```
python run_model.py 
```

To run MAVEN-Few dataset, should modified corresponding code in ```data_utils.py``` 
```python
def get_train_examples(self, data_dir):
    logger.info("LOOKING AT {} train".format(data_dir))
    return self.create_examples(os.path.join(data_dir, 'event_dict_train_data.json'), DOC_TYPE_TRAIN, DATA_TYPE_ONTOED)
    # return self.create_examples(os.path.join(data_dir, 'maven_train.json'), DOC_TYPE_TRAIN, DATA_TYPE_NEW)
```


**Hint**

- Modify ```--model_name_or_path``` parameter to decide whether to use offline bert files.
- Explicit labeling of the CUDA device is required in our operating environment. Please decide whether to keep this code based on your needs. This code section is located at the top of the ```run_model.py``` file.
```os.environ["CUDA_VISIBLE_DEVICES"] = "0"```

**hyperparameters**

```python
self.ratio_proto_emb = 0.4 #  convolution weight α
self.margin = 0.08  #  modulus threshold γ
self.r_gamma = 8  #  rotation threshold λ
self.emb = 100  #  the dimension of event embedding
self.loss_scale = nn.Parameter(torch.tensor([-0.5] * 3).to(device))  #  uncertainty values
```
These hyperparameters can be found in the ```model.py``` file.

**Low-resource Evaluation**

To reproduce the result in paper, can add the following code in ```data_utils``` in ```create_examples``` method.
```python
turn += 1
if turn % {2|4|10} != 1 and doc_type == DOC_TYPE_TRAIN:  #  50%|25%|10%
    continue
```

**How to Cite**

Thank you very much for your interest in our work. If you use or extend our work, please cite the following paper:
```python
@InProceedings{ISWC2024_HAEE,
title="HAEE: Low-Resource Event Detection with Hierarchy-Aware Event Graph Embeddings",
author="Ding, Guoxuan
and Guo, Xiaobo
and Chen, Gaode
and Wang, Lei
and Zha, Daren",
booktitle="The Semantic Web -- ISWC 2023",
year="2023",
publisher="Springer Nature Switzerland",
address="Cham",
pages="61--79",
}
```
