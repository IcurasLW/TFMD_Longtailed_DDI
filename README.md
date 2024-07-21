# Devil in the Tail: A Multi-Modal Framework for Drug-Drug Interactino Prediction In Long Tail Distinction


This instructoin aims to help the reproduction of the result. The file provided are as follow:

* **main**: main file to train and test
* **classifier**: model architecture
* **losses**: loss function definition including Tailed Focal Loss and Focal Loss
* **utils**: some helper functions
* **dataset**: Only DDIMDL and MUFFIN dataset are provided for reviewing purpose, which the raw data are publically in [DDIMDL](https://github.com/YifanDengWHU/DDIMDL) and [MUFFIN](https://github.com/xzenglab/MUFFIN). An official authorization from DrugBank Official is required to retrieve the dataset. We will not provide the raw data of DrugBank, See details [here](https://go.drugbank.com/releases/latest). The preprocessed files and scripts will be given.



graph and SMILES data have embedding have been preprocessed named as "graph_embedding.npy" and "SMILES_embedding.pt" in dataset folder, the embedding script will be shared on Github repository upon the paper acceptance since it excceeds the CMT file limitation.

# Environment set-up

The following specific environment needs to be installed. The model was runned on Unbuntu 22.04.2.

```bash
    conda create --name tfmd python==3.9
    conda activate tfmd
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install -r requirements.txt
    conda install gcc_linux-64 gxx_linux-64 mpi4py
```


# Data Preparation
The raw data of dataset are publically available in the following address. DB-DDI110 and DB-DDI171 are extracted from full Drugbank Dataset due to the missing modalities, which is our future work. The data preprocessing and extraction scripts and our processed data will be offered upon acceptance for reproduction purpose. 

* **DDIMDL** : https://github.com/YifanDengWHU/DDIMDL 
* **MUFFIN** : https://github.com/xzenglab/MUFFIN
* **DrugBank** : https://go.drugbank.com/releases/latest



The data should fomated as csv file as shown in the following example:

For drug feature file, named as ***features.csv***:
|drugs_id | Smiles | Targets | Enzymes|
| :---: | :---: | :---: | :---: | 
| DB00122 | C[N+](C)(C)CCO | P36544\|Q9Y5K3\|P22303\|P49585\|O14939\|P06276\|Q13393\|Q8TCT1 | Q9Y6K0\|Q9Y259\|P28329\|P35790\|Q8NE62|



For drug-drug interaction file ***ddi.csv***:


| id1 | name1 | id2 | name2 | interaction|
| :---: | :---: | :---: | :---: | :---: | 
| DB06605 | Apixaban | DB00006 | Bivalirudin | Drug A may increase the anticoagulant activities of Drug B. |


Place ***features.csv*** and ***ddi.csv*** into ./dataset folder



# Feature Extraction from Pre-trained Model
The Pre-trained models for feature extraction are publically available:

* **Chemformer**: https://github.com/MolecularAI/Chemformer for SMILES string embedding.
* **Pretrained Graph**: https://lifesci.dgl.ai/api/model.pretrain.html for SMILES graph embedding.



### SMILES Sequential Embedding 
Place the models `.ckpt` file and the corresponding corpus file `bart_vocab.txt` downloaded from MoLBARAT diretory into `./SMILES_Embeddings/models` and `./SMILES_Embeddings/`. The `models` directory should contains 3 sub-directories: `fined-tuned`, `pre-trained`, `rand_init_downstream`. The experimental results can be reproduced by `./SMILES_Embeddings/models/pre-trained/combined-large/step=1000000.ckpt`

```bash
    python -u SMILES_Embedding.py
```


### SMIELS Graph Embedding
See https://github.com/xzenglab/MUFFIN for detailed steps and necessary file to retrieve the embedding of graph for your customized dataset and run `pretrain_smiles_embedding.py` in TFMD repository as it is well-debugged. The pre-trained Graph model should be able to download automatically once the script is executed.


```bash
    python -u pretrain_smiles_embedding.py
```



# Run Training
To run the model, you need to navigate to TFMD folder and run in terminal:
```bash
    python -u main.py
```
