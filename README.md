# Devil in the Tail: A Multi-Modal Framework for Drug-Drug Interactino Prediction In Long Tail Distinction


This instructoin aims to help the reproduction of the result. The file provided are as follow:


# Environment set-up
The following specific environment needs to be installed. The model was runned on Unbuntu 22.04.2.

```bash
    conda create --name tfmd python==3.9
    conda activate tfmd
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
    pip install -r requirements.txt
    conda install gcc_linux-64 gxx_linux-64 mpi4py
    python -m pip install git+https://github.com/MolecularAI/pysmilesutils.git
```


# Downloan dataset
We offer five preprocessed and ready-to-use dataset in [Google Drive](https://drive.google.com/drive/folders/1iMUh6sIuXTA9zA-TedHRMFIm0OB72-vs?usp=sharing). The raw data of DDIMDL and MUFFIN can be download from their github repository [DDIMDL](https://github.com/YifanDengWHU/DDIMDL) and [MUFFIN](https://github.com/xzenglab/MUFFIN). We do not offer raw data of DrugBank as it is required to retrieve official authorization from DrugBank officials. See details [here](https://go.drugbank.com/releases/latest). The preprocessed scripts will be release soon.

# Retrieve Drugbank data and Cleaning
We construct our dataset DBDDI-110 and DBDDI171 from drugbank raw dataset. We are working on cleaning the scraping scripts of the preprocessing drugbank dataset. It should be released soon.   ------- 21/07/2024


# Data Preparation
Onece you download the dataset from above, you will have 2 diretory `models` and `dataset`. `dataset` contains ready-to-use files to reproduce our results. **Move `dataset` under `TFMD/`** to run the scripts. You can prepare you own datasset by constructin files in the following format:

The data should fomated as csv file as shown in the following example:

For drug feature file, named as ***features.csv***:
|drugs_id | Smiles | Targets | Enzymes|
| :---: | :---: | :---: | :---: | 
| DB00122 | C[N+](C)(C)CCO | P36544\|Q9Y5K3\|P22303\|P49585\|O14939\|P06276\|Q13393\|Q8TCT1 | Q9Y6K0\|Q9Y259\|P28329\|P35790\|Q8NE62|


For drug-drug interaction file ***ddi.csv***:

| id1 | name1 | id2 | name2 | interaction|
| :---: | :---: | :---: | :---: | :---: | 
| DB06605 | Apixaban | DB00006 | Bivalirudin | Drug A may increase the anticoagulant activities of Drug B. |


For Graph embedding files DBDDI_171_drugname_smiles.txt for example:

```txt
Compound::DB00122\tC[N+](C)(C)CCO\n
```


# Data preprocessing
The Pre-trained models for feature extraction are publically available:

* **Chemformer**: https://github.com/MolecularAI/MolBART.git for SMILES Sequential embedding.
* **Pretrained Graph**: https://lifesci.dgl.ai/api/model.pretrain.html for SMILES graph embedding.


### SMILES Sequential Embedding 
For sequential embedding, the pretrained model is originally avaiable in [here](https://github.com/MolecularAI/MolBART.git). But they seems delete it for some reasons, you can download the pretrained weight from our google drive. we use the pretrained-large model of Molbart `models/pre-trained/combined-large/step=1000000.ckpt` and utilze only the encoder part. 

Once you download the `models` from google drive. Place the `models` diretory under `Sequential_Embeddings/molbart/` and run

```bash
    python -u SMILES_Embedding.py
```


### SMIELS Graph Embedding
Run `pretrain_smiles_embedding.py` in TFMD repository as it is well-debugged to our dataset. The pre-trained Graph model should be able to download automatically once the script is executed.

```bash
    python -u pretrain_smiles_embedding.py
```

See https://github.com/xzenglab/MUFFIN for detailed steps and necessary file for your customized dataset 



# Run Training
To run the model, you need to navigate to TFMD folder and run in terminal:
```bash
    python -u main.py
```
