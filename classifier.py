import torch
import torch.nn as nn
from tqdm import trange
from losses import *
from utils import *

torch.autograd.set_detect_anomaly(True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float32)
torch.cuda.manual_seed_all(0)

class Classifier_NN(nn.Module):
    def __init__(
                self,
                drug_features:object,
                num_classes,
                dropout=0.5,
                ):
        super().__init__()
        hidden_size_1 = 2048
        hidden_size_2 = 1024
        hidden_size_3 = 512
        hidden_size_4 = 256
        self.drug_features = drug_features
        global_input_size = 0
        self.w_graph = nn.Linear(in_features=300, out_features=1024)
        
        
        if drug_features.smiles_features != None:
            smiles_input_size = drug_features.smiles_features.shape[1]
            global_input_size += smiles_input_size


            self.smiles_block_1 = nn.Sequential(nn.Linear(in_features=smiles_input_size, out_features=hidden_size_1),
                                                nn.BatchNorm1d(hidden_size_1),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
                                                nn.BatchNorm1d(hidden_size_2),
                                                nn.ReLU(inplace=False))

            
            self.smiles_block_2 = nn.Sequential(nn.Linear(in_features=hidden_size_2, out_features=hidden_size_2),
                                                nn.BatchNorm1d(hidden_size_2),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(in_features=hidden_size_2, out_features=hidden_size_3),
                                                nn.BatchNorm1d(hidden_size_3),
                                                nn.ReLU(inplace=False))

            
            self.smiles_block_3 = nn.Sequential(nn.Linear(in_features=hidden_size_3, out_features=hidden_size_3),
                                                nn.BatchNorm1d(hidden_size_3),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(in_features=hidden_size_3, out_features=hidden_size_4),
                                                nn.BatchNorm1d(hidden_size_4),
                                                nn.ReLU(inplace=False)
                                                )
        
        
        if drug_features.graph_features != None:
            self.graph_block_1 = nn.Sequential(nn.Linear(in_features=1024 + smiles_input_size, out_features=hidden_size_1),
                                                nn.BatchNorm1d(hidden_size_1),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
                                                nn.BatchNorm1d(hidden_size_2),
                                                nn.ReLU(inplace=False))

            self.graph_block_2 = nn.Sequential(nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
                                                nn.BatchNorm1d(hidden_size_2),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(in_features=hidden_size_2, out_features=hidden_size_3),
                                                nn.BatchNorm1d(hidden_size_3),
                                                nn.ReLU(inplace=False))

            self.graph_block_3 = nn.Sequential(nn.Linear(in_features=hidden_size_2, out_features=hidden_size_3),
                                                nn.BatchNorm1d(hidden_size_3),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(in_features=hidden_size_3, out_features=hidden_size_4),
                                                nn.BatchNorm1d(hidden_size_4),
                                                nn.ReLU(inplace=False)
                                                )


        if drug_features.enzyme_features != None:
            enzyme_input_size = drug_features.enzyme_features.shape[1]
            self.enzyme_block_1 = nn.Sequential(nn.Linear(in_features=enzyme_input_size, out_features=hidden_size_1),
                                                nn.BatchNorm1d(hidden_size_1),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
                                                nn.BatchNorm1d(hidden_size_2),
                                                nn.ReLU(inplace=False))

            self.enzyme_block_2 = nn.Sequential(nn.Linear(in_features=hidden_size_2, out_features=hidden_size_2),
                                                nn.BatchNorm1d(hidden_size_2),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(in_features=hidden_size_2, out_features=hidden_size_3),
                                                nn.BatchNorm1d(hidden_size_3),
                                                nn.ReLU(inplace=False))

            self.enzyme_block_3 = nn.Sequential(nn.Linear(in_features=hidden_size_3, out_features=hidden_size_3),
                                                nn.BatchNorm1d(hidden_size_3),
                                                nn.ReLU(inplace=False))
        
        
        if drug_features.target_features != None:
            target_input_size = drug_features.target_features.shape[1]

            self.target_block_1 = nn.Sequential(nn.Linear(in_features=target_input_size + enzyme_input_size, out_features=hidden_size_1),
                                                nn.BatchNorm1d(hidden_size_1),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
                                                nn.BatchNorm1d(hidden_size_2),
                                                nn.ReLU(inplace=False))

            self.target_block_2 = nn.Sequential(nn.Linear(in_features=hidden_size_1, out_features=hidden_size_2),
                                                nn.BatchNorm1d(hidden_size_2),
                                                nn.ReLU(inplace=False),
                                                nn.Linear(in_features=hidden_size_2, out_features=hidden_size_3),
                                                nn.BatchNorm1d(hidden_size_3),
                                                nn.ReLU(inplace=False))

            self.target_block_3 = nn.Sequential(nn.Linear(in_features=hidden_size_2, out_features=hidden_size_3),
                                                nn.BatchNorm1d(hidden_size_3),
                                                nn.ReLU(inplace=False))


        self.final_layer = nn.Sequential(
                                        nn.Linear(in_features=4942, out_features=2048),
                                        nn.BatchNorm1d(2048),
                                        nn.ReLU(inplace=False),
                                        nn.Dropout(p=dropout, inplace=False),
                                        
                                        nn.Linear(in_features=2048, out_features=1024),
                                        nn.BatchNorm1d(1024),
                                        nn.ReLU(inplace=False),
                                        nn.Dropout(p=dropout, inplace=False),
                                        
                                        nn.Linear(in_features=1024, out_features=512),
                                        nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=False),
                                        nn.Dropout(p=dropout, inplace=False),
                                        
                                        nn.Linear(in_features=512, out_features=256),
                                        nn.BatchNorm1d(256),
                                        nn.ReLU(inplace=False),
                                        nn.Dropout(p=dropout, inplace=False),
                                        
                                        nn.Linear(in_features=256, out_features=num_classes)
                                        )
    
        self.max_pool = nn.MaxPool1d(kernel_size=4)

    def forward(self, train_X):
        '''
        Input drug batch to generate the smiles, enzyme, target vector pairs
        The drug pairs used to generate smiles and enzymes and target of the 
        '''
        
        # prepare input data
        drug_pairs = [(each['id1'], each['id2']) for _, each in train_X.iterrows()]
        drug_1_index, drug_2_index = self.find_index(drug_pairs)
        if self.drug_features.smiles_features != None:
            smiles_in = self.drug_features.smiles_features
        else:
            smiles_in = 0
        
        
        if self.drug_features.enzyme_features != None:
            enzyme_in = self.drug_features.enzyme_features
        else:
            enzyme_in = 0
        
        
        if self.drug_features.target_features != None:
            target_in = self.drug_features.target_features
        else:
            target_in = 0
        
        
        if self.drug_features.graph_features != None:
            graph_in = self.drug_features.graph_features
            graph_in = self.w_graph(graph_in)
        else:
            graph_in = 0 
        
        target_in = torch.cat([target_in, enzyme_in], dim=1)
        graph_in = torch.cat([graph_in, smiles_in], dim=1)
        
        graph_output = self.graph_block_1(graph_in)
        smiles_output = self.smiles_block_1(smiles_in)
        enzyme_output = self.enzyme_block_1(enzyme_in)
        target_output = self.target_block_1(target_in)
        
        
        # Second Layer
        graph_output = torch.cat([graph_output, smiles_output], dim=1)
        target_output = torch.cat([target_output, enzyme_output], dim=1)
        
        graph_output = self.graph_block_2(graph_output)
        smiles_output = self.smiles_block_2(smiles_output)
        enzyme_output = self.enzyme_block_2(enzyme_output)
        target_output = self.target_block_2(target_output)
        
        
        # Third layer
        graph_output = torch.cat([graph_output, smiles_output], dim=1)
        target_output = torch.cat([target_output, enzyme_output], dim=1)
        
        graph_output = self.graph_block_3(graph_output)
        smiles_output = self.smiles_block_3(smiles_output)
        enzyme_output = self.enzyme_block_3(enzyme_output)
        target_output = self.target_block_3(target_output)


        graph_in, smiles_in, enzyme_in, target_in = self.max_pool(graph_in), self.max_pool(smiles_in), self.max_pool(enzyme_in), self.max_pool(target_in)
        fusion = torch.cat([graph_output, smiles_output, enzyme_output, target_output, graph_in, smiles_in, enzyme_in, target_in], dim=1)

        drug_1 = fusion[drug_1_index]
        drug_2 = fusion[drug_2_index]
        output = torch.cat([drug_1, drug_2], dim=1)
        output = self.final_layer(output)
        return output


    def find_index(self, drug_pairs):
        drug_1_index = []
        drug_2_index = []
        for each in drug_pairs:
            drug_1 = each[0]
            drug_2 = each[1]
            drug_1_index.append(self.drug_features.drugs_list.index(drug_1))
            drug_2_index.append(self.drug_features.drugs_list.index(drug_2))
        
        return drug_1_index, drug_2_index