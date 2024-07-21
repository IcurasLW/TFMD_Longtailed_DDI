import numpy as np
import torch.nn as nn
import torch
import pandas as pd
import torch.nn.functional as F



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float32)
torch.cuda.manual_seed_all(0)




class Tailed_FocalLoss(nn.Module):
    def __init__(self, threshold, beta=2, gamma=2, reduction:str='mean'):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction
        self.threshold = threshold
        
        
    def forward(self, probs, labels):
        '''
        formula for focal loss: average(-alpha*(1-Pt)log(Pt))
        my probs = [[],[],[]]
        tagert = [[31],[11],[3]]
        '''
        Pt_index = labels.squeeze()
        Pt = probs[np.arange(len(Pt_index)), Pt_index]
        FL_loss = -((1-Pt) ** self.gamma) * torch.log2(Pt)
        gate = torch.where(Pt_index >= self.threshold, 1, 0).to(DEVICE)
        Pt_dash = Pt**gate
        loss = FL_loss + (-torch.log2(Pt_dash**self.beta))
        
        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction:str='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
    
    
    def forward(self, probs, labels):
        '''
        formula for focal loss: average(-alpha*(1-Pt)log(Pt))
        my probs = [[],[],[]]
        tagert = [[31],[11],[3]]
        '''

        Pt_index = labels.squeeze()
        Pt = probs[np.arange(len(Pt_index)), Pt_index]
        loss = -((1-Pt) ** self.gamma) * torch.log2(Pt)
        
        if self.reduction=='mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)


class WCELoss(nn.Module):
    def __init__(self, train_data_frq:pd.Series):
        super().__init__()  
        self.weights = torch.tensor(np.sum(train_data_frq.values) / train_data_frq.values).to(DEVICE).to(torch.float32)
        # self.wceloss = nn.CrossEntropyLoss(weight=weight, reduction='mean')
        
    def forward(self, probs, labels):
        weight = self.weights[labels]
        probs = F.softmax(probs, dim=1)
        Pt_index = labels.squeeze()
        Pt = probs[np.arange(len(Pt_index)), Pt_index]
        loss = - weight * torch.log2(Pt)
        return torch.mean(loss)


class ClassBalancedLoss(nn.Module):
    def __init__(self, train_data_frq:pd.Series, loss_type="softmax", beta=0.999, gamma=1):
        super().__init__()  
        self.ny = train_data_frq
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma

    def forward(self, probs, labels): 
        power_term = self.ny.loc[labels.cpu().detach().numpy()]
        denominator = 1.0 - np.power(self.beta, power_term)
        weights = torch.tensor((1.0 - self.beta) / np.array(denominator)).to(DEVICE)
        probs = F.softmax(probs, dim=1)
        Pt_index = labels.squeeze()
        Pt = probs[np.arange(len(Pt_index)), Pt_index]
        
        if self.loss_type == "focal":
            cb_loss = - weights * ((1-Pt) ** self.gamma) * torch.log2(Pt)
        
        elif self.loss_type == "softmax":
            cb_loss = - weights * torch.log2(Pt)
            
        return torch.mean(cb_loss)



class BalancedSoftmax(nn.Module):
    """
    Balanced Softmax Loss
    """
    def __init__(self, train_data_frq):
        super().__init__()
        self.sample_per_class = torch.tensor(train_data_frq)

    def forward(self, logits, label, reduction='mean'):
        spc = self.sample_per_class.type_as(logits)
        spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
        logits = logits + spc.log()
        loss = F.cross_entropy(input=logits, target=label, reduction=reduction)
        return loss




class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super( ).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, probs, labels):
        index = torch.zeros_like(probs, dtype=torch.uint8)
        index.scatter_(1, labels.data.view(-1, 1), 1)
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = probs - batch_m
        output = torch.where(index, x_m, probs)
        return F.cross_entropy(self.s*output, labels, weight=self.weight)