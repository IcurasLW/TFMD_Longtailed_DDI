import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import argparse
import os
import math
from torch import optim
from sklearn.model_selection import StratifiedKFold
from tqdm import trange
from losses import *
from utils import *
from classifier import *
import csv
# from SMILES_Embeddings.SMILES_Embedding import *
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.functional import softmax

torch.autograd.set_detect_anomaly(True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
torch.set_default_dtype(torch.float32)
torch.cuda.manual_seed_all(0)


def train(args, model, train_X, train_y, optimizer, loss_fn):
    
    model.train()
    batch_size = args.batch_size
    batch_nums = math.ceil(len(train_X)/batch_size)
    metrics = None
    
    losses = []
    y_pred = []
    for i in trange(batch_nums):
        # print(f'------------------Training Batch {i}/{batch_nums} started------------------')
        if i * batch_size < len(train_X):
            start_index = i * batch_size
            end_index = (i+1) * batch_size
        else:
            start_index = (i-1) * batch_size
            end_index = len(train_X) - 1

        optimizer.zero_grad()
        y_pred_batch = model(train_X[start_index:end_index])
        y_true_batch = torch.tensor(list(train_y[start_index:end_index])).to(DEVICE)
        loss = loss_fn(y_pred_batch, y_true_batch)
        y_pred_batch = softmax(y_pred_batch, dim=1)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        y_pred.extend(list(y_pred_batch.detach().cpu().numpy()))
        
    metrics = evaluate(args, y_pred, list(train_y), mode='train')
    metrics['loss'] = np.mean(losses)
    return metrics



def test(args, model, test_X, test_y, loss_fn):
    
    model.eval()
    batch_size = args.batch_size
    batch_nums = math.ceil(len(test_X)/batch_size)
    y_pred = []
    losses = []
    with torch.inference_mode():
        for i in trange(batch_nums):
            if i * batch_size < len(test_X):
                start_index = i * batch_size
                end_index = (i+1) * batch_size
            else:
                start_index = (i-1) * batch_size
                end_index = len(test_X) - 1
                
            y_pred_batch = model(test_X[start_index:end_index])
            y_true_batch = torch.tensor(list(test_y[start_index:end_index])).to(DEVICE)
            loss = loss_fn(y_pred_batch, y_true_batch)
            y_pred_batch = softmax(y_pred_batch, dim=1)
            losses.append(loss.item())
            y_pred.extend(list(y_pred_batch.detach().cpu().numpy()))
            
    y_true = np.array(test_y.values)
    y_pred = np.array(y_pred)
    metrics = evaluate(args, y_pred, y_true, mode='test')
    metrics['loss'] = np.mean(losses)
    return metrics



def cross_validate(args, drug_features, loss_function, df_ddis, num_classes ,K=5, epochs=100):
    
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
    Y = make_label(df_ddis)

    for k, (train_index, test_index) in enumerate(skf.split(df_ddis, Y)):
        model = Classifier_NN(
                        drug_features=drug_features,
                        num_classes=num_classes,
                        dropout=args.dropout
                        )


        save_dir = './output/'
        create_folder(save_dir + f'{k}/')
        
        print(f'----------------------------Fold {k}----------------------------')
        lr = args.lr
        model.to(DEVICE)
        train_X_k, test_X_k = df_ddis.loc[train_index], df_ddis.loc[test_index]
        train_y_k, test_y_k = Y[train_index], Y[test_index]
        optimizer = optim.Adam(params=model.parameters(), lr=lr)
        for e in range(epochs+1):
            print(f'----------------------------Epoch {e}----------------------------')
            
            train_metrics = train(
                                args=args,
                                model=model,
                                train_X=train_X_k,
                                train_y=train_y_k,
                                loss_fn=loss_function,
                                optimizer=optimizer
                                )
            
            save_results(save_path=save_dir + f"{k}/", 
                         fold_num=k, 
                         metrics=train_metrics, 
                         mode='train',
                         loss=args.loss_function, 
                         num_cls=args.num_class)


            test_metrics = test(
                                args=args,
                                model=model,
                                test_X=test_X_k,
                                test_y=test_y_k,
                                loss_fn=loss_function
                                )
            
            
            save_results(save_path=save_dir + f"{k}/", 
                         fold_num=k, 
                         metrics=test_metrics, 
                         mode='test',
                         loss=args.loss_function, 
                         num_cls=args.num_class)
            
            # save model file
            if e % 5 == 0 and e > 100:
                save_path = save_dir + f"{k}/"
                create_folder(save_path)
                path_to_save = save_path + f'{k}_fold_model_{args.loss_function}_at_epoch_{e}.pth'
                torch.save({
                            'epoch': e,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'fold':k,
                            }, path_to_save)


def train_no_CV(args, drug_features, loss_function, df_ddis, num_classes ,epochs=100):
    sorted_ddis = df_ddis['interaction'].value_counts().to_dict()
    dict_ddis = {ddis:i for i, ddis in enumerate(sorted_ddis.keys())}
    df_ddis['labels_num'] = df_ddis['interaction'].map(dict_ddis)
    model = Classifier_NN(
                            drug_features=drug_features,
                            num_classes=num_classes,
                            dropout=args.dropout
                            )
    
    save_dir = './output/0/'
    lr = args.lr
    model.to(DEVICE)
    train_df, test_df = split_train_test(df_ddis, test_size=0.2)
    train_X_k, test_X_k = train_df, test_df
    train_y_k, test_y_k = train_df['labels_num'], test_df['labels_num']

    train_X_k, test_X_k = train_df, test_df
    train_y_k, test_y_k = train_df['labels_num'], test_df['labels_num']
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=5e-5)
    scheduler = MultiStepLR(optimizer, milestones=[i for i in range(70, 150, 10)], gamma=0.5)
    
    for e in range(epochs+1):
        print(f'----------------------------Epoch {e}----------------------------')
        
        train_metrics = train(args=args,
                            model=model,
                            train_X=train_X_k,
                            train_y=train_y_k,
                            loss_fn=loss_function,
                            optimizer=optimizer
                            )
        
        save_results(save_path=save_dir, 
                    fold_num=0, 
                     metrics=train_metrics, 
                     mode='train', 
                     loss=args.loss_function, 
                     num_cls=args.num_class)
        scheduler.step()

        test_metrics = test(args=args,
                            model=model,
                            test_X=test_X_k,
                            test_y=test_y_k,
                            loss_fn=loss_function
                            )
        
        save_results(save_path=save_dir, 
                     fold_num=0, 
                     metrics=test_metrics, 
                     mode='test',
                     loss=args.loss_function, 
                     num_cls=args.num_class)

def record_grad(model):
    grad_sum = 0
    for param in model.parameters():
        if param.grad is not None:
            grad_sum += param.grad.sum()
            
    return grad_sum.item()



def main(args):
    torch.manual_seed(0)
    data_path = args.data_path
    ddis_data_path = data_path + "ddi.csv"
    features_data_path = data_path + "features.csv"
    df_ddis = pd.read_csv(ddis_data_path)
    if 'smiles' in args.feature_path:
        print('Use SMILES')
        smiles_features = torch.load(data_path + 'SMILES_embedding.pt').to(DEVICE).to(torch.float32)
    else:
        smiles_features = None
    
    
    if 'target' in args.feature_path:
        print('Use Target')
        targets_sim_matrix = make_features_matrix(features_data_path, mode='Targets')
        pca = PCA(0.99)
        targets_sim_matrix = torch.tensor(pca.fit_transform(targets_sim_matrix)).to(DEVICE).to(torch.float32)
    else:
        targets_sim_matrix = None
    
    
    if 'enzyme' in args.feature_path:
        print('Use Enzyme')
        enzymes_sim_matrix = make_features_matrix(features_data_path, mode='Enzymes')
        pca = PCA(0.99)
        enzymes_sim_matrix = torch.tensor(pca.fit_transform(enzymes_sim_matrix)).to(DEVICE).to(torch.float32)
    else:
        enzymes_sim_matrix = None
    
    
    if 'graph' in args.feature_path:
        print('Use Graph')
        graph_embedding = torch.tensor(np.load(data_path+'graph_embedding.npy')).to(DEVICE).to(torch.float32)
    else:
        graph_embedding = None
    
    
    drug_features = Drug_Features(
                                smiles_features=smiles_features,
                                target_features=targets_sim_matrix,
                                enzyme_features=enzymes_sim_matrix,
                                graph_features=graph_embedding,
                                dataset_path=features_data_path
                                )
    
    
    # Decide loss function
    if args.loss_function == 'focalloss':
        loss_fn = FocalLoss()
    elif args.loss_function == 'crossentropy':
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss_function == 'tailedfocalloss':
        threshold = findthreshold(df=df_ddis, tail_factor=0.9)
        loss_fn = Tailed_FocalLoss(beta=2, threshold=threshold)
    elif args.loss_function == 'cbloss':
        sorted_ddis = df_ddis['interaction'].value_counts().to_dict()
        dict_ddis = {ddis:i for i, ddis in enumerate(sorted_ddis.keys())}
        df_ddis['labels_num'] = df_ddis['interaction'].map(dict_ddis)
        train_data_frq = df_ddis['labels_num'].value_counts()
        loss_fn = ClassBalancedLoss(train_data_frq)
    elif args.loss_function == 'BSloss':
        sorted_ddis = df_ddis['interaction'].value_counts().to_dict()
        dict_ddis = {ddis:i for i, ddis in enumerate(sorted_ddis.keys())}
        df_ddis['labels_num'] = df_ddis['interaction'].map(dict_ddis)
        train_data_frq = df_ddis['labels_num'].value_counts()
        loss_fn = BalancedSoftmax(train_data_frq)
    elif args.loss_function == 'WCEloss':
        sorted_ddis = df_ddis['interaction'].value_counts().to_dict()
        dict_ddis = {ddis:i for i, ddis in enumerate(sorted_ddis.keys())}
        df_ddis['labels_num'] = df_ddis['interaction'].map(dict_ddis)
        train_data_frq = df_ddis['labels_num'].value_counts()
        loss_fn = WCELoss(train_data_frq)
    elif args.loss_function == 'LDAMloss':
        sorted_ddis = df_ddis['interaction'].value_counts().to_dict()
        dict_ddis = {ddis:i for i, ddis in enumerate(sorted_ddis.keys())}
        df_ddis['labels_num'] = df_ddis['interaction'].map(dict_ddis)
        train_data_frq = df_ddis['labels_num'].value_counts()
        loss_fn = LDAMLoss(cls_num_list=train_data_frq.values)
    else:
        raise Exception('No losses defined')


    if args.num_class != 171:
        cross_validate(
                        args=args,
                        drug_features=drug_features,
                        loss_function=loss_fn,
                        df_ddis=df_ddis,
                        K=5,
                        epochs=args.epochs,
                        num_classes=args.num_class
                        )
    else:
        train_no_CV(
                    args=args,
                    drug_features=drug_features,
                    loss_function=loss_fn,
                    df_ddis=df_ddis,
                    epochs=args.epochs,
                    num_classes=args.num_class
                    )


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_class", type=int, default=171)
    parser.add_argument("--data_path", type=str, default='/media/nathan/DATA/1Adelaide/TFMD/dataset/171/')
    parser.add_argument("--feature_path", type=list, default=['graph', 'smiles', 'target', 'enzyme']) #
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.50)
    parser.add_argument("--loss_function", type=str, default='tailedfocalloss')
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--cross-validate", type=str, default='stratified')
    
    
    args = parser.parse_args()
    main(args)
