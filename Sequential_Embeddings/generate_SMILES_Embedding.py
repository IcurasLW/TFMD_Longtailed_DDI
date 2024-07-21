import torch
import pandas as pd
import molbart.util as util
# from SMILES_Sequential_Embeddings.molbart.decoder import DecodeSampler
# from SMILES_Sequential_Embeddings.molbart.models.pre_train import BARTModel
from molbart.data.datasets import ReactionDataset
from molbart.data.datamodules import FineTuneReactionDataModule
import os



def build_dataset(df):
    smiles = df['Smiles'].to_list()
    dataset = ReactionDataset(smiles, smiles)
    return dataset


def build_datamodule(batch_size, dataset, tokeniser, max_seq_len):
    test_idxs = range(len(dataset))
    dm = FineTuneReactionDataModule(
        dataset,
        tokeniser,
        batch_size,
        max_seq_len,
        val_idxs=[],
        test_idxs=test_idxs
    )
    return dm


def concat_tensor(tensor_1, tensor_2):
    '''
    tensor_1 has the length of (20, 20, 1024)

    tensor_2 has the length of (30, 20, 1024)

    The concat function returns shape (30, 40, 1024)
    the lower length of first layer, fill up with zeros for the augment words position.
    '''

    seq_len_1 = tensor_1.size()[0]
    seq_len_2 = tensor_2.size()[0]

    if seq_len_1 > seq_len_2:
        concat_seq_len = seq_len_1
        short_ts = tensor_2
        padded_ts = torch.zeros(concat_seq_len, short_ts.size()[1], short_ts.size()[2])
        padded_ts[:short_ts.size()[0], :, :] = short_ts
        tensor_1 = tensor_1.to('cpu')
        merged_tensor = torch.cat([tensor_1, padded_ts], dim=1)

    else:
        concat_seq_len = seq_len_2
        short_ts = tensor_1
        padded_ts = torch.zeros(concat_seq_len, short_ts.size()[1], short_ts.size()[2])
        padded_ts[:short_ts.size()[0], :, :] = short_ts
        tensor_2 = tensor_2.to('cpu')
        merged_tensor = torch.cat([padded_ts, tensor_2], dim=1)
        
    return merged_tensor


def smiles_embedding(model, smiles_loader):
    device = "cuda:0" if util.use_gpu else "cpu"
    model = model.to(device)
    model.eval()
    output = torch.zeros(0,0,1024)

    for _, batch in enumerate(smiles_loader):
        device_batch = {
            key: val.to(device) if type(val) == torch.Tensor else val for key, val in batch.items()
        }

        enc_input = device_batch["encoder_input"]
        enc_mask = device_batch["encoder_pad_mask"]
        # Freezing the weights reduces the amount of memory leakage in the transformer
        model.freeze()

        encode_input = {
            "encoder_input": enc_input,
            "encoder_pad_mask": enc_mask
        }
        with torch.no_grad():
            embedding = model.encode(encode_input)
            memory = model.encoder(embedding)
            
        output = concat_tensor(output, memory)
    model.unfreeze()
    return output


def aggregate(data, pooling_strategy='average'):
    
    def ave_pooling(data):
        list_tensors = []
        for i in range(data.size()[1]):
            count = 0
            sum_embedding = 0
            for j in range(data.size()[0]):
                if torch.all(data[j][i] != 0):
                    count += 1
                    sum_embedding += data[j][i]
                else:
                    continue
            list_tensors.append(sum_embedding / count)

        output = torch.stack(list_tensors)
        return output


    def max_pooling(data):
        list_tensors = []
        for i in range(data.size()[1]):
            cur_tensor = []
            for j in range(data.size()[0]):
                cur_tensor.append(data[j][i])
                
            stack_tensor = torch.stack(cur_tensor)
            cur_tensor, _ = torch.max(stack_tensor, dim=0)
            list_tensors.append(cur_tensor)
        
        output = torch.stack(list_tensors)
        return output


    if pooling_strategy=='average':
        return ave_pooling(data)
    elif pooling_strategy=='max':
        return max_pooling(data)
    else:
        raise KeyError("Only 'average' and 'max' pooling are allowed")



def running_SMILES_embeddings(data_path):
    vocab = "/media/nathan/DATA/1Adelaide/TFMD/Sequential_Embeddings/bart_vocab.txt"
    model_path = "/media/nathan/DATA/1Adelaide/TFMD/Sequential_Embeddings/molbart/models/pre-trained/combined-large/step=1000000.ckpt" # You can change to model from MolBart

    chem_token_start_idx = util.DEFAULT_CHEM_TOKEN_START
    num_beams = 10
    batch_size = 20

    print("Building tokeniser...")
    tokeniser = util.load_tokeniser(vocab, chem_token_start_idx)


    print("Reading SMILES...")
    df = pd.read_csv(data_path)
    dataset = build_dataset(df)


    print("Loading model...")
    model = util.load_bert(model_path) # only the encoder part is loaded.
    model.num_beams = num_beams


    dm = build_datamodule(batch_size, dataset, tokeniser, model.max_seq_len)
    dm.setup()
    test_loader = dm.test_dataloader()


    print("Embedding SMILES...")
    SMILES_embeddings = smiles_embedding(model, test_loader)
    SMILES_embeddings = aggregate(SMILES_embeddings)
    print("Finish Processing...")

    return SMILES_embeddings



if __name__ == "__main__":

    data_path = '/media/nathan/DATA/1Adelaide/TFMD/dataset/65/'      # 1. Select your dataset. Given dataset: ddi_171, ddi110, ddimdl, muffin
    ddis_data_path = data_path + "ddi.csv"
    features_data_path = data_path + "features.csv"
    df_ddis = pd.read_csv(ddis_data_path)
    smiles_features = running_SMILES_embeddings(features_data_path)
    torch.save(smiles_features, os.path.join(data_path, "SMILES_embedding.pt"))            # 2. move this embedding file into corresponding diretory, for example ---> /dataset/ddi_171/SMILES_features.pt