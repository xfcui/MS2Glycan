import os
import time
import glypy
import torch
import pickle
import argparse
import torch.nn as nn
import torch.optim as optim

from torch.nn import functional
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from data.process import  GlycanDBCSV, GraphormerDGLDataset, find_submass, collator, create_all_glycan
from modules import GraphormerGraphEncoder
from modules import ionCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mass_free_reducing_end = 18.01056468370001
mass_proton = 1.00727647


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--encoder_embed_dim', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=21)
    parser.add_argument('--num_atoms', type=int, default=512 * 7)
    parser.add_argument('--num_in_degree', type=int, default=512)
    parser.add_argument('--num_out_degree', type=int, default=512)
    parser.add_argument('--num_edges', type=int, default=512 * 3)
    parser.add_argument('--num_spatial', type=int, default=512)
    parser.add_argument('--num_edge_dis', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--edge_type', type=str, default="multi_hop")
    parser.add_argument('--multi_hop_max_dist', type=int, default=5)
    parser.add_argument('--encoder_layers', type=int, default=2)
    parser.add_argument('--encoder_attention_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    parser.add_argument('--act_dropout', type=float, default=0.1)
    parser.add_argument('--encoder_normalize_before', action="store_true")
    parser.add_argument('--pre_layernorm', action="store_true")


    parser.add_argument('--mgf_file', type=str, default='sample_data/mouse-tissue.mgf')            
    parser.add_argument('--graph_model', type=str, default='ckpts/graphormer_nolung1.pt')                
    parser.add_argument('--cnn_model', type=str, default='ckpts/allmodel_nolung1.pt')
    
    parser.add_argument('--glycan_db', type=str, default='sample_data/all.pkl')                                              
    parser.add_argument('--combination', type=str, default='sample_data/combination.pkl')  
    parser.add_argument('--csv_file_train', type=str, default='sample_data/test.csv')                       
    return parser.parse_args()


class GraphormerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.graph_encoder = GraphormerGraphEncoder(
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            encoder_normalize_before=args.encoder_normalize_before,
            pre_layernorm=args.pre_layernorm,
        )
        self.masked_lm_pooler = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.lm_head_transform_weight = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.activation_fn = nn.GELU()
        self.layer_norm = nn.LayerNorm(args.encoder_embed_dim)
        self.embed_out = nn.Linear(args.encoder_embed_dim, args.num_classes, bias=False)
        self.siteclassifier = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(args.encoder_embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batched_data):
        inner_states, attn_weight = self.graph_encoder(batched_data)
        x = inner_states[-1].transpose(0, 1)
        x = self.lm_head_transform_weight(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        prediction = self.embed_out(x)
        siteprediction = self.siteclassifier(x[:,1:,:]).view(-1,512)
        siteprediction = self.relu(siteprediction)
        siteprediction = self.fc2(siteprediction)
        siteprediction = self.sigmoid(siteprediction)
    
        return x[:, 0, :], prediction[:, 0, :],siteprediction.view(len(batched_data['idx']),-1)
    
class GraphormerIonCNN(nn.Module):
    def __init__(self, args, ion_mass, sugar_classes, graph_embedding=None):
        super().__init__()
        self.graph_embedding = graph_embedding
        self.ionCNN = ionCNN(
            encoder_embed_dim=args.encoder_embed_dim,
            ion_mass=ion_mass, 
            sugar_classes=sugar_classes)
        self.weight= torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.embed_out = nn.Linear(
            2*args.encoder_embed_dim, args.num_classes, bias=False)

    def forward(self, batched_data):
        out = self.ionCNN(batched_data)
        graph_embed, _,siteprediction= self.graph_embedding(batched_data)
        combined = torch.cat((graph_embed, out*self.weight), dim=1)
        out = self.embed_out(combined)

        return out,siteprediction
    
def train_loss(model, optimizer, sample, targets):
    model.train()
    optimizer.zero_grad()
    sample_size = len(sample['idx'])
    sample_dict = {'batched_data': sample}
    logits,logits_site = model(**sample_dict)
    targets = targets.to(torch.long)
    targets_site = sample['site']
    batch_size,max_node,_ = sample['x'].size()
    result_tensor = torch.zeros(batch_size, max_node, dtype=torch.float32).to(device)
    result_tensor.scatter_(1, targets_site.unsqueeze(1), 1)
    torch.cuda.empty_cache()
    loss = functional.cross_entropy(
        logits, targets.reshape(-1), reduction="mean"
    )
    criterion = nn.BCELoss()
    loss1 = criterion(logits_site, result_tensor)
    total_loss = loss + loss1
    total_loss.backward()
    optimizer.step()

    classification_correct = (torch.argmax(logits, dim=-1).reshape(-1) == targets.reshape(-1)).sum()
    localization_correct=((torch.argmax(logits_site, dim=-1).reshape(-1) == targets_site.reshape(-1))).sum() 
    return total_loss, classification_correct, localization_correct,sample_size

def value_loss(model, sample, targets):
    model.eval()
    sample_size = len(sample['idx'])
    sample_dict = {'batched_data': sample}
    with torch.no_grad():
        logits,logits_site = model(**sample_dict)
        targets = targets.to(torch.long)
        targets_site = sample['site']
        batch_size,max_node,_ = sample['x'].size()
        result_tensor = torch.zeros(batch_size, max_node, dtype=torch.float32).to(device)
        result_tensor.scatter_(1, targets_site.unsqueeze(1), 1)
        torch.cuda.empty_cache()
        loss = functional.cross_entropy(
            logits, targets.reshape(-1), reduction="mean"
        )
        criterion = nn.BCELoss()
        loss1 = criterion(logits_site, result_tensor)
        
    classification_correct = (torch.argmax(logits, dim=-1).reshape(-1) == targets.reshape(-1)).sum()
    localization_correct=((torch.argmax(logits_site, dim=-1).reshape(-1) == targets_site.reshape(-1))).sum() 
    return loss+loss1, classification_correct, localization_correct,sample_size


if __name__ == '__main__':
    args=parse_args()
    sugar_classes_name = ['Fuc', 'Man', 'GlcNAc', 'NeuAc', 'NeuGc']
    sugar_classes = [glypy.monosaccharides[name].mass() - mass_free_reducing_end for name in sugar_classes_name]
    with open(args.combination, 'rb') as f:
        combination = pickle.load(f)
    combination = torch.tensor(combination, device=device)
    ion_mass = find_submass(combination, sugar_classes)

    start_time = time.time()

    graphormer_model = GraphormerModel(args)
    if os.path.exists(args.graph_model):
        graphormer_model.load_state_dict(torch.load(args.graph_model,map_location=device),strict=False)
    graphormer_model.to(device)
    model = GraphormerIonCNN(args, ion_mass, sugar_classes, graphormer_model)
    if os.path.exists(args.cnn_model):
        model.load_state_dict(torch.load(args.cnn_model,map_location=device),strict=False)
    model.to(device)

    create_all_glycan(args.glycan_db)
    dataset_dict = GlycanDBCSV(args,ion_mass)
    graphormer_datset = GraphormerDGLDataset(dataset=dataset_dict,seed=args.seed)
    train_idx, val_idx = train_test_split(list(range(len(graphormer_datset))), test_size=0.1)
    train_set = Subset(graphormer_datset, train_idx)
    val_set = Subset(graphormer_datset, val_idx)
    train_dataloader = DataLoader(  train_set,
                                    batch_size=args.batch_size,
                                    collate_fn=lambda x: {key:value for key, value in collator(x).items()},
                                    shuffle=False,
                                    )
    val_dataloader = DataLoader(    val_set,
                                    batch_size=args.batch_size,
                                    collate_fn=lambda x: {key:value for key, value in collator(x).items()},
                                    shuffle=False,
                                )

    end_time = time.time()
    print("Model loading time: ",end_time - start_time)
    print('-----------------------------------------------------------------')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_classification = 0
    best_localization = 0
    for epoch in range(args.num_epoch):
        start_time = time.time()

        train_epoch_loss = 0
        train_classification_correct = 0
        train_localization_correct = 0
        train_sizes = 0
        for i, sample in enumerate(train_dataloader):
            samples={}
            for key, value in sample.items():
                samples[key]=value.to(device)
            loss, classification_correct, localization_correct, sample_size = train_loss(model, optimizer, samples, samples["y"])
            train_classification_correct += classification_correct
            train_localization_correct += localization_correct
            train_sizes += sample_size
            train_epoch_loss += loss 
        train_classification_correct = train_classification_correct / train_sizes
        train_localization_correct = train_localization_correct / train_sizes

        val_epoch_loss = 0
        val_classification_correct = 0
        val_localization_correct = 0
        val_sizes = 0
        for j, test_sample in enumerate(val_dataloader):
            test_samples={}
            for key, value in test_sample.items():
                test_samples[key]=value.to(device)
            loss, classification_correct, localization_correct, sample_size = value_loss(model, test_samples,test_samples["y"])
            val_classification_correct += classification_correct
            val_localization_correct += localization_correct
            val_sizes += sample_size
            val_epoch_loss += loss

        val_classification_correct = val_classification_correct / val_sizes
        val_localization_correct = val_localization_correct / val_sizes

        if val_classification_correct > best_classification and val_localization_correct > best_localization:
            torch.save(model.state_dict(), args.cnn_model)
            torch.save(graphormer_model.state_dict(), args.graph_model)
            best_classification = val_classification_correct
            best_localization = val_localization_correct

        end_time = time.time()
        print("Time to train an epoch: ",end_time - start_time)
        print(f'\tTrain Loss: {train_epoch_loss:.3f} | \tVal Loss: {val_epoch_loss:.3f}')
        print(f'\tTrain Classification Accuracy: {train_classification_correct:.3f} | Train Localization Accuracy: {train_localization_correct:.3f} | Number of Training Samples: {train_sizes}')
        print(f'\tTest Classification Accuracy: {val_classification_correct:.3f} | Val Localization Accuracy: {val_localization_correct:.3f} | Number of Validation Samples: {val_sizes}')
        print('-----------------------------------------------------------------')

