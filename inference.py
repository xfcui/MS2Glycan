import dgl
import torch
import glypy
import argparse
import pickle
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from data.process import GlycanCSV, GraphormerDGLDataset, preprocess_dgl_graph, find_submass, collator, test_glycan_accuracy, graph2glycan
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
    parser.add_argument('--graph_model', type=str, default='ckpts/graphormer_nolung.pt')                
    parser.add_argument('--cnn_model', type=str, default='ckpts/allmodel_nolung.pt')
    
    parser.add_argument('--glycan_db', type=str, default='sample_data/all.pkl')                                              
    parser.add_argument('--combination', type=str, default='sample_data/combination.pkl')  
    parser.add_argument('--csv_file_predict', type=str, default='sample_data/test.csv')                       
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
    
def generate_new_graph(parent_node, nodes_onehot, graph):
    graph_feature = graph.ndata['x']
    num_node = graph.number_of_nodes()
    num_new_node = nodes_onehot.shape[0]
    graph.add_nodes(num_new_node)

    onehot_newfeature = torch.zeros((num_new_node, 6), dtype=torch.int32, device=device)
    onehot_newfeature[:,0:5]=nodes_onehot
    graph_feature = torch.cat((graph_feature, onehot_newfeature), dim=0).to(torch.int32)
    # print(graph_feature)
    for i in range(num_new_node):
        graph.add_edges([parent_node], [num_node+i])
    graph.ndata['x'] = torch.tensor(graph_feature)
    graph.ndata['x'][:, 5] = 0

    out_degree = graph.out_degrees()
    leaf_nodes = torch.where(out_degree == 0)[0]
    graph.ndata['x'][leaf_nodes, 5] =1
    return graph



if __name__ == '__main__':

    args=parse_args()
    sugar_classes_name = ['Fuc', 'Man', 'GlcNAc', 'NeuAc', 'NeuGc']
    sugar_classes = [glypy.monosaccharides[name].mass() - mass_free_reducing_end for name in sugar_classes_name]
    with open(args.combination, 'rb') as f:
        combination = pickle.load(f)
    combination = torch.tensor(combination, device=device)

    ion_mass = find_submass(combination, sugar_classes)
    graphormer_model = GraphormerModel(args)
    graphormer_model.load_state_dict(torch.load(args.graph_model,map_location=device),strict=False)
    graphormer_model.to(device)
    model = GraphormerIonCNN(args, ion_mass, sugar_classes, graphormer_model)
    model.load_state_dict(torch.load(args.cnn_model,map_location=device),strict=False)
    model.to(device)

    dataset_dict = GlycanCSV(args,ion_mass,combination)

    graphormer_datset = GraphormerDGLDataset(dataset=dataset_dict,seed=args.seed)
    test_test = Subset(graphormer_datset,list(range(len(graphormer_datset))))
    train_dataloader = DataLoader(      test_test,
                                        batch_size=args.batch_size,
                                        collate_fn=lambda x: {key:value for key, value in collator(x).items()},
                                        shuffle=False,
                                        )

    model.eval()

    complete_graph = []
    target_graph = []
    unable2predict = []
    for i, sample in enumerate(tqdm(train_dataloader)):
        samples={}
        for key, value in sample.items():
            samples[key]=value.to(device)
        sample = samples    
        with torch.no_grad():
            while 1:
                sample_size = len(sample['idx'])             
                graphs = dgl.unbatch(sample['graph'])
                ys = sample["y"]
                
                sample_dict = {'batched_data': sample}
                logits,sites = model(**sample_dict)

                left_comps = sample['left_comps']
                left_comps_ = left_comps.repeat(1, 1, len(combination)).view(sample_size, len(combination), -1)
                combination_ = combination.repeat(sample_size, 1, 1).view(sample_size, len(combination), -1)
                invalid_entry = (left_comps_ - combination_) < 0
                invalid_entry_s = torch.any(invalid_entry, -1)
                logits[invalid_entry_s] = float('-inf')
                prediction = torch.argmax(logits, dim=-1)
                prediction = combination[prediction].view(sample_size, len(sugar_classes))
                left_comps -= prediction
                new_pyggraphs = []
                
                for idx, graph in enumerate(graphs):

                    sample_id = int(sample['idx'][idx])
                    theoretical_mzs = sample['theoretical_mz'][idx]
                    observed_mz = sample['observed_mz'][idx]
                    intensity = sample['intensity'][idx]
                    cur_left_comp = left_comps[idx]
                    new_feature = prediction[idx]
                    num_feature = new_feature.sum()
                    mono_mass = new_feature[:-1].float() @ torch.tensor(sugar_classes[:-1], device=device)
                    theoretical_mzs += mono_mass
                    onehot_newfeature = torch.zeros((num_feature, len(sugar_classes)+1), dtype=torch.int32, device=device)

                    for n in range(num_feature - 1):
                        feature = torch.nonzero(new_feature)[-1]
                        new_feature[feature] -= 1
                        onehot_newfeature[n, 0:5] = new_feature
                    new_feature = onehot_newfeature[:, 0:5] + cur_left_comp
                    out_degree = graph.out_degrees()
                    leaf_nodes = out_degree == 0
                    leaf = torch.nonzero(leaf_nodes).squeeze(dim=1)
                    site = sites[int(idx)]
                    max_prob, max_index = torch.max(site[leaf], dim=0)
                    max_index_in_original = leaf[max_index]
                    new_graph = generate_new_graph(max_index_in_original, new_feature, graph)

                    if torch.all(cur_left_comp == 0):
                        complete_graph.append(new_graph)
                        target_graph.append(ys[idx])
                    elif torch.any(cur_left_comp < 0):
                        unable2predict.append(int(ys[idx][0]))
                        left_comps[idx].fill_(0)
                    else:
                        new_pyggraph = preprocess_dgl_graph(new_graph.to('cpu'), ys[idx].view(1, -1), sample_id, cur_left_comp.view(1, len(sugar_classes)),torch.tensor([1]).unsqueeze(0), theoretical_mzs, observed_mz, intensity)
                        new_pyggraphs.append(new_pyggraph)
                if new_pyggraphs:
                    sample = {key:value.to(device) for key, value in collator(new_pyggraphs).items()}
                if len(new_pyggraphs)==0:
                    break
    
    print('Number of successful predictions:', len(complete_graph))
    print('Number of unable predictions:', len(unable2predict))

    with open(args.glycan_db, 'rb') as f:
        glycan_dict = pickle.load(f)
    target_glycan = [[glycan_dict[str(int(item[0]))]['GLYCAN'], int(item[1]), int(item[2]), item[3].item()] for item
                    in target_graph]

    predict_glycans = []
    for i, graph in enumerate(complete_graph):
        glycan = graph2glycan(graph, sugar_classes_name) if graph else None
        predict_glycans.append(glycan)

    test_glycan_accuracy(target_glycan, predict_glycans, args.csv_file_predict)
