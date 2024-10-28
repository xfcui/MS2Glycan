import os
import re
import csv
import dgl
import copy
import torch
import pickle
import itertools
import collections
import numpy as np

from dgl import DGLGraph
from dgl.data import DGLDataset
from torch_geometric.data import Dataset
from torch_geometric.data import Data as PYGGraph

import glypy
from glypy.io import glycoct as glypy_glycoct
from glypy.structure.glycan_composition import MonosaccharideResidue

import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos

mass_free_reducing_end = 18.01056468370001
mass_proton = 1.00727647


def find_submass(combination, sugar_classes):
    combination = torch.tensor(combination)
    unique_ions = []
    for idx in range(combination.shape[0]):
        new_feature = torch.squeeze(combination[idx, :].clone().detach())[:len(sugar_classes)-1]
        num_feature = new_feature.sum()
        onehot_newfeature = torch.zeros(num_feature, dtype=torch.float32)
        for n in range(num_feature):
            feature = torch.nonzero(new_feature)[-1]
            new_feature[feature] -= 1
            onehot_newfeature[n] = sugar_classes[feature]
        all_ions = torch.combinations(onehot_newfeature)
        all_ions = torch.sum(all_ions, dim=-1)
        all_ions = torch.cat((all_ions, onehot_newfeature))
        all_ions = torch.cat((all_ions, torch.sum(onehot_newfeature, dim=-1).unsqueeze(0)))
        unique_ion = torch.unique(all_ions, sorted=True)
        unique_ions.append(unique_ion)
    out = torch.nn.utils.rnn.pad_sequence(unique_ions, batch_first=True)
    return out



class GlycanCSV(DGLDataset):
    def __init__(self, args, ion_mass,combination):
        self.graphs = []
        self.labels = []
        self.left_compositions = []
        self.observed_mzs = []
        self.theoretical_mzs = []
        self.intensities = []

        self.csv_file = args.csv_file_predict
        self.combination = combination
        self.ion_mass = ion_mass

        self.input_spectrum_file, self.spectrum_location_dict = spectrum_preprocessing(args)
        super().__init__(name='glycan_csv')

    def process(self):
        dir = self.csv_file.split('.csv')[0] + "-predict"
        if not os.path.exists(dir):
            os.mkdir(dir)
        if not os.path.exists(dir+'/left_composition.pkl'):  
            num_fractions = 5
            tissue_name = ['MouseBrain', 'MouseHeart', 'MouseKidney', 'MouseLiver', 'MouseLung']
            with open(self.csv_file, 'r') as csvfile:
                csvreader = csv.DictReader(csvfile)
                for row in csvreader:
                    tissue = row['Source File'].split('-')[0]
                    tissue_id = tissue_name.index(tissue)
                    fraction = int(row['Source File'].split('.')[0][-1])
                    fraction_id = tissue_id * num_fractions + fraction
                    psm_scan = row['Scan']
                    scan = 'F' + str(fraction_id) + ':' + psm_scan
                    try:
                        peptide_only_mass = float(row['PepMass'])
                    except:
                        precursor_mass = float(row['Mass'])
                        target_glycan_mass = float(row['Glycan Mass'])
                        adduct_mass = float(row['Adduct Mass'])
                        isotope_shift = float(row['Isotopic Shift'])
                        peptide_only_mass = precursor_mass - target_glycan_mass - adduct_mass +isotope_shift * mass_proton + mass_proton
                    sugar_classes = ['Fuc', 'Hex', 'HexNAc', 'NeuAc', 'NeuGc']
                    target_glycan_id = row['Glycan ID']
                    composition = row['Glycan']
                    glycan_lst = composition.replace('(', ',').replace(')', ',').split(',')
                    ori_comp_tensor = torch.zeros((1, len(sugar_classes)+1))
                    for k, g in enumerate(sugar_classes):
                        if g in glycan_lst:
                            i = glycan_lst.index(g)
                            ori_comp_tensor[:, k] = int(glycan_lst[i+1])
                    edges_src, edges_dst = np.array([0]).astype('int'), np.array([1]).astype('int')
                    root_G = dgl.graph((edges_src, edges_dst), num_nodes=2)
                    special_token = torch.cat((ori_comp_tensor, ori_comp_tensor))
                    root_G.ndata['x'] = torch.tensor(special_token,dtype=torch.int32) 
                    out_degree = root_G.out_degrees()
                    leaf_nodes = torch.where(out_degree == 0)[0]
                    root_G.ndata['x'][leaf_nodes, 5] = 1
                    
                    self.graphs.append(root_G)
                    self.left_compositions.append(ori_comp_tensor[:,0:5])
                    self.labels.append(torch.tensor([[int(target_glycan_id), fraction_id, int(psm_scan), peptide_only_mass]]))
                    theoretical_mz = torch.add(peptide_only_mass, self.ion_mass)
                    mz, intensity = read_spectrum(self.input_spectrum_file, self.spectrum_location_dict, scan, peptide_only_mass)
                    self.theoretical_mzs.append(torch.tensor(theoretical_mz))
                    self.observed_mzs.append(torch.tensor(mz))
                    self.intensities.append(torch.tensor(intensity))
            with open(dir + "/" + "graphs.pkl", 'wb') as fr:
                pickle.dump(self.graphs, fr)
            with open(dir + "/" + "labels.pkl", 'wb') as fr:
                pickle.dump(self.labels, fr)
            with open(dir + "/" + "left_compositions.pkl", 'wb') as fr:
                pickle.dump(self.left_compositions, fr)
            with open(dir + "/" + "theoretical_mzs.pkl", 'wb') as f:
                pickle.dump(self.theoretical_mzs, f)
            with open(dir + "/" + "observed_mzs.pkl", 'wb') as fr:
                pickle.dump(self.observed_mzs, fr)
            with open(dir + "/" + "intensities.pkl", 'wb') as f:
                pickle.dump(self.intensities, f)

        else:
            with open(dir + "/" + "graphs.pkl", 'rb') as fr:
                self.graphs = pickle.load(fr)
            with open(dir + "/" + "labels.pkl", 'rb') as fr:
                self.labels = pickle.load(fr)
            with open(dir + "/" + "left_compositions.pkl", 'rb') as fr:
                self.left_compositions = pickle.load(fr)
            with open(dir + "/" + "theoretical_mzs.pkl", 'rb') as fr:
                self.theoretical_mzs = pickle.load(fr)
            with open(dir + "/" + "observed_mzs.pkl", 'rb') as fr:
                self.observed_mzs = pickle.load(fr)
            with open(dir + "/" + "intensities.pkl", 'rb') as fr:
                self.intensities = pickle.load(fr) 
        print('test glycans = ',len(self.graphs))

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i], self.left_compositions[i],torch.tensor([1]).unsqueeze(0), self.theoretical_mzs[i], self.observed_mzs[i], self.intensities[i]

    def __len__(self):
        return len(self.graphs)



def spectrum_preprocessing(args):
    input_spectrum_file = args.mgf_file
    tissue_name = ['MouseBrain', 'MouseHeart', 'MouseKidney', 'MouseLiver', 'MouseLung']
    num_fractions = 5

    spectrum_location_file = input_spectrum_file + '.locations.pkl'
    if os.path.exists(spectrum_location_file):
        with open(spectrum_location_file, 'rb') as fr:

            data = pickle.load(fr)
            spectrum_location_dict, spectrum_rtinseconds_dict, spectrum_count = data
    else:
        input_spectrum_handle = open(input_spectrum_file, 'r')
        spectrum_location_dict = {}
        spectrum_rtinseconds_dict = {}
        line = True
        while line:
            current_location = input_spectrum_handle.tell()
            line = input_spectrum_handle.readline()
            if "BEGIN IONS" in line:
                spectrum_location = current_location
            elif "RAWFILE=" in line:
                rawfile = re.split('=|\r|\n|\\\\', line)[-2]
                tissue = rawfile.split('-')[0]
                tissue_id = tissue_name.index(tissue)
                fraction = int(rawfile.split('.')[0][-1])
                fraction_id = tissue_id*num_fractions+fraction
            elif "SCANS=" in line:
                scan = re.split('=|\r|\n', line)[1]
                scan = 'F' + str(fraction_id) + ':' + scan
                spectrum_location_dict[scan] = spectrum_location
            elif "RTINSECONDS=" in line:
                rtinseconds = float(re.split('=|\r|\n', line)[1])
                spectrum_rtinseconds_dict[scan] = rtinseconds
        spectrum_count = len(spectrum_location_dict)
        with open(spectrum_location_file, 'wb') as fw:
            pickle.dump((spectrum_location_dict, spectrum_rtinseconds_dict, spectrum_count), fw)
        input_spectrum_handle.close()

    return input_spectrum_file, spectrum_location_dict

def read_spectrum(input_spectrum_file, spectrum_location_dict, scan_id, peptide_mass):
    input_spectrum_handle = open(input_spectrum_file, 'r')
    spectrum_location = spectrum_location_dict[scan_id]
    input_file_handle = input_spectrum_handle
    input_file_handle.seek(spectrum_location)

    line = input_file_handle.readline()
    line = input_file_handle.readline()
    line = input_file_handle.readline()
    line = input_file_handle.readline()
    line = input_file_handle.readline()
    line = input_file_handle.readline()
    line = input_file_handle.readline()
    line = input_file_handle.readline()
    line = input_file_handle.readline()

    mz_list = []
    intensity_list = []
    line = input_file_handle.readline()
    while not "END IONS" in line:
        mz, intensity = re.split(' |\n', line)[:2]
        mz_float = float(mz)
        intensity_float = float(intensity)
        mz_list.append(mz_float)
        intensity_list.append(intensity_float)
        line = input_file_handle.readline()

    return mz_list, intensity_list





def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [
        (
            item.graph_data,
            item.idx,
            item.observed_mzs,
            item.intensities,
            item.theoretical_mzs,
            item.left_comps,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
            item.site
        )
        for item in items
    ]
    (
        graphs,
        idxs,
        observed_mz,
        intensities,
        theoretical_mzs,
        left_compses,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
        sites
    ) = zip(*items)
    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)

    left_comps = torch.cat(left_compses)
    site = torch.cat(sites)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])
    return dict(
        graph=dgl.batch(graphs),
        idx=torch.LongTensor(idxs),
        theoretical_mz=torch.nn.utils.rnn.pad_sequence(theoretical_mzs, batch_first=True),
        observed_mz=torch.nn.utils.rnn.pad_sequence(observed_mz, batch_first=True),
        intensity=torch.nn.utils.rnn.pad_sequence(intensities, batch_first=True),
        left_comps=left_comps,
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree, 
        x=x,
        edge_input=edge_input,
        y=y,
        site=site
    )


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_unsqueeze(x, padlen):
    x = x + 1 
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)

def topological_equality(predict, traget):
    taken_b = set()
    b_children = list(traget.children(links=True))
    a_children = list(predict.children(links=True))
    if predict.mass() == traget.mass():
        for a_pos, a_link in a_children:
            a_child = a_link.child
            matched = False
            for b_pos, b_link in b_children:
                b_child = b_link.child
                if (b_pos, b_child.id) in taken_b:
                    continue
                if topological_equality(a_child,b_child):          
                    matched = True
                    taken_b.add((b_pos, b_child.id))
                    break
            if not matched and len(a_children) > 0:
                return False
        if len(taken_b) != len(b_children):  
            return False
        return True
    return False

def read_csv_files(csvfile):
    glycan_psm = {}
    tissue_name = ['MouseBrain', 'MouseHeart', 'MouseKidney', 'MouseLiver', 'MouseLung']
    num_fractions = 5
    with open(csvfile, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            tissue = row['Source File'].split('-')[0]
            tissue_id = tissue_name.index(tissue)
            fraction = int(row['Source File'].split('.')[0][-1])
            fraction_id = tissue_id * num_fractions + fraction
            psm_scan = row['Scan']
            scan = 'F' + str(fraction_id) + ':' + psm_scan
            glycan_psm[scan] = row
    return glycan_psm

def get_b_y_set(glycan, resolution):
    mass_free_reducing_end = 18.01056468370001
    glycan_clone = glycan.clone()
    glycan_b_set = set()
    glycan_y_set = set()

    for links, frags in itertools.groupby(glycan_clone.fragments(), lambda f: f.link_ids.keys()):
        y_ion, b_ion = list(frags)
        y_mass_reduced = y_ion.mass - mass_free_reducing_end
        b_mass_int = int(round(b_ion.mass * resolution))
        y_mass_int = int(round(y_mass_reduced * resolution))
        glycan_b_set.add(b_mass_int)
        glycan_y_set.add(y_mass_int)
    return glycan_b_set, glycan_y_set


def test_glycan_accuracy(target_glycans, predict_glycans, csvfile):
    print("\n---------------------------")
    print("Begin to evaluate the prediction accuracy")

    resolution = 1e3
    num_targets = float(len(target_glycans))
    num_target_y = 0.
    num_predict_y = 0.
    num_correct_y = 0.

    composition_matched = 0
    num_correct_glycans = 0

    composition_incorrect = []
    glycan_psm = read_csv_files(csvfile)
    dir = csvfile.split('.csv')[0]

    with open(dir+'_denovo.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        rand_psm = list(glycan_psm.keys())[0]
        csvwriter.writerow(list(glycan_psm[rand_psm].keys()) + ['Denovo Result'] + ['Denovo Glycan'])
        for idx, (target, candidate) in enumerate(zip(target_glycans, predict_glycans)):
            target_glycan, fraction_id, psm_scan, peptide_only_mass = target
            scan_id = 'F' + str(fraction_id) + ':' + str(psm_scan)
            psm = glycan_psm[scan_id]
            target_b_set, target_y_set = get_b_y_set(target_glycan, resolution)
            num_target_y += len(target_y_set)
            predict_b_set, predict_y_set = get_b_y_set(candidate, resolution) if candidate else (set(), set())
            correct_y_set = target_y_set.intersection(predict_y_set)  

            correct_glycan = 1 if topological_equality(candidate.root,target_glycan.root) else 0 

            num_predict_y += len(predict_y_set)                   
            num_correct_y += len(correct_y_set)                   
            num_correct_glycans += correct_glycan                 

            if candidate:
                if correct_glycan:
                    csvwriter.writerow(list(psm.values()) + ['1'] + [glypy_glycoct.dumps(candidate).replace('\n', ' ')])
                else:
                    csvwriter.writerow(list(psm.values()) + ['0'] + [glypy_glycoct.dumps(candidate).replace('\n', ' ')])
            else:
                csvwriter.writerow(list(psm.values()) + ['0'] + ['None'])
                
            pglyco_lst = [int(MonosaccharideResidue.from_monosaccharide(node).mass())for node in
                          candidate.iternodes()] if candidate is not None else None
            GF_lst = [int(MonosaccharideResidue.from_monosaccharide(node).mass()) for node in
                      target_glycan.iternodes()]
            pglyco_counter = collections.Counter(pglyco_lst)
            GF_counter = collections.Counter(GF_lst)
            if pglyco_counter == GF_counter:
                composition_matched += 1                         
            else:
                composition_incorrect.append(scan_id)             


    print("Number of predictions: ", int(num_targets))
    print("Number of correct glycan structures: ", num_correct_glycans)
    print("Number of correct glycan compositions: ", composition_matched)
    print("Structure metric = {:.3f}".format(num_correct_glycans / num_targets))
    print("Fragment ion metric = {:.3f}".format(num_correct_y / num_target_y))



def graph2glycan(graph, sugar_classes):
    num_nodes = graph.number_of_nodes()
    graph.ndata['x'][:,5]=0
    feature = graph.ndata['x'][1:-1] - graph.ndata['x'][2:]
    mono_list = torch.argmax(feature, dim=1)
    u, v = graph.edges(form='uv')
    u, v = u[2:] - 2, v[2:] - 2
    node_dict = {}
    for idx, node in enumerate(u):
        node = int(node)
        if node not in node_dict.keys():
            node_dict[node] = [int(v[idx])]
        else:
            node_dict[node].append(int(v[idx]))
    root = sugar_classes[mono_list[0]]
    glycan = glypy.Glycan(root=glypy.monosaccharides[root])
    try:
        for node in range(0, num_nodes-2):
            if node in node_dict.keys():
                for child in node_dict[node]:
                    child_mono = sugar_classes[mono_list[child]]
                    parent_mono = glycan[node]
                    parent_mono.add_monosaccharide(glypy.monosaccharides[child_mono])
                    glycan = glycan.reindex(method='bfs')
        leaves = list(glycan.leaves())
        glycan.canonicalize()
        for leaf in leaves:
            if MonosaccharideResidue.from_monosaccharide(leaf).residue_name() == 'Xyl':
                leaf.drop_monosaccharide(-1)
        glycan.canonicalize()
        return glycan
    except:
        return



class GraphormerDGLDataset(Dataset):
    def __init__(self,dataset: DGLDataset,seed: int = 0):
        self.dataset = dataset
        self.combination = dataset.combination
        self.seed = seed

    def __getitem__(self, idx):
        if isinstance(idx, int):
            result = self.dataset[idx]
            graph, y, left_comp,site,theoretical_mzs, observed_mzs, intensities = result
            
            return preprocess_dgl_graph(graph, y, idx, left_comp,site, theoretical_mzs, observed_mzs, intensities)
    
    def __len__(self) -> int:
        return len(self.dataset) 
    

def preprocess_dgl_graph(graph_data: DGLGraph, y: torch.Tensor, idx: int, left_comps,site, theoretical_mzs=None, observed_mzs=None, intensities=None) -> PYGGraph:
    N = graph_data.num_nodes()

    node_int_feature = graph_data.ndata['x'].clone().detach()
    edge_int_feature = torch.from_numpy(np.zeros(shape=[graph_data.num_edges(), 1])).long()

    edge_index = graph_data.edges()
    attn_edge_type = torch.zeros(
        [N, N, edge_int_feature.shape[1]], dtype=torch.long
    )
    attn_edge_type[
        edge_index[0].long(), edge_index[1].long()
    ] = convert_to_single_emb(edge_int_feature)
    dense_adj = graph_data.adj().to_dense().type(torch.int)
    dense_adj = dense_adj + dense_adj.t()
    shortest_path_result, path = algos.floyd_warshall(dense_adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  

    pyg_graph = PYGGraph()
    pyg_graph.graph_data = graph_data
    pyg_graph.x = convert_to_single_emb(node_int_feature, 512)
    pyg_graph.adj = dense_adj
    pyg_graph.site = site
    pyg_graph.attn_bias = attn_bias
    pyg_graph.attn_edge_type = attn_edge_type
    pyg_graph.spatial_pos = spatial_pos
    pyg_graph.in_degree = dense_adj.long().sum(dim=1).view(-1)
    pyg_graph.out_degree = pyg_graph.in_degree
    pyg_graph.edge_input = torch.from_numpy(edge_input).long()
    if y.dim() == 0:
        y = y.unsqueeze(-1)
    pyg_graph.y = y
    pyg_graph.idx = idx
    pyg_graph.left_comps = left_comps
    pyg_graph.theoretical_mzs = theoretical_mzs
    pyg_graph.observed_mzs = observed_mzs
    pyg_graph.intensities = intensities

    return pyg_graph


def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def create_all_glycan(glycan_dir):
    print('Generate training data')
    dir = glycan_dir.split('.pkl')[0]
    if not os.path.exists(dir):
        os.mkdir(dir)
    if not os.path.exists(dir+'/process.pkl'):
        with open(glycan_dir, 'rb') as f:
            glycan_dict = pickle.load(f)
            process = dict()
            for glycan_id in list(glycan_dict.keys()):
                glycan = glycan_dict[glycan_id]['GLYCAN'].clone()
                sugar_classes = ['Fuc', 'Man', 'GlcNAc', 'NeuAc', 'NeuGc']
                sugar_classes = [glypy.monosaccharides[name].mass()-mass_free_reducing_end for name in sugar_classes]
                compositions = [MonosaccharideResidue.from_monosaccharide(node).mass() for node in
                                glycan.iternodes(method='bfs')]
                compositions = [sugar_classes.index(mass) for mass in compositions]
                composition_counter = collections.Counter(compositions)
                glycan = glypy_glycoct.loads(glycan.serialize()).reindex(method='bfs')
                tree_glycopsm_list, labels, left_comps,site_list = cut2trees(glycan, sugar_classes)
                graphs = []
                label = []
                masses = []
                left_compositions = []
                sites = []

                for idx, tree in enumerate(tree_glycopsm_list[:-1]):
                    tree = glypy_glycoct.loads(tree)
                    left_comp = torch.tensor(left_comps[idx])
                    graph, site = tree_to_graph(tree, sugar_classes, left_comp,site_list[idx])

                    unordered_label = labels[idx].tolist()
                    graphs.append(graph)
                    label.append(unordered_label)
                    masses.append(tree.mass())
                    left_compositions.append(left_comp)
                    sites.append(site)

                edges_src, edges_dst = np.array([0]).astype('int'), np.array([1]).astype('int')
                root_G = dgl.graph((edges_src, edges_dst), num_nodes=2)
                initial_comp = torch.zeros((1, len(sugar_classes)+1))
                for k in composition_counter.keys():
                    initial_comp[:, k] = composition_counter[k]

                initial_comp = torch.cat((initial_comp, initial_comp))
                root_G.ndata['x'] = torch.tensor(initial_comp, dtype=torch.int32) 
                out_degree = root_G.out_degrees()
                leaf_nodes = torch.where(out_degree == 0)[0]
                root_G.ndata['x'][leaf_nodes, 5] = 1

                graphs.append(root_G)
                label.append(labels[-1].tolist())
                left_compositions.append(torch.tensor(left_comps[-1]))
                masses.append(0)
                sites.append(site_list[-1])
                
                process[glycan_id] = (graphs,left_compositions,sites,label,masses)

            with open(dir + "/" + "process.pkl", 'wb') as f:
                pickle.dump(process, f)


def cut2trees(target_glycan, sugar_classes):
    tree_glycopsm_list = []
    labels = []
    left_compositions = []
    site_list = []
    cur_node = target_glycan.index[-1]
    ori_comp = [MonosaccharideResidue.from_monosaccharide(node).mass() for node in target_glycan.iternodes(method='bfs')]
    ori_comp_lst = [sugar_classes.index(mass) for mass in ori_comp]
    ori_comp = collections.Counter(ori_comp_lst)
    while cur_node.parents():
        parent = cur_node.parents()[0][-1]
        site_list.append(parent.id)
        children = parent.children()
        cur_label = torch.zeros((1, len(sugar_classes)), dtype=torch.int32)
        for child in children:
            child_pos = child[0]
            parent.drop_monosaccharide(child_pos)
            mass = MonosaccharideResidue.from_monosaccharide(child[1]).mass()
            cur_label[:, sugar_classes.index(mass)] += 1
        labels.append(cur_label)
        target_glycan.reindex(method='bfs')
        tree_glycopsm_list.append(target_glycan.serialize())
        cur_comp = [MonosaccharideResidue.from_monosaccharide(node).mass() for node in target_glycan.iternodes(method='bfs')]
        cur_comp = [sugar_classes.index(mass) for mass in cur_comp]
        cur_comp = collections.Counter(cur_comp)
        ori_comp_copy = copy.deepcopy(ori_comp)
        comp_tensor = torch.zeros((1, len(sugar_classes)))
        for k in ori_comp_copy.keys():
            diff = ori_comp_copy[k]-cur_comp[k]
            comp_tensor[:, k] = diff
        left_compositions.append(comp_tensor)
        cur_node = target_glycan[-1]

    tree_glycopsm_list.append('<sos>')
    last_label = torch.zeros((1, len(sugar_classes)), dtype=torch.int32)
    last_sugar = sugar_classes.index(MonosaccharideResidue.from_monosaccharide(cur_node).mass())
    last_label[:, last_sugar] = 1
    labels.append(last_label)
    ori_comp_tensor = torch.zeros((1, len(sugar_classes)))
    for k in ori_comp.keys():
        ori_comp_tensor[:, k] = ori_comp[k]
    left_compositions.append(ori_comp_tensor)
    return tree_glycopsm_list, labels, left_compositions,site_list


def tree_to_graph(tree, sugar_classes, left_comp,site):
    nodes = dict()
    node_id_to_index = {}
    edges_src = [0, 1]
    edges_dst = [1, 2]
    num_nodes = len(tree.clone().index)
    cur_comp = []
    cur_comp = collections.Counter(cur_comp)
    tree = tree.clone(index_method='bfs')
    for node in tree.index[::-1]:
        node_index = num_nodes-len(nodes)+1
        node_id = node.id
        parents = node.parents()
        if parents:
            node.drop_monosaccharide(parents[0][0])
        node_name = MonosaccharideResidue.from_monosaccharide(node).mass()
        node_sugar_index = sugar_classes.index(node_name)
        comp_tensor = torch.zeros((len(sugar_classes)))
        for k in cur_comp.keys():
            comp_tensor[k] = cur_comp[k]
        cur_comp[node_sugar_index] += 1
        nodes[node_index] = {'id': node_id, 'name': node_name, 'sugar_index': node_sugar_index, 'left_comp': comp_tensor+left_comp}
        node_id_to_index[node_id] = node_index
    num_nodes = len(nodes)
    initial_comp = torch.zeros((len(sugar_classes)+1))
    for k in cur_comp.keys():
        initial_comp[k] = cur_comp[k]

    nodes_onehot = torch.stack([initial_comp+torch.cat((left_comp, torch.zeros(1, 1)), dim=1)]*2+[torch.cat((nodes[node]['left_comp'], torch.zeros(1, 1)), dim=1) for node in range(2, num_nodes+2)])
    nodes_onehot = nodes_onehot.view(-1, len(sugar_classes)+1).to(torch.int32)
    for link in tree.link_index:
        parent_index = node_id_to_index[link.parent.id]
        child_index = node_id_to_index[link.child.id]
        edges_src.append(parent_index)
        edges_dst.append(child_index)

    edges_src, edges_dst = np.array(edges_src).astype('int'), np.array(edges_dst).astype('int')
    graph = dgl.graph((edges_src, edges_dst), num_nodes=num_nodes+2)
    graph.ndata['x'] = torch.tensor(nodes_onehot, dtype=torch.int32)
    out_degree = graph.out_degrees()
    leaf_nodes = torch.where(out_degree == 0)[0]
    graph.ndata['x'][leaf_nodes, 5] = 1

    return graph,node_id_to_index[site] 


class GlycanDBCSV(DGLDataset):
    def __init__(self, args, ion_mass):
        self.graphs = []
        self.labels = []
        self.left_compositions = []
        self.sites = []
        self.observed_mzs = []
        self.intensities = []
        self.theoretical_mzs = []

        self.ion_mass = ion_mass
        self.csvfile_train = args.csv_file_train
        self.glycan_db = args.glycan_db
        with open(args.combination, 'rb') as f:
            self.combination = pickle.load(f)
        self.input_spectrum_file, self.spectrum_location_dict = spectrum_preprocessing(args)
        
        super().__init__(name='glycan_csv')

    def process(self):
        dir = self.csvfile_train.split('.csv')[0]
        with open(self.glycan_db.split('.pkl')[0] + "/process.pkl",'rb') as fr:
            process = pickle.load(fr)
        if not os.path.exists(dir):
            os.mkdir(dir)
        if not os.path.exists(dir+'/left_composition.pkl'):
            num_fractions = 5
            tissue_name = ['MouseBrain', 'MouseHeart', 'MouseKidney', 'MouseLiver', 'MouseLung']
            with open(self.csvfile_train, 'r') as csvfile:
                csvreader = csv.DictReader(csvfile)
                for row in csvreader:
                    tissue = row['Source File'].split('-')[0]
                    tissue_id = tissue_name.index(tissue)
                    fraction = int(row['Source File'].split('.')[0][-1])
                    fraction_id = tissue_id * num_fractions + fraction
                    psm_scan = row['Scan']
                    scan = 'F' + str(fraction_id) + ':' + psm_scan
                    precursor_mass = float(row['Mass'])
                    target_glycan_mass = float(row['Glycan Mass'])
                    adduct_mass = float(row['Adduct Mass'])
                    isotope_shift = float(row['Isotopic Shift'])
                    peptide_only_mass = precursor_mass - target_glycan_mass - adduct_mass + isotope_shift * mass_proton + mass_proton
                    target_glycan_id = row['Glycan ID']
                    mz, intensity = read_spectrum(self.input_spectrum_file, self.spectrum_location_dict, scan, peptide_only_mass)
                    (graphs,left_compositions,sites,label,masses)=process[target_glycan_id]

                    for idx in range(len(graphs)):
                        self.graphs.append(graphs[idx])
                        self.left_compositions.append(left_compositions[idx])
                        self.labels.append(self.combination.index(label[idx]))

                        if masses[idx] == 0:
                            current_mass = peptide_only_mass
                            theoretical_mz = torch.add(current_mass, self.ion_mass)
                        else:
                            current_mass = masses[idx] - mass_free_reducing_end + peptide_only_mass
                            theoretical_mz = torch.add(current_mass, self.ion_mass)
                        self.theoretical_mzs.append(torch.tensor(theoretical_mz))
                        self.observed_mzs.append(torch.tensor(mz))
                        self.intensities.append(torch.tensor(intensity))
                        self.sites.append(sites[idx])
            
            with open(dir + "/left_composition.pkl", 'wb') as fr:
                pickle.dump(self.left_compositions, fr)
            with open(dir + "/graphs.pkl", 'wb') as fr:
                pickle.dump(self.graphs, fr)
            with open(dir + "/labels.pkl", 'wb') as fr:
                pickle.dump(self.labels, fr)
            with open(dir + "/theoretical_mzs.pkl", 'wb') as fr:
                pickle.dump(self.theoretical_mzs, fr)
            with open(dir + "/observed_mzs.pkl", 'wb') as fr:
                pickle.dump(self.observed_mzs, fr)
            with open(dir + "/intensities.pkl", 'wb') as fr:
                pickle.dump(self.intensities, fr)
            with open(dir + "/sites.pkl", 'wb') as fr:
                pickle.dump(self.sites, fr)

        else:
            with open(dir+"/left_composition.pkl", 'rb') as fr:
                self.left_compositions = pickle.load(fr)
            with open(dir+"/graphs.pkl", 'rb') as fr:
                self.graphs = pickle.load(fr)
            with open(dir+"/labels.pkl", 'rb') as fr:
                self.labels = pickle.load(fr)
            with open(dir+"/theoretical_mzs.pkl", 'rb') as fr:
                self.theoretical_mzs = pickle.load(fr)
            with open(dir+"/intensities.pkl", 'rb') as fr:
                self.intensities = pickle.load(fr)
            with open(dir+"/observed_mzs.pkl", 'rb') as fr:
                self.observed_mzs = pickle.load(fr)
            with open(dir+"/sites.pkl", 'rb') as fr:
                self.sites = pickle.load(fr)

        print('Number of training data: ', len(self.graphs))


    def __getitem__(self, i):
        return self.graphs[i], torch.tensor(self.labels[i]), self.left_compositions[i], torch.tensor(self.sites[i]).unsqueeze(0),self.theoretical_mzs[i], self.observed_mzs[i], self.intensities[i]

    def __len__(self):
        return len(self.graphs)


def create_combination(args,glycan_dir):
    with open(glycan_dir, 'rb') as f:
        glycan_dict = pickle.load(f)

    combination = []
    for glycan_id in list(glycan_dict.keys()):  
        glycan = glycan_dict[glycan_id]['GLYCAN'].clone()
        sugar_classes_name = ['Fuc', 'Man', 'GlcNAc', 'NeuAc', 'NeuGc']
        sugar_classes = [glypy.monosaccharides[name].mass()-mass_free_reducing_end for name in sugar_classes_name]    

        glycan = glypy_glycoct.loads(glycan.serialize()).reindex(method='bfs')
        tree_glycopsm_list, labels, left_comps,site_list = cut2trees(glycan, sugar_classes)

        for idx, tree in enumerate(tree_glycopsm_list[:-1]):
            unordered_label = labels[idx].tolist()
            if  unordered_label not in combination:
                combination.append(unordered_label)

    with open(glycan_dir.split('.pkl')[0] + "combination.pkl", 'wb') as f:
        pickle.dump(combination, f)