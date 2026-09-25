import numpy as np
from rdkit.Chem import MolFromSmiles, MolToSmiles, rdmolops, rdFingerprintGenerator
from .features import atom_features, encode_bond_15
import rdkit.Chem as Chem
from jarvis.core.specie import get_node_attributes
from collections import defaultdict
from itertools import combinations
import torch
import dgl


def get_mol(smiles, addH=False):
    if not isinstance(smiles, str):
        smiles = str(smiles)
    if smiles.strip().lower() in ('nan', 'none', '', 'null'):
        raise ValueError(f"Invalid/Empty SMILES encountered: {repr(smiles)}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles}")

    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if addH:
        mol = Chem.AddHs(mol)
    return mol


def node_features(mol, use_bond_feature=True, encoder_atom=True):
    atom_feature = {}
    atom_neighbor = defaultdict(list)
    bond_feature_dict = {}

    for atom in mol.GetAtoms():
        if encoder_atom:
            atom_feat = np.array(get_node_attributes(atom.GetSymbol(), atom_features="cgcnn"))
        else:
            atom_feat = atom_features(atom, explicit_H=False)
        atom_feature[atom.GetIdx()] = atom_feat

    atom_total_features = []

    if use_bond_feature:
        for bond in mol.GetBonds():
            bond_id = bond.GetIdx()
            bond_feat = encode_bond_15(bond)
            bond_feature_dict[bond_id] = bond_feat

            atom_neighbor[bond.GetBeginAtomIdx()].append(bond_id)
            atom_neighbor[bond.GetEndAtomIdx()].append(bond_id)

        for atom_idx in sorted(atom_feature.keys()):
            atom_feat = atom_feature[atom_idx]
            bond_feat = []

            for bond_id in atom_neighbor[atom_idx]:
                bond_feat.append(bond_feature_dict[bond_id])

            if bond_feat:
                avg_bond_feat = np.mean(bond_feat, axis=0)
            else:
                avg_bond_feat = np.zeros(15)
            atom_feat = np.concatenate([atom_feat, avg_bond_feat])

            atom_total_features.append(list(atom_feat))
    else:
        for atom_idx in sorted(atom_feature.keys()):
            atom_feat = atom_feature[atom_idx]
            atom_total_features.append(list(atom_feat))

    atom_total_features_tensor = torch.tensor(atom_total_features, dtype=torch.float32)

    return atom_total_features_tensor


def smiles_to_graph(smiles):
    mol = get_mol(smiles, addH=False)
    if mol is None:
        raise ValueError("Could not parse SMILES string: {}".format(smiles))

    src_list, dst_list = [], []
    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])

    num_atoms = int(mol.GetNumAtoms())
    node_feats = node_features(mol, use_bond_feature=True, encoder_atom=True)
    atom_feats_size = node_feats.size(1)

    g = dgl.graph((torch.tensor(src_list, dtype=torch.long),
                   torch.tensor(dst_list, dtype=torch.long)),
                  num_nodes=num_atoms)
    g.ndata['feat'] = node_feats
    return g, atom_feats_size


def get_morgan_fingerprint(smiles, radius=2, nBits=2048):
    mol = MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(nBits, dtype=torch.float32)

    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fp_array = fp_gen.GetFingerprintAsNumPy(mol)

    return torch.from_numpy(fp_array).to(dtype=torch.float32)


def frag_graph(smiles):
    mol = get_mol(smiles, addH=False)
    fragmented_mol = Chem.FragmentOnBRICSBonds(mol)
    atom_list = Chem.GetMolFrags(fragmented_mol, sanitizeFrags=True)
    fragment_mols = Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=True)
    fragments = [MolToSmiles(x, False) for x in fragment_mols]

    adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)

    fragment_bonds = set()
    for frag1, frag2 in combinations(range(len(atom_list)), 2):
        for atom1 in atom_list[frag1]:
            for atom2 in atom_list[frag2]:
                if atom1 < mol.GetNumAtoms() and atom2 < mol.GetNumAtoms():
                    if adjacency_matrix[atom1, atom2] == 1 or atom1 == atom2:
                        fragment_bonds.add((frag1, frag2))
                        fragment_bonds.add((frag2, frag1))

    if fragment_bonds:
        src, dst = zip(*fragment_bonds)
        graph = dgl.graph((src, dst), num_nodes=len(fragments))
    else:
        graph = dgl.graph(([], []), num_nodes=len(fragments))

    fragment_features = []
    atom_matrix = np.zeros((len(fragments), mol.GetNumAtoms()), dtype=int)

    for i, frag_smiles in enumerate(fragments):
        feature = get_morgan_fingerprint(frag_smiles)
        fragment_features.append(feature)

        for atom in atom_list[i]:
            if atom < mol.GetNumAtoms():
                atom_matrix[i, atom] = 1

    fragment_features = torch.stack(fragment_features)

    graph.ndata['feat'] = fragment_features
    graph.ndata['atom_indices'] = torch.tensor(atom_matrix, dtype=torch.float32)

    if graph.number_of_edges() > 0:
        graph.edata['type'] = torch.ones(graph.number_of_edges(), dtype=torch.int32)
    else:
        graph.edata['type'] = torch.empty((0,), dtype=torch.int32)

    return graph, fragment_features.size(1)


def pre_dgl_graphs(id2smiles):
    molecular_graphs = {}
    frag_graphs = {}

    atom_feats_size = 0
    frag_feats_size = 0

    if not id2smiles:
        raise RuntimeError(
            "id2smiles is empty! "
            "Please check: 1) Drug_Information.txt coverage, 2) entity mapping logic in datasets.py"
        )

    for eid, smiles in id2smiles.items():
        molecular_graphs[eid], atom_feats_size = smiles_to_graph(smiles)
        frag_graphs[eid], frag_feats_size = frag_graph(smiles)

    return molecular_graphs, frag_graphs, atom_feats_size, frag_feats_size


def merge_graphs(graph1, graph2):
    n1 = int(graph1.number_of_nodes())
    n2 = int(graph2.number_of_nodes())

    s1, d1 = [torch.as_tensor(t, dtype=torch.long, device='cpu') for t in graph1.edges()]
    s2, d2 = [torch.as_tensor(t, dtype=torch.long, device='cpu') for t in graph2.edges()]

    # Intra-drug edges (graph2 node indices offset by n1)
    inter_src = torch.cat([s1, s2 + n1])
    inter_dst = torch.cat([d1, d2 + n1])

    # Cross edges (Cartesian product full connection)
    cross_src_fwd = torch.arange(n1, dtype=torch.long).repeat_interleave(n2)
    cross_dst_fwd = torch.arange(n1, n1 + n2, dtype=torch.long).repeat(n1)
    cross_full_src = torch.cat([cross_src_fwd, cross_dst_fwd])
    cross_full_dst = torch.cat([cross_dst_fwd, cross_src_fwd])

    # Build heterogeneous graph
    merged_graph = dgl.heterograph({
        ('drug', 'inter', 'drug'): (inter_src, inter_dst),
        ('drug', 'cross', 'drug'): (cross_full_src, cross_full_dst)
    }, num_nodes_dict={'drug': n1 + n2})

    # Concatenate features
    merged_graph.ndata['feat'] = torch.cat([
        graph1.ndata['feat'].cpu(),
        graph2.ndata['feat'].cpu()
    ], dim=0)

    return merged_graph
