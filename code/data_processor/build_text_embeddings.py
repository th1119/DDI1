"""
Physicochemical descriptor builder: RDKit molecular descriptors.
No transformers, no network, no external data required.

Usage:
  python data_processor/build_text_embeddings.py --dataset Ryu

Output:
  data/Ryu/text_embeddings.pt  (dict: entity_id -> 187-dim vector)
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys


def compute_molecular_descriptors(smiles):
    """20 physicochemical descriptors + 167 MACCS keys = 187-dim."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(187, dtype=np.float32)

    # Physicochemical descriptors (20-dim)
    desc = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumSaturatedRings(mol),
        Descriptors.NumAromaticHeterocycles(mol),
        Descriptors.NumAromaticCarbocycles(mol),
        Descriptors.NumAliphaticHeterocycles(mol),
        Descriptors.NumAliphaticCarbocycles(mol),
        Descriptors.NumSaturatedHeterocycles(mol),
        Descriptors.NumSaturatedCarbocycles(mol),
        Descriptors.LabuteASA(mol),
        Descriptors.BalabanJ(mol) if Descriptors.RingCount(mol) > 0 else 0.0,
    ]

    # MACCS keys (167-dim)
    try:
        maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
    except:
        maccs = np.zeros(167, dtype=np.float32)

    return np.concatenate([desc, maccs]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Ryu')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_name', type=str, default='text_embeddings.pt')
    args = parser.parse_args()

    main_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    data_dir = os.path.abspath(os.path.join(main_dir, '..', args.data_root, args.dataset))

    print(f"Data dir: {data_dir}")

    # 1. Load drug information
    smiles_path = os.path.join(data_dir, 'Drug_Information.txt')
    df = pd.read_csv(smiles_path, sep='\t', header=None, names=['entity', 'smiles'])
    df = df.dropna(subset=['entity', 'smiles'])
    df['entity'] = df['entity'].astype(str).str.replace(r'\.0+$', '', regex=True).str.strip()
    print(f"Loaded {len(df)} drugs")

    # 2. Compute descriptors
    embeddings = []
    entities = []
    failed = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing descriptors"):
        feat = compute_molecular_descriptors(row['smiles'])
        if feat.sum() == 0:
            failed += 1
        embeddings.append(feat)
        entities.append(row['entity'])

    embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Failed molecules: {failed}")

    # 3. Z-score normalization
    mean = embeddings.mean(dim=0, keepdim=True)
    std = embeddings.std(dim=0, keepdim=True).clamp(min=1e-8)
    embeddings = (embeddings - mean) / std

    # 4. Save
    text_emb_dict = {entities[i]: embeddings[i] for i in range(len(entities))}
    output_path = os.path.join(data_dir, args.output_name)
    torch.save(text_emb_dict, output_path)
    print(f"Saved to: {output_path}")
    print(f"Dict size: {len(text_emb_dict)}, dim: {embeddings.shape[1]}")


if __name__ == '__main__':
    main()
