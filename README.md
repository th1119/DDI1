# Reasoning over Relations: Modeling Inter-Relational Structure for Drug-Drug Interaction Prediction

Official code.

## Usage

### Transductive — Ryu

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29501 train.py \
  --experiment_name default_v3_1_fold1 \
  --dataset Ryu --batch_size 4 --accum_steps 4 --num_negatives 1 \
  --add_transpose_rels --Folds 1 --fold_index 1 \
  --num_epochs 15 --early_stop_epoch 100 --eval_every_iter 2000 \
  --label_smoothing 0.03 --lr 5e-4 --weight_decay_rate 5e-5 \
  --emb_dim 256 --kg_emb_dim 64 --gcn_dropout 0.2 \
  --gamma_focal 2.5 --beta 0.999 \
  --use_uncond --use_pair_attention --use_pair_context \
  --use_desc --desc_dim 187 --max_neighbors 64 \
  --disable_kg_contrast
```

### Transductive — DrugBank

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29501 train.py \
  --experiment_name default_v3_1_drugbank_fold1 \
  --dataset DrugBank --batch_size 4 --accum_steps 4 --num_negatives 1 \
  --add_transpose_rels --Folds 1 --fold_index 1 \
  --num_epochs 15 --early_stop_epoch 100 --eval_every_iter 2000 \
  --label_smoothing 0.03 --lr 5e-4 --weight_decay_rate 5e-5 \
  --emb_dim 256 --kg_emb_dim 64 --gcn_dropout 0.2 \
  --gamma_focal 2.5 --beta 0.999 \
  --use_uncond --use_pair_attention --use_pair_context \
  --use_desc --desc_dim 187 --max_neighbors 64 \
  --disable_kg_contrast
```

### Inductive S1 — Ryu

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29501 train.py \
  --experiment_name default_v3_1_S1_fold1 \
  --dataset Ryu --batch_size 4 --accum_steps 4 --num_negatives 1 \
  --add_transpose_rels --Folds 1 --fold_index 1 \
  --num_epochs 15 --early_stop_epoch 10 --eval_every_iter 2000 \
  --label_smoothing 0.03 --lr 5e-4 --weight_decay_rate 5e-5 \
  --emb_dim 256 --kg_emb_dim 64 --gcn_dropout 0.2 \
  --gamma_focal 2.5 --beta 0.999 \
  --use_uncond --use_pair_attention --use_pair_context \
  --use_desc --desc_dim 187 --max_neighbors 64 \
  --prop_drop_prob 0.3 \
  --inductive --inductive_setting S1 \
  --disable_kg_contrast
```

### Inductive S2 — Ryu

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29501 train.py \
  --experiment_name default_v3_1_S2_fold1 \
  --dataset Ryu --batch_size 4 --accum_steps 4 --num_negatives 1 \
  --add_transpose_rels --Folds 1 --fold_index 1 \
  --num_epochs 15 --early_stop_epoch 10 --eval_every_iter 2000 \
  --label_smoothing 0.03 --lr 5e-4 --weight_decay_rate 5e-5 \
  --emb_dim 256 --kg_emb_dim 64 --gcn_dropout 0.2 \
  --gamma_focal 2.5 --beta 0.999 \
  --use_uncond --use_pair_attention --use_pair_context \
  --use_desc --desc_dim 187 --max_neighbors 64 \
  --prop_drop_prob 0.3 \
  --inductive --inductive_setting S2 \
  --disable_kg_contrast
```

Other folds: set `--fold_index 2..5` and rename `--experiment_name` accordingly (e.g. `default_v3_1_fold2`).
