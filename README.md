# FTF-ER

Feature-Topology Fusion-Based Experience Replay Method for Continual Graph Learning

(ACM MM 2024)

## Requirement

* python==3.8.17
* scipy==1.9.3
* numpy==1.23.5
* torch==1.13.1
* networkx==3.1
* scikit-learn==1.1.3
* matplotlib==3.7.4
* ogb==1.3.6
* dgl==0.9.1
* petsc4py==3.20.0

## Usage

```
python train.py --dataset CoraFull-CL --gpu 0 --method FTF_ER --ftfer_args budget:60\;sampler:mix\;beta:0.5 --backbone GAT --minibatch False
```

```
python train.py  --dataset Reddit-CL --gpu 0 --method FTF_ER --ftfer_args budget:100\;sampler:mix\;beta:0.5 --backbone GCN --minibatch True --batch_size 20000
```
