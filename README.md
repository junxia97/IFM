# Intro
This is a Pytorch implementation of the NeurIPS paper “Understanding the Limitations of Deep Models for Molecular Property Prediction: Insights and Solutions” 

## Installation
We used the following Python packages for core development. We tested on `Python 3.7.4`.
```
pytorch                   1.13.1+cu117
scikit-learn              0.20.1
XGBoost                   1.6.2
DGL                       0.4.2
RDKit                     2022.09.5
hyperopt                  0.1.2
```

## Dataset download
All the necessary data files can be downloaded from the following link [Google Drive](https://drive.google.com/drive/folders/1ZYdYQ0TtmShJC-z6dr4BU1aPfeQSE9gD?usp=sharing), which include the molecules' SMILES strings, labels and various fingerprints. The SMILES vocabulary files can be found in [Github](https://github.com/DSPsleeporg/smiles-transformer/)

## Create the Directory for the results 

```
mkdir stat_res
```

## How to run this code 

To run this code, for `xgboost` you can use 


```
python xgb.py --data_label tox21 --runseed 43 --patience 50 --opt_iters 50 --repetitions 50 --num_pools 5
```

or for `rf`

```
python rf.py --data_label tox21 --runseed 43 --patience 50 --opt_iters 50 --repetitions 50 --num_pools 5
```

or for `svm`

```
python svm.py --data_label tox21 --runseed 43 --patience 50 --opt_iters 50 --repetitions 50 --num_pools 28
```

or for `gcn`

```
python gnn.py --data_label tox21 --epochs 300 --runseed 43 --batch_size 128 --patience 50 --opt_iters 50 --repetitions 50 --model_name gcn
```
or for `mlp`

```
python ifm_mlp.py --data_label tox21 --embed None --epochs 300 --runseed 43 --batch_size 128 --patience 50 --opt_iters 50 --repetitions 50
```
or for `IFM-MLP` 

```
python ifm_mlp.py --data_label tox21 --embed IFM --epochs 300 --runseed 43 --batch_size 128 --patience 50 --opt_iters 50 --repetitions 50
```


## Citation
```
@inproceedings{
xia2023understanding,
title={Understanding the Limitations of Deep Models for Molecular property prediction: Insights and Solutions},
author={Jun Xia and Lecheng Zhang and Xiao Zhu and Yue Liu and Zhangyang Gao and Bozhen Hu and Cheng Tan and Jiangbin Zheng and Siyuan Li and Stan Z. Li},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023}
}
```

## Acknowledgment
[1] SMILES Transformer: Pre-trained Molecular Fingerprint for Low Data Drug Discovery (Honda et al., Arixv 2019)           
[2] Exposing the limitations of molecular machine learning with activity cliffs (Tilborg et al., JCIM 2022)   


