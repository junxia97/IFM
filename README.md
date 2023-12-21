# Intro
This is a Pytorch implementation of the NeurIPS paper “Understanding the Limitations of Deep Models for Molecular Property Prediction: Insights and Solutions”: 

## Installation
We used the following Python packages for core development. We tested on `Python 3.7.4`.
```
pytorch                   1.13.1+cu117
scikit-learn              0.20.1
XGBoost                   0.80
DGL                       0.4.1
RDKit                     2019.09.1
hyperopt                  0.2
```

## Dataset download
All the necessary data files can be downloaded from the following link [Google Drive](https://drive.google.com/drive/folders/1ZYdYQ0TtmShJC-z6dr4BU1aPfeQSE9gD?usp=sharing), which include the molecules' SMILES strings, labels and various fingerprints. The SMILES vocabulary files can be found in [Github](https://github.com/DSPsleeporg/smiles-transformer/)


## Benchmarking Experiments
Take the MLP model as an example,
```
python mlp.py
```
The smiles indlude the codes for models that take SMILES as input.

## Feature Embedding Methods for Molecules
Take the MLP model as an example,
```
python emlp.py --embed EMBED --data_label DATASET
```
This will train and evaluate the mlp model on the dataset `DATASET` using the feature embedding method `EMBED`. The results of vanilla model without embedding method can be reproduced by setting `EMBED == None`.

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
[3] Maxsmi: maximizing molecular property prediction performance with confidence estimation using smiles augmentation and deep learning. (Kimber et al., Artificial Intelligence in the Life Sciences 2021)

