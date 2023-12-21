from rdkit import Chem
import datetime
import dgl
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import model_zoo
from dgl.data.chem import one_hot_encoding
from dgl.data.utils import split_dataset
from sklearn.metrics import roc_auc_score
from dgl.data.chem import BaseAtomFeaturizer, BaseBondFeaturizer, ConcatFeaturizer, atom_type_one_hot, atom_degree_one_hot, \
    atom_implicit_valence_one_hot, atom_formal_charge, atom_num_radical_electrons, atom_hybridization_one_hot, \
    atom_is_aromatic, atom_total_num_H_one_hot, atom_total_degree_one_hot, atom_chiral_tag_one_hot, atom_mass, \
    bond_type_one_hot, bond_is_conjugated, bond_is_in_ring, bond_stereo_one_hot
from functools import partial
import dgl
from dgl.nn.pytorch.conv import RelGraphConv
from dgl.model_zoo.chem.classifiers import MLPBinaryClassifier
from dgl.nn.pytorch.glob import WeightAndSum
from dgl import BatchedDGLGraph
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc, \
    mean_absolute_error, r2_score, matthews_corrcoef


def chirality(atom):  # the chirality information defined in the AttentiveFP
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


AttentiveFPAtomFeaturizer = BaseAtomFeaturizer(
    featurizer_funcs={'h': ConcatFeaturizer([
        partial(atom_type_one_hot, allowable_set=[
            'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At'],
                encode_unknown=True),
        partial(atom_degree_one_hot, allowable_set=list(range(6))),
        atom_formal_charge, atom_num_radical_electrons,
        partial(atom_hybridization_one_hot, encode_unknown=True),
        atom_is_aromatic,  # A placeholder for aromatic information,
        atom_total_num_H_one_hot, chirality
    ],
    )})

AttentiveFPBondFeaturizer = BaseBondFeaturizer(featurizer_funcs={'e': ConcatFeaturizer([bond_type_one_hot,
                                                                                        bond_is_conjugated,
                                                                                        bond_is_in_ring,
                                                                                        partial(bond_stereo_one_hot,
                                                                                                allowable_set=[Chem.rdchem.BondStereo.STEREONONE,
                                                                                                               Chem.rdchem.BondStereo.STEREOANY,
                                                                                                               Chem.rdchem.BondStereo.STEREOZ,
                                                                                                               Chem.rdchem.BondStereo.STEREOE], encode_unknown=True)])})


class MyAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_field='h'):
        super(MyAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [atom_type_one_hot,
                 atom_degree_one_hot,
                 atom_implicit_valence_one_hot,
                 atom_formal_charge,
                 atom_num_radical_electrons,
                 atom_hybridization_one_hot,
                 atom_is_aromatic,
                 atom_total_num_H_one_hot,
                 atom_total_degree_one_hot,
                 atom_chiral_tag_one_hot,
                 atom_mass]
            )})


def set_random_seed(seed=0):
    """Set random seed.

    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration

        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        #print(y_pred.shape)
        #print(y_true.shape)
        self.mask.append(mask.detach().cpu())

    def roc_precision_recall_score(self):
        """Compute AUC_PRC for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_pred = torch.sigmoid(y_pred)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            precision, recall, _thresholds = precision_recall_curve(task_y_true, task_y_pred, pos_label=1)
            scores.append(auc(recall, precision))
        return scores

    def roc_auc_score(self):
        """Compute roc-auc score for each task.

        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # Todo: support categorical classes
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)  # 求得为正例的概率
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(roc_auc_score(task_y_true, task_y_pred))
        return scores

    def l1_loss(self, reduction):
        """Compute l1 loss for each task.

        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(F.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())
        return scores

    def rmse(self):
        """Compute RMSE for each task.

        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        #print(y_pred.shape)
        #print(y_pred)
        y_true = torch.cat(self.y_true, dim=0)
        #print(y_true.shape)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            #print(str(n_tasks)+'/'+str(task)+" "+str(task_w))
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(np.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item()))
        return scores

    def mae(self):
        """Compute mae for each task.

        Returns
        -------
        list of float
            mae for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(mean_absolute_error(task_y_true, task_y_pred))
        return scores

    def r2(self):
        """Compute r2 score for each task.

        Returns
        -------
        list of float
            r2 score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(r2_score(task_y_true, task_y_pred))
        return scores

    def compute_metric(self, metric_name, reduction='mean'):
        """Compute metric for each task.

        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task

        Returns
        -------
        list of float
            Metric value for each task
        """
        assert metric_name in ['roc_auc', 'l1', 'rmse', 'prc_auc', 'mae', 'r2'], \
            'Expect metric name to be "roc_auc", "l1", "rmse", "prc_auc", "mae", "r2" got {}'.format(metric_name)  # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常
        assert reduction in ['mean', 'sum']
        if metric_name == 'roc_auc':
            return self.roc_auc_score()
        if metric_name == 'l1':
            return self.l1_loss(reduction)
        if metric_name == 'rmse':
            return self.rmse()
        if metric_name == 'prc_auc':
            return self.roc_precision_recall_score()
        if metric_name == 'mae':
            return self.mae()
        if metric_name == 'r2':
            return self.r2()
        if metric_name == 'mcc':
            return self.mcc()


class EarlyStopping(object):
    """Early stop performing

    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
    patience : int
        Number of epochs to wait before early stop
        if the metric stops getting improved
    filename : str or None
        Filename for storing the model checkpoint
    """
    def __init__(self, mode='higher', patience=10, filename=None):
        if filename is None:
            dt = datetime.datetime.now()
            filename = '{}_early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second)
            print(filename)
        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        dt = datetime.datetime.now()
        filename = '{:02d}-{:02d}-{:02d}.pth'.format( dt.hour, dt.minute, dt.second)
            
        self.filename = filename
        
        print(filename)
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):  # 当前模型如果是更优模型，则保存当前模型
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])

def collate_molgraphs(data):
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks


