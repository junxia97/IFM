import torch
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import datetime
import random
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc, \
    mean_absolute_error, r2_score, matthews_corrcoef
import pickle

def init_parameter_uniform(parameter: nn.Parameter, n: int) -> None:
    nn.init.uniform_(parameter, -1/np.sqrt(n), 1/np.sqrt(n))

def statistical(y_true, y_pred, y_pro):
    c_mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = list(c_mat.flatten())
    se = tp/(tp+fn)
    sp = tn/(tn+fp)
    acc = (tp+tn)/(tn+fp+fn+tp)
    mcc = (tp*tn-fp*fn)/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)+1e-8)
    auc_prc = auc(precision_recall_curve(y_true, y_pro, pos_label=1)[1],
                  precision_recall_curve(y_true, y_pro, pos_label=1)[0])
    auc_roc = roc_auc_score(y_true, y_pro)
    return tn, fp, fn, tp, se, sp, acc, mcc, auc_prc, auc_roc

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
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
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

class MyDataset(object):
    def __init__(self, Xs, Ys):
        self.Xs = torch.tensor(Xs, dtype=torch.float32)
        self.masks = torch.tensor(~np.isnan(Ys) * 1.0, dtype=torch.float32)
        # convert np.nan to 0
        self.Ys = torch.tensor(np.nan_to_num(Ys), dtype=torch.float32)


    def __len__(self):
        return len(self.Ys)

    def __getitem__(self, idx):

        X = self.Xs[idx]
        Y = self.Ys[idx]
        mask = self.masks[idx]

        return X, Y, mask

class EarlyStopping(object):

    def __init__(self, mode='higher', patience=10, filename=None): 
        if filename is None:
            dt = datetime.datetime.now()
            filename = '{}_early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher  
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
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
            # print(
            #     f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])

class MyDNN(nn.Module):
    def __init__(self, inputs, hidden_units, outputs, dp_ratio, reg):
        """
        :param inputs: number of inputs
        :param hidden_units: [128, 256, 512]
        :param out_puts: number of outputs
        :param dp_ratio:
        :param reg:
        """
        super(MyDNN, self).__init__()
        # parameters
        self.reg = reg

        # layers
        self.hidden1 = nn.Linear(inputs, hidden_units[0])
        self.dropout1 = nn.Dropout(dp_ratio)

        self.hidden2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.dropout2 = nn.Dropout(dp_ratio)

        self.hidden3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.dropout3 = nn.Dropout(dp_ratio)

        if reg:
            self.output = nn.Linear(hidden_units[2], 1)
        else:
            self.output = nn.Linear(hidden_units[2], outputs)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(self.dropout1(x))

        x = self.hidden2(x)
        x = F.relu(self.dropout2(x))

        x = self.hidden3(x)
        x = F.relu(self.dropout3(x))

        return self.output(x)

class LE(nn.Module):
    def __init__(self, n_tokens: int, d_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_tokens, 1, d_out))
        self.bias = nn.Parameter(torch.Tensor(n_tokens, d_out))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        d_out = self.weight.shape[-1]
        init_parameter_uniform(self.weight, d_out)
        init_parameter_uniform(self.bias, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (n_batch, n_features, d_in)
        returns: (n_batch, n_features, d_out)
        """
        x = x.unsqueeze(-1)
        x = (x.unsqueeze(-2)@self.weight[None]).squeeze(-2)
        x = x + self.bias[None]
        return x

class PLE(nn.Module):
    def __init__(self, n_num_features: int, d_out: int, sigma: float) -> None:
        super().__init__()
        self.d_out = d_out
        self.sigma = sigma
        coefficients = torch.Tensor(n_num_features, d_out)
        self.coefficients = nn.Parameter(coefficients)
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.normal_(self.coefficients, 0.0, self.sigma)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = 2*np.pi*self.coefficients[None]*x[..., None]
        return torch.cat([torch.cos(x), torch.sin(x)], -1)

class LE_DNN(nn.Module):
    def __init__(self, inputs, hidden_units, outputs, d_out, dp_ratio, reg):
        super(LE_DNN, self).__init__()
        # parameters
        self.reg = reg
        # layers
        self.hidden1 = nn.Linear(inputs * d_out, hidden_units[0])
        self.dropout1 = nn.Dropout(dp_ratio)

        self.hidden2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.dropout2 = nn.Dropout(dp_ratio)

        self.hidden3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.dropout3 = nn.Dropout(dp_ratio)

        if reg:
            self.output = nn.Linear(hidden_units[2], 1)
        else:
            self.output = nn.Linear(hidden_units[2], outputs)
        self.embedding = LE(inputs, d_out)

    def forward(self, x):

        x = self.embedding(x).view(x.size(0), -1)
        x = self.hidden1(x)
        x = F.relu(self.dropout1(x))

        x = self.hidden2(x)
        x = F.relu(self.dropout2(x))

        x = self.hidden3(x)
        x = F.relu(self.dropout3(x))

        return self.output(x)

class LSIM_DNN(nn.Module):
    def __init__(self, inputs, hidden_units, outputs, d_out, sigma, dp_ratio, reg):
        super(LSIM_DNN, self).__init__()
        # parameters
        self.reg = reg
        # layers
        self.hidden1 = nn.Linear(inputs, hidden_units[0])
        self.dropout1 = nn.Dropout(dp_ratio)

        self.hidden2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.dropout2 = nn.Dropout(dp_ratio)

        self.hidden3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.dropout3 = nn.Dropout(dp_ratio)

        if reg:
            self.output = nn.Linear(hidden_units[2], 1)
        else:
            self.output = nn.Linear(hidden_units[2], outputs)
        self.embedding = PLE(inputs, d_out, sigma)
        self.linear = nn.Linear(d_out * 2, inputs)

    def forward(self, x):
        x = self.embedding(x).sum(1)
        x = F.relu(self.linear(x))
        x = self.hidden1(x)
        x = F.relu(self.dropout1(x))

        x = self.hidden2(x)
        x = F.relu(self.dropout2(x))

        x = self.hidden3(x)
        x = F.relu(self.dropout3(x))

        return self.output(x)

class gaussian_encoding(nn.Module):   
    def __init__(self, n_num_features: int, d_out: int, sigma: float) -> None:
        super().__init__()
        self.d_out = d_out
        self.sigma = sigma
        self.n_num_features = n_num_features
        self.size = (d_out, n_num_features)
        self.B = torch.randn(self.size) * sigma
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (n_batch, n_features)
        returns: (n_batch, n_features * 2 * d_out)
        """
        self.B = self.B.to(x.device)
        xp = 2 * np.pi * x @ self.B.T
        return torch.cat((torch.cos(xp), torch.sin(xp)), dim=-1)
    
class GM_DNN(nn.Module):
    def __init__(self, inputs, hidden_units, outputs, d_out, sigma, dp_ratio, reg):
        """
        :param inputs: number of inputs
        :param hidden_units: [128, 256, 512]
        :param out_puts: number of outputs
        :param dp_ratio:
        :param reg:
        """
        super(GM_DNN, self).__init__()
        # parameters
        self.reg = reg
        self.d_out = d_out
        self.sigma = sigma
        # layers
        self.hidden1 = nn.Linear(d_out * 2, hidden_units[0])
        self.dropout1 = nn.Dropout(dp_ratio)

        self.hidden2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.dropout2 = nn.Dropout(dp_ratio)

        self.hidden3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.dropout3 = nn.Dropout(dp_ratio)

        if reg:
            self.output = nn.Linear(hidden_units[2], 1)
        else:
            self.output = nn.Linear(hidden_units[2], outputs)

        self.embedding = gaussian_encoding(inputs, d_out, sigma)


    def forward(self, x):
        x = self.embedding(x)
        x = self.hidden1(x)
        x = F.relu(self.dropout1(x))

        x = self.hidden2(x)
        x = F.relu(self.dropout2(x))

        x = self.hidden3(x)
        x = F.relu(self.dropout3(x))

        return self.output(x)
    
class SineLayer(nn.Module):    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class IFM_DNN(nn.Module):
    def __init__(self, inputs, hidden_units, outputs, d_out, sigma, dp_ratio, first_omega_0, hidden_omega_0, reg):
        """
        :param inputs: number of inputs
        :param hidden_units
        :param out_puts: number of outputs
        :param dp_ratio:
        :param reg:
        """
        super(IFM_DNN, self).__init__()
        # parameters
        self.reg = reg
        # layers
        self.hidden1 = SineLayer(inputs, hidden_units[0], is_first=True, omega_0=first_omega_0)
        self.dropout1 = nn.Dropout(dp_ratio)

        self.hidden2 = SineLayer(hidden_units[0], hidden_units[1], is_first=False, omega_0=hidden_omega_0)
        self.dropout2 = nn.Dropout(dp_ratio)

        self.hidden3 = SineLayer(hidden_units[1], hidden_units[2], is_first=False, omega_0=hidden_omega_0)
        self.dropout3 = nn.Dropout(dp_ratio)

        if reg:
            self.output = nn.Linear(hidden_units[2], 1)
            with torch.no_grad():
                self.output.weight.uniform_(-np.sqrt(6 / hidden_units[2]) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_units[2]) / hidden_omega_0)
        else:
            self.output = nn.Linear(hidden_units[2], outputs)
            with torch.no_grad():
                self.output.weight.uniform_(-np.sqrt(6 / hidden_units[2]) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_units[2]) / hidden_omega_0)

    def forward(self, x):

        x = self.hidden1(x)
        x = F.relu(self.dropout1(x))
        x = self.dropout1(x)

        x = self.hidden2(x)
        x = F.relu(self.dropout2(x))
        x = self.dropout2(x)

        x = self.hidden3(x)
        x = F.relu(self.dropout3(x))
        x = self.dropout3(x)

        return self.output(x)

class SIM_encoding(nn.Module):
    def __init__(self, n_num_features: int, d_out: int, sigma: float) -> None:
        super().__init__()
        self.d_out = d_out
        self.sigma = sigma
        self.n_num_features = n_num_features
        self.coeffs = 2 * np.pi * sigma ** (torch.arange(d_out) / d_out)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (n_batch, n_features)
        returns: (n_batch, n_features * 2 * d_out)
        """
        xp = self.coeffs.to(x.device) * torch.unsqueeze(x, -1)
        xp_cat = torch.cat((torch.cos(xp), torch.sin(xp)), dim=-1)
        return xp_cat.flatten(-2, -1) 

class SIM_DNN(nn.Module):
    def __init__(self, inputs, hidden_units, outputs, d_out, sigma, dp_ratio, reg):
        """
        :param inputs: number of inputs
        :param hidden_units
        :param out_puts: number of outputs   m
        :param dp_ratio:
        :param reg:
        """
        super(SIM_DNN, self).__init__()
        # parameters
        self.reg = reg
        self.d_out = d_out
        self.sigma = sigma
        # layers
        self.hidden1 = nn.Linear(d_out * 2 * inputs, hidden_units[0])
        self.dropout1 = nn.Dropout(dp_ratio)

        self.hidden2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.dropout2 = nn.Dropout(dp_ratio)

        self.hidden3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.dropout3 = nn.Dropout(dp_ratio)

        if reg:
            self.output = nn.Linear(hidden_units[2], 1)
        else:
            self.output = nn.Linear(hidden_units[2], outputs)
        self.embedding = SIM_encoding(inputs, d_out, sigma)

    def forward(self, x):
        x = self.embedding(x)
        x = self.hidden1(x)
        x = F.relu(self.dropout1(x))

        x = self.hidden2(x)
        x = F.relu(self.dropout2(x))

        x = self.hidden3(x)
        x = F.relu(self.dropout3(x))

        return self.output(x)

def collate_fn(data_batch):
    Xs, Ys, masks = map(list, zip(*data_batch))

    Xs = torch.stack(Xs, dim=0)
    Ys = torch.stack(Ys, dim=0)
    masks = torch.stack(masks, dim=0)

    return Xs, Ys, masks

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子