import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pretrain_trfm import e_TrfmSeq2seq, TrfmSeq2seq
from build_vocab import WordVocab
from utils import split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
from dnn_torch_utils import Meter, MyDataset, EarlyStopping, MyDNN, collate_fn, set_random_seed
from hyperopt import fmin, tpe, hp, rand, STATUS_OK, Trials, partial
import sys
import copy
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, MSELoss
import gc
import time
start_time = time.time()
from sklearn.model_selection import train_test_split
import warnings
from dataset import Seq2seqDataset

def get_inputs(sm):
    seq_len = 220
    sm = sm.split()
    if len(sm)>218:
        print('SMILES is too long ({:d})'.format(len(sm)))
        sm = sm[:109]+sm[-109:]
    ids = [vocab.stoi.get(token, unk_index) for token in sm]
    ids = [sos_index] + ids + [eos_index]
    seg = [1]*len(ids)
    padding = [pad_index]*(seq_len - len(ids))
    ids.extend(padding), seg.extend(padding)
    return ids, seg

def get_array(smiles):
    x_id, x_seg = [], []
    for sm in smiles:
        a,b = get_inputs(sm)
        x_id.append(a)
        x_seg.append(b)
    return torch.tensor(x_id), torch.tensor(x_seg)

def bit2np(bitvector):
    bitstring = bitvector.ToBitString()
    intmap = map(int, bitstring)
    return np.array(list(intmap))

def extract_morgan(smiles, targets):
    x,X,y = [],[],[]
    for sm,target in zip(smiles,targets):
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            print(sm)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) # Morgan (Similar to ECFP4)
        x.append(sm)
        X.append(bit2np(fp))
        y.append(target)
    return x,np.array(X),np.array(y)

def ablation_hiv(X, X_test, y, y_test, rate, n_repeats):
    roc = np.empty(n_repeats)
    prc = np.empty(n_repeats)
    for i in range(n_repeats):
        clf = MLPClassifier(max_iter=1000)
        if rate == 1:
            X_train, y_train = X, y
        else:
            X_train, _, y_train, __ = train_test_split(X, y, test_size=1 - rate, stratify=y)
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)
        roc[i] = roc_auc_score(y_test, y_score[:, 1])
        prc[i] = average_precision_score(y_test, y_score[:, 1])
    ret = {}
    ret['roc mean'] = np.mean(roc)
    ret['roc std'] = np.std(roc)
    ret['prc mean'] = np.mean(prc)
    ret['prc std'] = np.std(prc)
    return ret

device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

pad_index = 0
unk_index = 1
eos_index = 2
sos_index = 3
mask_index = 4
data_label=sys.argv[1]
vocab = WordVocab.load_vocab('data/vocab.pkl')
df = pd.read_csv('data/'+data_label+'.csv')

df_train=df[df['group']=='train']
df_test=df[df['group']=='test']
df_valid=df[df['group']=='valid']

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
set_random_seed(seed=43)

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

def run_a_train_epoch(model, data_loader, loss_func, optimizer, args):
    model.train()

    train_metric = Meter()  # for each epoch
    for batch_id, batch_data in (enumerate(train_loader)):
        sm,y=batch_data
        optimizer.zero_grad()
        sm = torch.t(sm.to(args['device'])) # (T,B)
        outputs = model(sm)
        y=y.to(args['device']).float()
        loss = (loss_func(outputs, y).float())
        loss=loss.mean()
        loss.backward()
        optimizer.step()
        #optimizer.zero_grad()
        train_metric.update(outputs, y)
        print(loss)
        #optimizer.zero_grad()
        #optimizer.step()
        #torch.cuda.empty_cache()
    '''
    for batch_id, batch_data in enumerate(data_loader):
        Xs, Ys, masks = batch_data

        # transfer the data to device(cpu or cuda)
        Xs, Ys, masks = Xs.to(args['device']), Ys.to(args['device']), masks.to(args['device'])

        outputs = model(Xs)
        loss = (loss_func(outputs, Ys) * (masks != 0).float()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs.cpu()
        Ys.cpu()
        masks.cpu()
        loss.cpu()
#        torch.cuda.empty_cache()

        train_metric.update(outputs, Ys, masks)
    '''
    if args['reg']:
        rmse_score = np.mean(train_metric.compute_metric(args['metric']))  # in case of multi-tasks
        mae_score = np.mean(train_metric.compute_metric('mae'))  # in case of multi-tasks
        r2_score = np.mean(train_metric.compute_metric('r2'))  # in case of multi-tasks
        return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
    else:
        roc_score = np.mean(train_metric.compute_metric(args['metric']))  # in case of multi-tasks
        prc_score = np.mean(train_metric.compute_metric('prc_auc'))  # in case of multi-tasks
        print(roc_score)
        print(prc_score)
        return {'roc_auc': roc_score, 'prc_auc': prc_score}


def run_an_eval_epoch(model, data_loader, args):

    model.eval()

    eval_metric = Meter()  # for each epoch
    for batch_id, batch_data in (enumerate(data_loader)):
        sm,y=batch_data
        sm = torch.t(sm.to(args['device'])) # (T,B)
        outputs = model(sm)
        y=y.to(args['device']).float()
        '''
        loss = (loss_func(outputs, y).float()).mean()
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        '''
        eval_metric.update(outputs, y)
    '''
    model.eval()

    eval_metric = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            Xs, Ys, masks = batch_data
            # transfer the data to device(cpu or cuda)
            Xs, Ys, masks = Xs.to(args['device']), Ys.to(args['device']), masks.to(args['device'])

            outputs = model(Xs)

            outputs.cpu()
            Ys.cpu()
            masks.cpu()
#            torch.cuda.empty_cache()
            eval_metric.update(outputs, Ys, masks)
    '''
    if args['reg']:
        rmse_score = np.mean(eval_metric.compute_metric(args['metric']))  # in case of multi-tasks
        mae_score = np.mean(eval_metric.compute_metric('mae'))  # in case of multi-tasks
        r2_score = np.mean(eval_metric.compute_metric('r2'))  # in case of multi-tasks
        return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
    else:
        roc_score = np.mean(eval_metric.compute_metric(args['metric']))  # in case of multi-tasks
        prc_score = np.mean(eval_metric.compute_metric('prc_auc'))  # in case of multi-tasks
        print(roc_score)
        print(prc_score)
        return {'roc_auc': roc_score, 'prc_auc': prc_score}


def get_pos_weight(Ys):
    Ys = torch.tensor(np.nan_to_num(Ys), dtype=torch.float32)
    num_pos = torch.sum(Ys, dim=0)
    num_indices = torch.tensor(len(Ys))
    return (num_indices - num_pos) / num_pos


def standardize(col):
    return (col - np.mean(col)) / np.std(col)


def all_one_zeros(series):
    if (len(series.dropna().unique()) == 2):
        flag = False
    else:
        flag = True
    return flag


tasks_dic = {'freesolv': ['activity'], 'esol': ['activity'], 'lipop': ['activity'], 'bace': ['activity'],
             'bbbp': ['activity'], 'hiv': ['activity'],
             'clintox': ['FDA_APPROVED', 'CT_TOX'],
             'sider': ['SIDER1', 'SIDER2', 'SIDER3', 'SIDER4', 'SIDER5', 'SIDER6', 'SIDER7', 'SIDER8', 'SIDER9',
                       'SIDER10', 'SIDER11', 'SIDER12', 'SIDER13', 'SIDER14', 'SIDER15', 'SIDER16', 'SIDER17',
                       'SIDER18', 'SIDER19', 'SIDER20', 'SIDER21', 'SIDER22', 'SIDER23', 'SIDER24', 'SIDER25',
                       'SIDER26', 'SIDER27'],
             'tox21': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
                       'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
             'muv': [
                 "MUV-466", "MUV-548", "MUV-600", "MUV-644", "MUV-652", "MUV-689", "MUV-692", "MUV-712", "MUV-713",
                 "MUV-733", "MUV-737", "MUV-810", "MUV-832", "MUV-846", "MUV-852", "MUV-858", "MUV-859"
             ],
             'toxcast': ['ACEA_T47D_80hr_Negative', 'ACEA_T47D_80hr_Positive',
                         'APR_HepG2_CellCycleArrest_24h_dn',
                         'APR_HepG2_CellCycleArrest_72h_dn', 'APR_HepG2_CellLoss_24h_dn',
                         'APR_HepG2_CellLoss_72h_dn', 'APR_HepG2_MicrotubuleCSK_72h_up',
                         'APR_HepG2_MitoMass_24h_dn', 'APR_HepG2_MitoMass_72h_dn',
                         'APR_HepG2_MitoMembPot_24h_dn', 'APR_HepG2_MitoMembPot_72h_dn',
                         'APR_HepG2_MitoticArrest_24h_up', 'APR_HepG2_MitoticArrest_72h_up',
                         'APR_HepG2_OxidativeStress_24h_up',
                         'APR_HepG2_OxidativeStress_72h_up',
                         'APR_HepG2_StressKinase_72h_up', 'APR_HepG2_p53Act_24h_up',
                         'APR_HepG2_p53Act_72h_up', 'ATG_AP_1_CIS_up', 'ATG_Ahr_CIS_up',
                         'ATG_BRE_CIS_up', 'ATG_CMV_CIS_up', 'ATG_CRE_CIS_up',
                         'ATG_DR4_LXR_CIS_dn', 'ATG_DR5_CIS_up', 'ATG_EGR_CIS_up',
                         'ATG_ERE_CIS_up', 'ATG_ERa_TRANS_up', 'ATG_E_Box_CIS_dn',
                         'ATG_HIF1a_CIS_up', 'ATG_HSE_CIS_up', 'ATG_IR1_CIS_dn',
                         'ATG_ISRE_CIS_dn', 'ATG_MRE_CIS_up', 'ATG_NRF2_ARE_CIS_up',
                         'ATG_Oct_MLP_CIS_up', 'ATG_PBREM_CIS_up', 'ATG_PPARg_TRANS_up',
                         'ATG_PPRE_CIS_up', 'ATG_PXRE_CIS_dn', 'ATG_PXRE_CIS_up',
                         'ATG_PXR_TRANS_up', 'ATG_Pax6_CIS_up', 'ATG_RORE_CIS_up',
                         'ATG_RXRb_TRANS_up', 'ATG_SREBP_CIS_up', 'ATG_Sp1_CIS_up',
                         'ATG_TCF_b_cat_CIS_dn', 'ATG_VDRE_CIS_up', 'ATG_Xbp1_CIS_up',
                         'ATG_p53_CIS_dn', 'BSK_3C_Eselectin_down', 'BSK_3C_HLADR_down',
                         'BSK_3C_ICAM1_down', 'BSK_3C_IL8_down', 'BSK_3C_MCP1_down',
                         'BSK_3C_MIG_down', 'BSK_3C_Proliferation_down', 'BSK_3C_SRB_down',
                         'BSK_3C_Thrombomodulin_up', 'BSK_3C_TissueFactor_down',
                         'BSK_3C_VCAM1_down', 'BSK_3C_Vis_down', 'BSK_3C_uPAR_down',
                         'BSK_4H_Eotaxin3_down', 'BSK_4H_MCP1_down',
                         'BSK_4H_Pselectin_down', 'BSK_4H_SRB_down', 'BSK_4H_VCAM1_down',
                         'BSK_4H_VEGFRII_down', 'BSK_4H_uPAR_down', 'BSK_BE3C_HLADR_down',
                         'BSK_BE3C_IL1a_down', 'BSK_BE3C_IP10_down', 'BSK_BE3C_MIG_down',
                         'BSK_BE3C_MMP1_down', 'BSK_BE3C_MMP1_up', 'BSK_BE3C_PAI1_down',
                         'BSK_BE3C_SRB_down', 'BSK_BE3C_TGFb1_down', 'BSK_BE3C_tPA_down',
                         'BSK_BE3C_uPAR_down', 'BSK_BE3C_uPA_down', 'BSK_CASM3C_HLADR_down',
                         'BSK_CASM3C_IL6_down', 'BSK_CASM3C_IL8_down',
                         'BSK_CASM3C_LDLR_down', 'BSK_CASM3C_MCP1_down',
                         'BSK_CASM3C_MCSF_down', 'BSK_CASM3C_MIG_down',
                         'BSK_CASM3C_Proliferation_down', 'BSK_CASM3C_SAA_down',
                         'BSK_CASM3C_SRB_down', 'BSK_CASM3C_Thrombomodulin_up',
                         'BSK_CASM3C_TissueFactor_down', 'BSK_CASM3C_VCAM1_down',
                         'BSK_CASM3C_uPAR_down', 'BSK_KF3CT_ICAM1_down',
                         'BSK_KF3CT_IL1a_down', 'BSK_KF3CT_IP10_down',
                         'BSK_KF3CT_MCP1_down', 'BSK_KF3CT_MMP9_down', 'BSK_KF3CT_SRB_down',
                         'BSK_KF3CT_TGFb1_down', 'BSK_KF3CT_TIMP2_down',
                         'BSK_KF3CT_uPA_down', 'BSK_LPS_CD40_down',
                         'BSK_LPS_Eselectin_down', 'BSK_LPS_IL1a_down', 'BSK_LPS_IL8_down',
                         'BSK_LPS_MCP1_down', 'BSK_LPS_MCSF_down', 'BSK_LPS_PGE2_down',
                         'BSK_LPS_SRB_down', 'BSK_LPS_TNFa_down',
                         'BSK_LPS_TissueFactor_down', 'BSK_LPS_VCAM1_down',
                         'BSK_SAg_CD38_down', 'BSK_SAg_CD40_down', 'BSK_SAg_CD69_down',
                         'BSK_SAg_Eselectin_down', 'BSK_SAg_IL8_down', 'BSK_SAg_MCP1_down',
                         'BSK_SAg_MIG_down', 'BSK_SAg_PBMCCytotoxicity_down',
                         'BSK_SAg_Proliferation_down', 'BSK_SAg_SRB_down',
                         'BSK_hDFCGF_CollagenIII_down', 'BSK_hDFCGF_IL8_down',
                         'BSK_hDFCGF_IP10_down', 'BSK_hDFCGF_MCSF_down',
                         'BSK_hDFCGF_MIG_down', 'BSK_hDFCGF_MMP1_down',
                         'BSK_hDFCGF_PAI1_down', 'BSK_hDFCGF_Proliferation_down',
                         'BSK_hDFCGF_SRB_down', 'BSK_hDFCGF_TIMP1_down',
                         'BSK_hDFCGF_VCAM1_down', 'CEETOX_H295R_11DCORT_dn',
                         'CEETOX_H295R_ANDR_dn', 'CEETOX_H295R_CORTISOL_dn',
                         'CEETOX_H295R_ESTRONE_dn', 'CEETOX_H295R_ESTRONE_up',
                         'NHEERL_ZF_144hpf_TERATOSCORE_up', 'NVS_NR_bER', 'NVS_NR_hER',
                         'NVS_NR_hPPARg', 'NVS_NR_hPXR', 'NVS_NR_mERa', 'OT_AR_ARSRC1_0960',
                         'OT_ER_ERaERb_0480', 'OT_ER_ERaERb_1440', 'OT_ER_ERbERb_0480',
                         'OT_ER_ERbERb_1440', 'OT_ERa_EREGFP_0120', 'OT_FXR_FXRSRC1_0480',
                         'OT_NURR1_NURR1RXRa_0480', 'TOX21_ARE_BLA_agonist_ratio',
                         'TOX21_AR_BLA_Antagonist_ratio', 'TOX21_AR_LUC_MDAKB2_Antagonist',
                         'TOX21_AR_LUC_MDAKB2_Antagonist2', 'TOX21_AhR_LUC_Agonist',
                         'TOX21_Aromatase_Inhibition', 'TOX21_ERa_BLA_Antagonist_ratio',
                         'TOX21_ERa_LUC_BG1_Agonist', 'TOX21_FXR_BLA_antagonist_ratio',
                         'TOX21_MMP_ratio_down', 'TOX21_TR_LUC_GH3_Antagonist',
                         'TOX21_p53_BLA_p1_ratio', 'TOX21_p53_BLA_p2_ch2',
                         'TOX21_p53_BLA_p2_ratio', 'TOX21_p53_BLA_p2_viability',
                         'TOX21_p53_BLA_p3_ratio', 'TOX21_p53_BLA_p4_ratio',
                         'TOX21_p53_BLA_p5_ratio', 'Tanguay_ZF_120hpf_AXIS_up',
                         'Tanguay_ZF_120hpf_ActivityScore', 'Tanguay_ZF_120hpf_JAW_up',
                         'Tanguay_ZF_120hpf_MORT_up', 'Tanguay_ZF_120hpf_PE_up',
                         'Tanguay_ZF_120hpf_SNOU_up', 'Tanguay_ZF_120hpf_YSE_up']}

hyper_paras_space = {'l2': hp.uniform('l2', 0, 0.001),
                     'dropout': hp.uniform('dropout', 0, 0.5),
                     #'hidden_unit1': hp.choice('hidden_unit1', [64, 128, 256, 512]),
                     #'hidden_unit2': hp.choice('hidden_unit2', [64, 128, 256, 512]),
                     #'hidden_unit3': hp.choice('hidden_unit3', [64, 128, 256, 512])
                     }

dataset_label =data_label
if data_label=='esol' or data_label=='freesolv' or data_label=='lipop':
    task_type='reg'
else :
    task_type = 'cla' 
reg = True if task_type == 'reg' else False
epochs = 300  # training epoch
batch_size = 128
patience = 50
opt_iters = 5
repetitions = 10

args = {'device': device, 'metric': 'rmse' if reg else 'roc_auc', 'epochs': epochs,
        'patience': patience, 'task': data_label, 'reg': reg}

tasks = tasks_dic[data_label]

# training set
train_dataset=Seq2seqDataset(df_train['cano_smiles'].values,df_train[tasks].values.reshape(-1, len(tasks)),vocab)
test_dataset=Seq2seqDataset(df_test['cano_smiles'].values,df_test[tasks].values.reshape(-1, len(tasks)),vocab)
valid_dataset=Seq2seqDataset(df_valid['cano_smiles'].values,df_valid[tasks].values.reshape(-1, len(tasks)),vocab)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=16) 
validation_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=16)   

if not reg:
    pos_weights = get_pos_weight(df[tasks].values)

def hyper_opt(hyper_paras):

    my_model =e_TrfmSeq2seq(50,256,50, 4,0,reg,len(tasks))
    optimizer = torch.optim.Adadelta(my_model.parameters(), weight_decay=hyper_paras['l2'])
    file_name = './model_save/%s_%.4f_%.4f_early_stop.pth' % (args['task'], hyper_paras['dropout'],
                                                          hyper_paras['l2'])
    if reg:
        loss_func = MSELoss(reduction='none')
        stopper = EarlyStopping(mode='lower', patience=patience, filename=file_name)
    else:
        print(pos_weights)
        loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights.to(args['device']))
        stopper = EarlyStopping(mode='higher', patience=patience, filename=file_name)
    my_model.to(device)
    for i in range(epochs):
        run_a_train_epoch(my_model, train_loader, loss_func, optimizer, args)

        # early stopping
        val_scores = run_an_eval_epoch(my_model, validation_loader, args)

        early_stop = stopper.step(val_scores[args['metric']], my_model)

        if early_stop:
            break
    stopper.load_checkpoint(my_model)
    val_scores = run_an_eval_epoch(my_model, validation_loader, args)
    feedback = val_scores[args['metric']] if reg else (1 - val_scores[args['metric']])
    my_model.cpu()
    gc.collect()
    return feedback


# start hyper-parameters optimization
trials = Trials()  

print('******hyper-parameter optimization is starting now******')
opt_res = fmin(hyper_opt, hyper_paras_space, algo=tpe.suggest, max_evals=opt_iters, trials=trials)

# hyper-parameters optimization is over
print('******hyper-parameter optimization is over******')
print('the best hyper-parameters settings for ' + args['task'] + ' are:  ', opt_res)

# construct the model based on the optimal hyper-parameters
best_model = e_TrfmSeq2seq(50, 256,50, 4,opt_res['dropout'],reg,len(tasks))

best_file_name = './model_save/%s_%.4f_%.4f_early_stop.pth' % (args['task'], opt_res['dropout'],
                                                           opt_res['l2'])

best_model.load_state_dict(torch.load(best_file_name, map_location=device)['model_state_dict'])
best_model.to(device)
tr_scores = run_an_eval_epoch(best_model, train_loader, args)
val_scores = run_an_eval_epoch(best_model, validation_loader, args)
te_scores = run_an_eval_epoch(best_model, test_loader, args)
print('training set:', tr_scores)
print('validation set:', val_scores)
print('test set:', te_scores)