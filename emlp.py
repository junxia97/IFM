import argparse
import torch
import numpy as np
import pandas as pd
from emlp_utils import Meter, MyDataset, EarlyStopping, MyDNN, IFM_DNN, LE_DNN, LIFM_DNN, SM_DNN, GM_DNN, collate_fn, set_random_seed
from hyperopt import fmin, tpe, hp, rand, STATUS_OK, Trials, partial
import sys
import copy
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, MSELoss
import gc
import time
start_time = time.time()
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
set_random_seed(seed=43)

def run_a_train_epoch(model, data_loader, loss_func, optimizer, args):
    model.train()
    train_metric = Meter()  # for each epoch
    for batch_id, batch_data in enumerate(data_loader):
        Xs, Ys, masks = batch_data
        # transfer the data to device(cpu or cuda)
        Xs, Ys, masks = Xs.to(args.device), Ys.to(args.device), masks.to(args.device)
        outputs = model(Xs)
        loss = (loss_func(outputs, Ys) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        outputs.cpu()
        Ys.cpu()
        masks.cpu()
        loss.cpu()
        train_metric.update(outputs, Ys, masks)
    if args.reg:
        rmse_score = np.mean(train_metric.compute_metric(args.metric))  # in case of multi-tasks
        mae_score = np.mean(train_metric.compute_metric('mae'))  # in case of multi-tasks
        r2_score = np.mean(train_metric.compute_metric('r2'))  # in case of multi-tasks
        return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
    else:
        roc_score = np.mean(train_metric.compute_metric(args.metric))  # in case of multi-tasks
        prc_score = np.mean(train_metric.compute_metric('prc_auc'))  # in case of multi-tasks
        return {'roc_auc': roc_score, 'prc_auc': prc_score}


def run_an_eval_epoch(model, data_loader, args):
    model.eval()
    eval_metric = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            Xs, Ys, masks = batch_data
            # transfer the data to device(cpu or cuda)
            Xs, Ys, masks = Xs.to(args.device), Ys.to(args.device), masks.to(args.device)
            outputs = model(Xs)
            outputs.cpu()
            Ys.cpu()
            masks.cpu()
            eval_metric.update(outputs, Ys, masks)
    if args.reg:
        rmse_score = np.mean(eval_metric.compute_metric(args.metric))  # in case of multi-tasks
        mae_score = np.mean(eval_metric.compute_metric('mae'))  # in case of multi-tasks
        r2_score = np.mean(eval_metric.compute_metric('r2'))  # in case of multi-tasks
        return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
    else:
        roc_score = np.mean(eval_metric.compute_metric(args.metric))  # in case of multi-tasks
        prc_score = np.mean(eval_metric.compute_metric('prc_auc'))  # in case of multi-tasks
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

def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation')
    parser.add_argument('--device', type=int, default=5, help='which gpu to use if any (default: 0)')
    parser.add_argument('--data_label', type=str, default = 'sider', help='dataset.')
    parser.add_argument('--embed', type=str, default = 'GM', help='Embedding method: None, LE, LIFM, GM, SM, IFM.')
    parser.add_argument('--epochs', type=int, default=300, help='running epochs')
    parser.add_argument('--runseed', type=int, default=43, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--batch_size', type=int, default=128, help='batchsize')
    parser.add_argument('--patience', type=int, default=50, help='patience')
    parser.add_argument('--opt_iters', type=int, default=50, help='optimization_iters')
    parser.add_argument('--repetitions', type=int, default=50, help='splitting repetitions')
    args = parser.parse_args()
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)   
    args.task = args.data_label
    
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
    if args.data_label == 'esol' or args.data_label == 'freesolv' or args.data_label == 'lipop':
        task_type = 'reg'
        args.reg = True
        args.metric = 'rmse'
    else:
        task_type = 'cla'
        args.reg = False
        args.metric = 'roc_auc'

    if args.embed == 'None':
        hyper_paras_space = {'l2': hp.uniform('l2', 0, 0.01),
                        'dropout': hp.uniform('dropout', 0, 0.5),
                        'hidden_unit1': hp.choice('hidden_unit1', [64, 128, 256, 512]),
                        'hidden_unit2': hp.choice('hidden_unit2', [64, 128, 256, 512]),
                        'hidden_unit3': hp.choice('hidden_unit3', [64, 128, 256, 512])}
    elif args.embed == 'LE':
        hyper_paras_space = {'l2': hp.uniform('l2', 0, 0.01),
                        'dropout': hp.uniform('dropout', 0, 0.5),
                        'd_out': hp.choice('d_out', [3, 5, 7, 9]),
                        'hidden_unit1': hp.choice('hidden_unit1', [64, 128, 256, 512]),
                        'hidden_unit2': hp.choice('hidden_unit2', [64, 128, 256, 512]),
                        'hidden_unit3': hp.choice('hidden_unit3', [64, 128, 256, 512])}
    elif args.embed == 'LIFM': #L-IFM
        hyper_paras_space = {'l2': hp.uniform('l2', 0, 0.01),
                        'dropout': hp.uniform('dropout', 0, 0.5),
                        'd_out': hp.randint('d_out', 127),
                        'sigma': hp.loguniform('sigma', np.log(0.01), np.log(100)),
                        'hidden_unit1': hp.choice('hidden_unit1', [64, 128, 256, 512]),
                        'hidden_unit2': hp.choice('hidden_unit2', [64, 128, 256, 512]),
                        'hidden_unit3': hp.choice('hidden_unit3', [64, 128, 256, 512])}
    elif args.embed == 'SM':
        hyper_paras_space = {'l2': hp.uniform('l2', 0, 0.01),
                        'dropout': hp.uniform('dropout', 0, 0.5),
                        'd_out': hp.randint('d_out', 127),
                        'sigma': hp.loguniform('sigma', np.log(0.01), np.log(100)),
                        'hidden_unit1': hp.choice('hidden_unit1', [64, 128, 256, 512]),
                        'hidden_unit2': hp.choice('hidden_unit2', [64, 128, 256, 512]),
                        'hidden_unit3': hp.choice('hidden_unit3', [64, 128, 256, 512])}
    elif args.embed == 'GM':
        hyper_paras_space = {'l2': hp.uniform('l2', 0, 0.01),
                        'dropout': hp.uniform('dropout', 0, 0.5),
                        # 'd_out': hp.choice('d_out', [32, 64, 128, 256, 512]),
                        'd_out': hp.randint('d_out', 127),
                        'sigma': hp.loguniform('sigma', np.log(0.01), np.log(100)),
                        'hidden_unit1': hp.choice('hidden_unit1', [64, 128, 256, 512]),
                        'hidden_unit2': hp.choice('hidden_unit2', [64, 128, 256, 512]),
                        'hidden_unit3': hp.choice('hidden_unit3', [64, 128, 256, 512])}
    elif args.embed == 'IFM':
        hyper_paras_space = {'l2': hp.uniform('l2', 0, 0.01),
                        'dropout': hp.uniform('dropout', 0, 0.5),
                        # 'd_out': hp.choice('d_out', [32, 64, 128, 256, 512]),
                        'd_out': hp.randint('d_out', 6),
                        'sigma': hp.loguniform('sigma', np.log(0.01), np.log(100)),
                        'hidden_unit1': hp.choice('hidden_unit1', [64, 128, 256, 512]),
                        'hidden_unit2': hp.choice('hidden_unit2', [64, 128, 256, 512]),
                        'hidden_unit3': hp.choice('hidden_unit3', [64, 128, 256, 512])}
    elif args.embed == 'IFM':
        hyper_paras_space = {'l2': hp.uniform('l2', 0, 0.01),
                'dropout': hp.uniform('dropout', 0, 0.5),
                'd_out': hp.randint('d_out', 8),
                'sigma': hp.randint('sigma', 9),
                'hidden_unit1': hp.choice('hidden_unit1', [64, 128, 256, 512]),
                'hidden_unit2': hp.choice('hidden_unit2', [64, 128, 256, 512]),
                'hidden_unit3': hp.choice('hidden_unit3', [64, 128, 256, 512])}
    else:
        raise ValueError("Invalid Embedding Name")

    file_name = './dataset/'+args.data_label+'_moe_pubsubfp.csv'
    # preprocess data
    dataset_all = pd.read_csv(file_name)
    if args.data_label == 'freesolv':
        dataset_all.drop(columns=['vsa_pol', 'h_emd', 'a_donacc'], inplace=True)
    elif args.data_label == 'esol':
        dataset_all.drop(columns=['logS', 'h_logS', 'SlogP'], inplace=True)
    else:
        dataset_all.drop(columns=['SlogP', 'h_logD', 'logS'], inplace=True)
    tasks = tasks_dic[args.data_label]
    cols = copy.deepcopy(tasks)
    cols.extend(dataset_all.columns[len(tasks) + 1:])
    dataset = dataset_all[cols]
    x_cols = dataset_all.columns[len(tasks) + 1:]
    # remove the features with na
    if args.data_label != 'hiv':
        rm_cols1 = dataset[x_cols].isnull().any()[dataset[x_cols].isnull().any() == True].index
        dataset.drop(columns=rm_cols1, inplace=True)
    else:
        rm_indx1 = dataset[x_cols].isnull().T.any()[dataset[x_cols].isnull().T.any() == True].index
        dataset.drop(index=rm_indx1, inplace=True)
    x_cols = dataset.columns.drop(tasks)

    # Removing features with low variance
    # threshold = 0.05
    data_fea_var = dataset[x_cols].var()
    del_fea1 = list(data_fea_var[data_fea_var <= 0.05].index)
    dataset.drop(columns=del_fea1, inplace=True)
    x_cols = dataset.columns.drop(tasks)

    # pair correlations
    # threshold = 0.95
    data_fea_corr = dataset[x_cols].corr()
    del_fea2_col = []
    del_fea2_ind = []
    length = data_fea_corr.shape[1]
    for i in range(length):
        for j in range(i + 1, length):
            if abs(data_fea_corr.iloc[i, j]) >= 0.95:
                del_fea2_col.append(data_fea_corr.columns[i])
                del_fea2_ind.append(data_fea_corr.index[j])
    dataset.drop(columns=del_fea2_ind, inplace=True)
    # standardize the features
    cols_ = dataset.columns[len(tasks) + 1:]
    print(cols_)
    print('the retained features for %s is %d' % (args.task, len(cols_)))
    dataset[cols_] = dataset[cols_].apply(standardize, axis=0)

    dataseta=pd.read_csv('./dataset/dataset_used_for_modeling/'+args.data_label+'.csv')
    data_tr = dataset[dataseta.group == 'train']
    data_va = dataset[dataseta.group == 'valid']
    data_te = dataset[dataseta.group == 'test']
    # training set
    data_tr_y = data_tr[tasks].values.reshape(-1, len(tasks))
    data_tr_x = data_tr.iloc[:, len(tasks):].values #249
    # data_tr_x = data_tr.iloc[:, len(tasks):].values
    # test set
    data_te_y = data_te[tasks].values.reshape(-1, len(tasks))
    data_te_x = data_te.iloc[:, len(tasks):].values
    # data_te_x = data_te.iloc[:, len(tasks):].values

    # validation set
    data_va_y = data_va[tasks].values.reshape(-1, len(tasks))
    data_va_x = data_va.iloc[:, len(tasks):].values
    # data_va_x = data_va.iloc[:, len(tasks):].values

    # dataloader
    train_dataset = MyDataset(data_tr_x, data_tr_y)
    validation_dataset = MyDataset(data_va_x, data_va_y)
    test_dataset = MyDataset(data_te_x, data_te_y)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    inputs = data_tr_x.shape[1]
    if not args.reg:
        pos_weights = get_pos_weight(dataset[tasks].values)
    print("inputs:", inputs)

    def hyper_opt(hyper_paras):
        hidden_units = [hyper_paras['hidden_unit1'], hyper_paras['hidden_unit2'], hyper_paras['hidden_unit3']]
        if args.embed == 'None':
            my_model = MyDNN(inputs=inputs, hidden_units=hidden_units, dp_ratio=hyper_paras['dropout'], outputs=len(tasks), reg=args.reg)
        elif args.embed == 'LE':
            my_model = LE_DNN(inputs=inputs, hidden_units=hidden_units, d_out=hyper_paras['d_out'] + 1, dp_ratio=hyper_paras['dropout'], outputs=len(tasks), reg=args.reg)
        elif args.embed == 'LIFM':
            my_model = LIFM_DNN(inputs=inputs, hidden_units=hidden_units, d_out=hyper_paras['d_out'] + 1, sigma=hyper_paras['sigma'], dp_ratio=hyper_paras['dropout'], outputs=len(tasks), reg=args.reg)
        elif args.embed == 'SM':
            my_model = SM_DNN(inputs=inputs, hidden_units=hidden_units, d_out=hyper_paras['d_out'] + 1, sigma=hyper_paras['sigma'], dp_ratio=hyper_paras['dropout'], first_omega_0=1, hidden_omega_0=1, outputs=len(tasks), reg=args.reg)
        elif args.embed == 'GM':
            my_model = GM_DNN(inputs=inputs, hidden_units=hidden_units, d_out=hyper_paras['d_out'] + 1, sigma=hyper_paras['sigma'] + 1, dp_ratio=hyper_paras['dropout'], outputs=len(tasks), reg=args.reg)
        elif args.embed == 'IFM':
            my_model = IFM_DNN(inputs=inputs, hidden_units=hidden_units, d_out=hyper_paras['d_out'] + 1, sigma=hyper_paras['sigma'] + 1, dp_ratio=hyper_paras['dropout'], outputs=len(tasks), reg=args.reg)
        else:
            raise ValueError("Invalid Embedding Name")
        optimizer = torch.optim.Adadelta(my_model.parameters(), weight_decay=hyper_paras['l2'])
        file_name = './save_model/%s_%.4f_%d_%d_%d_%.4f_early_stop.pth' % (args.task, hyper_paras['dropout'],
                                                            hyper_paras['hidden_unit1'],
                                                            hyper_paras['hidden_unit2'],
                                                            hyper_paras['hidden_unit3'],
                                                            hyper_paras['l2'])
        if args.reg:
            loss_func = MSELoss(reduction='none')
            stopper = EarlyStopping(mode='lower', patience=args.patience, filename=file_name)
        else:
            loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights.to(args.device))
            stopper = EarlyStopping(mode='higher', patience=args.patience, filename=file_name)
        my_model.to(args.device)
        for i in range(args.epochs):
            # training
            run_a_train_epoch(my_model, train_loader, loss_func, optimizer, args)
            # early stopping
            val_scores = run_an_eval_epoch(my_model, validation_loader, args)
            early_stop = stopper.step(val_scores[args.metric], my_model)
            if early_stop:
                break
        stopper.load_checkpoint(my_model)
        val_scores = run_an_eval_epoch(my_model, validation_loader, args)
        feedback = val_scores[args.metric] if args.reg else (1 - val_scores[args.metric])
        my_model.cpu()
        gc.collect()
        return feedback

    # start hyper-parameters optimization
    trials = Trials()
    print('******hyper-parameter optimization is starting now******')
    opt_res = fmin(hyper_opt, hyper_paras_space, algo=tpe.suggest, max_evals=args.opt_iters, trials=trials)

    # hyper-parameters optimization is over
    print('******hyper-parameter optimization is over******')
    print('the best hyper-parameters settings for ' + args.task + ' are:  ', opt_res)

    # construct the model based on the optimal hyper-parameters
    hidden_unit1_ls = [64, 128, 256, 512]
    hidden_unit2_ls = [64, 128, 256, 512]
    hidden_unit3_ls = [64, 128, 256, 512]
    opt_hidden_units = [hidden_unit1_ls[opt_res['hidden_unit1']], hidden_unit2_ls[opt_res['hidden_unit2']],
                        hidden_unit3_ls[opt_res['hidden_unit3']]]

    if args.embed == 'None':
        best_model = MyDNN(inputs=inputs, hidden_units=opt_hidden_units, dp_ratio=opt_res['dropout'], outputs=len(tasks), reg=args.reg)
    elif args.embed == 'LE':
        best_model = LE_DNN(inputs=inputs, hidden_units=opt_hidden_units, d_out=opt_res['d_out']+ 1, dp_ratio=opt_res['dropout'], outputs=len(tasks), reg=args.reg)
    elif args.embed == 'LIFM':
        best_model = LIFM_DNN(inputs=inputs, hidden_units=opt_hidden_units, d_out=opt_res['d_out']+ 1, sigma=opt_res['sigma'], dp_ratio=opt_res['dropout'], outputs=len(tasks), reg=args.reg)
    elif args.embed == 'SM':
        best_model = SM_DNN(inputs=inputs, hidden_units=opt_hidden_units, d_out=opt_res['d_out'] + 1, sigma=opt_res['sigma'], dp_ratio=opt_res['dropout'], first_omega_0=1, hidden_omega_0=1, outputs=len(tasks), reg=args.reg)
    elif args.embed == 'GM':
        best_model = GM_DNN(inputs=inputs, hidden_units=opt_hidden_units, d_out=opt_res['d_out']+ 1, sigma=opt_res['sigma']+1, dp_ratio=opt_res['dropout'], outputs=len(tasks), reg=args.reg)
    elif args.embed == 'IFM':
        best_model = IFM_DNN(inputs=inputs, hidden_units=opt_hidden_units, d_out=opt_res['d_out']+ 1, sigma=opt_res['sigma']+1, dp_ratio=opt_res['dropout'], outputs=len(tasks), reg=args.reg)
    else:
        raise ValueError("Invalid Embedding Name")

    # best_model = GM_DNN(inputs=inputs, hidden_units=opt_hidden_units, outputs=len(tasks), embed=embed, d_out=d_out, sigma=sigma,
    #                 dp_ratio=opt_res['dropout'], reg=args.reg)
    best_file_name = './save_model/%s_%.4f_%d_%d_%d_%.4f_early_stop.pth' % (args.task, opt_res['dropout'],
                                                            hidden_unit1_ls[opt_res['hidden_unit1']],
                                                            hidden_unit1_ls[opt_res['hidden_unit2']],
                                                            hidden_unit1_ls[opt_res['hidden_unit3']],
                                                            opt_res['l2'])

    best_model.load_state_dict(torch.load(best_file_name, map_location=args.device)['model_state_dict'])
    best_model.to(args.device)
    tr_scores = run_an_eval_epoch(best_model, train_loader, args)
    val_scores = run_an_eval_epoch(best_model, validation_loader, args)
    te_scores = run_an_eval_epoch(best_model, test_loader, args)

    print('training set:', tr_scores)
    print('validation set:', val_scores)
    print('test set:', te_scores)

    # 50 repetitions based on the best model
    tr_res = []
    val_res = []
    te_res = []
    if args.data_label != 'muv' and args.data_label != 'toxcast':
        pass
        # dataset.drop(columns=['group'], inplace=True)
    else:
        file=pd.read_csv('./dataset/'+args.data_label+'_moe_pubsubfp.csv')

        # file = args.data_label + '_norepeat_moe_pubsubfp.csv'
        # repreprocess data
        dataset = pd.read_csv(file)
        dataset.drop(columns=['cano_smiles'], inplace=True)

        # remove the features with na
        x_cols = dataset.columns.drop(tasks)
        rm_cols1 = dataset[x_cols].isnull().any()[dataset[x_cols].isnull().any() == True].index
        dataset.drop(columns=rm_cols1, inplace=True)

        # Removing features with low variance
        # threshold = 0.05
        x_cols = dataset.columns.drop(tasks)
        data_fea_var = dataset[x_cols].var()
        del_fea1 = list(data_fea_var[data_fea_var <= 0.05].index)
        dataset.drop(columns=del_fea1, inplace=True)

        # pair correlations
        # threshold = 0.95
        x_cols = dataset.columns.drop(tasks)
        data_fea_corr = dataset[x_cols].corr()
        del_fea2_col = []
        del_fea2_ind = []
        length = data_fea_corr.shape[1]
        for i in range(length):
            for j in range(i + 1, length):
                if abs(data_fea_corr.iloc[i, j]) >= 0.95:
                    del_fea2_col.append(data_fea_corr.columns[i])
                    del_fea2_ind.append(data_fea_corr.index[j])
        dataset.drop(columns=del_fea2_ind, inplace=True)
        # standardize the features
        x_cols = dataset.columns.drop(tasks)
        print('the retained features for noreaptead %s is %d' % (args.task, len(x_cols)))
        dataset[x_cols] = dataset[x_cols].apply(standardize, axis=0)

    for split in range(1, args.repetitions + 1):
        # splitting the data set for classification
        if not args.reg:
            seed = split
            while True:
                training_data, data_te = train_test_split(dataset, test_size=0.1, random_state=seed)
                # the training set was further splited into the training set and validation set
                data_tr, data_va = train_test_split(training_data, test_size=0.1, random_state=seed)
                if np.any(data_tr[tasks].apply(all_one_zeros)) or \
                        np.any(data_va[tasks].apply(all_one_zeros)) or \
                        np.any(data_te[tasks].apply(all_one_zeros)):
                    print('\ninvalid random seed {} due to one class presented in the splitted {} sets...'.format(seed, args.data_label))
                    print('Changing to another random seed...\n')
                    seed = np.random.randint(50, 999999)
                else:
                    print('random seed used in repetition {} is {}'.format(split, seed))
                    break
        else:
            training_data, data_te = train_test_split(dataset, test_size=0.1, random_state=split)
            # the training set was further splited into the training set and validation set
            data_tr, data_va = train_test_split(training_data, test_size=0.1, random_state=split)
        # prepare data for training
        # training set
        data_tr_y = data_tr[tasks].values.reshape(-1, len(tasks))
        data_tr_x = data_tr.iloc[:, len(tasks):].values

        # test set
        data_te_y = data_te[tasks].values.reshape(-1, len(tasks))
        data_te_x = data_te.iloc[:, len(tasks):].values

        # validation set
        data_va_y = data_va[tasks].values.reshape(-1, len(tasks))
        data_va_x = data_va.iloc[:, len(tasks):].values

        # dataloader
        train_dataset = MyDataset(data_tr_x, data_tr_y)
        validation_dataset = MyDataset(data_va_x, data_va_y)
        test_dataset = MyDataset(data_te_x, data_te_y)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

        # best_model = GM_DNN(inputs=inputs, hidden_units=opt_hidden_units, outputs=len(tasks), embed=embed, d_out=d_out, sigma=sigma,
        #                 dp_ratio=opt_res['dropout'], reg=args.reg)

        if args.embed == 'None':
            best_model = MyDNN(inputs=inputs, hidden_units=opt_hidden_units, dp_ratio=opt_res['dropout'], outputs=len(tasks), reg=args.reg)
        elif args.embed == 'LE':
            best_model = LE_DNN(inputs=inputs, hidden_units=opt_hidden_units, d_out=opt_res['d_out']+ 1, dp_ratio=opt_res['dropout'], outputs=len(tasks), reg=args.reg)
        elif args.embed == 'LIFM':
            best_model = LIFM_DNN(inputs=inputs, hidden_units=opt_hidden_units, d_out=opt_res['d_out']+ 1, sigma=opt_res['sigma'], dp_ratio=opt_res['dropout'], outputs=len(tasks), reg=args.reg)
        elif args.embed == 'SM':
            best_model = SM_DNN(inputs=inputs, hidden_units=opt_hidden_units, d_out=opt_res['d_out'] + 1, sigma=opt_res['sigma'], dp_ratio=opt_res['dropout'], first_omega_0=1, hidden_omega_0=1, outputs=len(tasks), reg=args.reg)
        elif args.embed == 'GM':
            best_model = GM_DNN(inputs=inputs, hidden_units=opt_hidden_units, d_out=opt_res['d_out']+ 1, sigma=opt_res['sigma']+1, dp_ratio=opt_res['dropout'], outputs=len(tasks), reg=args.reg)
        elif args.embed == 'IFM':
            best_model = IFM_DNN(inputs=inputs, hidden_units=opt_hidden_units, d_out=opt_res['d_out']+ 1, sigma=opt_res['sigma']+1, dp_ratio=opt_res['dropout'], outputs=len(tasks), reg=args.reg)
        else:
            raise ValueError("Invalid Embedding Name")

        best_optimizer = torch.optim.Adadelta(best_model.parameters(), weight_decay=opt_res['l2'])
        file_name = './save_model/%s_%.4f_%d_%d_%d_%.4f_early_stop_%d.pth' % (args.task, opt_res['dropout'],
                                                                hidden_unit1_ls[opt_res['hidden_unit1']],
                                                                hidden_unit1_ls[opt_res['hidden_unit2']],
                                                                hidden_unit1_ls[opt_res['hidden_unit3']],
                                                                opt_res['l2'], split)
        if args.reg:
            loss_func = MSELoss(reduction='none')
            stopper = EarlyStopping(mode='lower', patience=args.patience, filename=file_name)
        else:
            loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weights.to(args.device))
            stopper = EarlyStopping(mode='higher', patience=args.patience, filename=file_name)
        best_model.to(args.device)

        for j in range(args.epochs):
            # training
            st = time.time()
            run_a_train_epoch(best_model, train_loader, loss_func, best_optimizer, args)
            end = time.time()
            # early stopping
            train_scores = run_an_eval_epoch(best_model, train_loader, args)
            val_scores = run_an_eval_epoch(best_model, validation_loader, args)
            early_stop = stopper.step(val_scores[args.metric], best_model)
            if early_stop:
                break
            print(
                'task:{} repetition {:d}/{:d} epoch {:d}/{:d}, training {} {:.3f}, validation {} {:.3f}, time:{:.3f}S'.format(
                    args.task, split, args.repetitions, j + 1, args.epochs, args.metric, train_scores[args.metric],
                    args.metric, val_scores[args.metric], end - st))
        stopper.load_checkpoint(best_model)
        tr_scores = run_an_eval_epoch(best_model, train_loader, args)
        val_scores = run_an_eval_epoch(best_model, validation_loader, args)
        te_scores = run_an_eval_epoch(best_model, test_loader, args)
        tr_res.append(tr_scores)
        val_res.append(val_scores)
        te_res.append(te_scores)
    if args.reg:
        cols = ['rmse', 'mae', 'r2']
    else:
        cols = ['auc_roc', 'auc_prc']
    tr = [list(item.values()) for item in tr_res]
    val = [list(item.values()) for item in val_res]
    te = [list(item.values()) for item in te_res]
    tr_pd = pd.DataFrame(tr, columns=cols); tr_pd['split'] = range(1, args.repetitions + 1); tr_pd['set'] = 'train'
    val_pd = pd.DataFrame(val, columns=cols); val_pd['split'] = range(1, args.repetitions + 1); val_pd['set'] = 'validation'
    te_pd = pd.DataFrame(te, columns=cols); te_pd['split'] = range(1, args.repetitions + 1); te_pd['set'] = 'test'
    sta_pd = pd.concat([tr_pd, val_pd, te_pd], ignore_index=True)
    # sta_pd.to_csv('./stat_res/'+ data_label + '_dnn_statistical_results_split50.csv', index=False)

    print('training mean:', np.mean(tr, axis=0), 'training std:', np.std(tr, axis=0))
    print('validation mean:', np.mean(val, axis=0), 'validation std:', np.std(val, axis=0))
    print('testing mean:', np.mean(te, axis=0), 'test std:', np.std(te, axis=0))
    end_time = time.time()
    print('total elapsed time is', end_time-start_time, 'S')

if __name__ == "__main__":
    main()