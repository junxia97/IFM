import time
import gc
import torch
from dgl.data.utils import Subset
import pandas as pd
from dgl.data.chem import csv_dataset, smiles_to_bigraph, MoleculeCSVDataset
from gnn_utils import AttentiveFPBondFeaturizer, AttentiveFPAtomFeaturizer, collate_molgraphs, \
    EarlyStopping, set_random_seed, Meter
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
import torch
from dgl.model_zoo.chem import MPNNModel, GCNClassifier, GATClassifier, AttentiveFP
import numpy as np
from sklearn.model_selection import train_test_split
from dgl import backend as B
from hyperopt import fmin, tpe, hp, Trials
import sys
import torch.nn as nn
import torch.nn.functional as F

start_time = time.time()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
set_random_seed(seed=43)

tasks_dic = {'freesolv': ['activity'], 'esol': ['activity'], 'lipop': ['activity'], 'bace': ['activity'],
             'bbbp': ['activity'], 'hiv': ['activity'],
             'clintox': ['FDA_APPROVED', 'CT_TOX'],
             'sider': ['SIDER1', 'SIDER2', 'SIDER3', 'SIDER4', 'SIDER5', 'SIDER6', 'SIDER7', 'SIDER8', 'SIDER9',
                       'SIDER10', 'SIDER11', 'SIDER12', 'SIDER13', 'SIDER14', 'SIDER15', 'SIDER16', 'SIDER17',
                       'SIDER18', 'SIDER19', 'SIDER20', 'SIDER21', 'SIDER22', 'SIDER23', 'SIDER24', 'SIDER25',
                       'SIDER26', 'SIDER27'],
             'tox21': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE',
                       'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
            'pcba':['PCBA-1030', 'PCBA-1379', 'PCBA-1452', 'PCBA-1454', 'PCBA-1457', 'PCBA-1458', 
                  'PCBA-1460', 'PCBA-1461', 'PCBA-1468', 'PCBA-1469', 'PCBA-1471', 'PCBA-1479', 'PCBA-1631', 
                     'PCBA-1634', 'PCBA-1688', 'PCBA-1721', 'PCBA-2100', 'PCBA-2101', 'PCBA-2147', 'PCBA-2242', 
                     'PCBA-2326', 'PCBA-2451', 'PCBA-2517', 'PCBA-2528', 'PCBA-2546', 'PCBA-2549', 'PCBA-2551', 
                     'PCBA-2662', 'PCBA-2675', 'PCBA-2676', 'PCBA-411', 'PCBA-463254', 'PCBA-485281', 
                     'PCBA-485290', 'PCBA-485294', 'PCBA-485297', 'PCBA-485313', 'PCBA-485314', 'PCBA-485341', 
                    'PCBA-485349', 'PCBA-485353', 'PCBA-485360', 'PCBA-485364', 'PCBA-485367', 'PCBA-492947', 
                     'PCBA-493208', 'PCBA-504327', 'PCBA-504332', 'PCBA-504333', 'PCBA-504339', 'PCBA-504444', 
                     'PCBA-504466', 'PCBA-504467', 'PCBA-504706', 'PCBA-504842', 'PCBA-504845', 'PCBA-504847', 
                     'PCBA-504891', 'PCBA-540276', 'PCBA-540317', 'PCBA-588342', 'PCBA-588453', 'PCBA-588456', 
                     'PCBA-588579', 'PCBA-588590', 'PCBA-588591', 'PCBA-588795', 'PCBA-588855', 'PCBA-602179', 
                     'PCBA-602233', 'PCBA-602310', 'PCBA-602313', 'PCBA-602332', 'PCBA-624170', 'PCBA-624171', 
                    'PCBA-624173', 'PCBA-624202', 'PCBA-624246', 'PCBA-624287', 'PCBA-624288', 'PCBA-624291', 
                     'PCBA-624296', 'PCBA-624297', 'PCBA-624417', 'PCBA-651635', 'PCBA-651644', 'PCBA-651768', 
                     'PCBA-651965', 'PCBA-652025', 'PCBA-652104', 'PCBA-652105', 'PCBA-652106', 'PCBA-686970', 
                     'PCBA-686978', 'PCBA-686979', 'PCBA-720504', 'PCBA-720532', 'PCBA-720542', 'PCBA-720551', 
                     'PCBA-720553', 'PCBA-720579', 'PCBA-720580', 'PCBA-720707', 'PCBA-720708', 'PCBA-720709', 
                     'PCBA-720711', 'PCBA-743255', 'PCBA-743266', 'PCBA-875', 'PCBA-881', 'PCBA-883', 'PCBA-884', 
                     'PCBA-885', 'PCBA-887', 'PCBA-891', 'PCBA-899', 'PCBA-902', 'PCBA-903', 'PCBA-904', 
                     'PCBA-912', 'PCBA-914', 'PCBA-915', 'PCBA-924', 'PCBA-925', 'PCBA-926', 'PCBA-927', 
                     'PCBA-938', 'PCBA-995'],
             'qm7':['u0_atom'],
             'qm8':['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2', 'E1-PBE0', 
                    'E2-PBE0', 'f1-PBE0', 'f2-PBE0', 'E1-PBE0.1', 'E2-PBE0.1', 
                    'f1-PBE0.1', 'f2-PBE0.1', 'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM'],
             'qm9':['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 
                    'h298', 'g298', 'cv', 'u0_atom', 'u298_atom', 'h298_atom', 'g298_atom'],
             'covid-19':['3CL_enzymatic_activity', 'ACE2_enzymatic_activity', 'HEK293_cell_line_toxicity_',
                    'Human_fibroblast_toxicity', 'MERS_Pseudotyped_particle_entry', 
                    'MERS_Pseudotyped_particle_entry_(Huh7_tox_counterscreen)', 
                    'SARS-CoV_Pseudotyped_particle_entry', 
                    'SARS-CoV_Pseudotyped_particle_entry_(VeroE6_tox_counterscreen)',
                    'SARS-CoV-2_cytopathic_effect_(CPE)', 
                    'SARS-CoV-2_cytopathic_effect_(host_tox_counterscreen)', 
                    'Spike-ACE2_protein-protein_interaction_(AlphaLISA)', 
                    'Spike-ACE2_protein-protein_interaction_(TruHit_Counterscreen)', 
                    'TMPRSS2_enzymatic_activity'],
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

def run_a_train_epoch(model, data_loader, loss_func, optimizer, args):
    model.train()
    train_metric = Meter()  # for each epoch
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        atom_feats = bg.ndata.pop('h') # 39
        bond_feats = bg.edata.pop('e') # 11

        # transfer the data to device(cpu or cuda)
        labels, masks, atom_feats, bond_feats = labels.to(args['device']), masks.to(args['device']), atom_feats.to(
            args['device']), bond_feats.to(args['device'])
        # outputs = model(bg, atom_feats) if args['model'] in ['gcn', 'gat'] else model(bg, atom_feats, bond_feats)
        outputs = model(bg, atom_feats, bond_feats)
        loss = (loss_func(outputs, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs.cpu()
        labels.cpu()
        masks.cpu()
        atom_feats.cpu()
        bond_feats.cpu()
        loss.cpu()
        torch.cuda.empty_cache()

        train_metric.update(outputs, labels, masks)

    if args['metric'] == 'rmse':
        rmse_score = np.mean(train_metric.compute_metric(args['metric']))  # in case of multi-tasks
        mae_score = np.mean(train_metric.compute_metric('mae'))  # in case of multi-tasks
        r2_score = np.mean(train_metric.compute_metric('r2'))  # in case of multi-tasks
        return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
    else:
        roc_score = np.mean(train_metric.compute_metric(args['metric']))  # in case of multi-tasks
        prc_score = np.mean(train_metric.compute_metric('prc_auc'))  # in case of multi-tasks
        return {'roc_auc': roc_score, 'prc_auc': prc_score}

def run_an_eval_epoch(model, data_loader, args):
    model.eval()
    eval_metric = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            atom_feats = bg.ndata.pop('h')
            bond_feats = bg.edata.pop('e')

            # transfer the data to device(cpu or cuda)
            labels, masks, atom_feats, bond_feats = labels.to(args['device']), masks.to(args['device']), atom_feats.to(
                args['device']), bond_feats.to(args['device'])
            outputs = model(bg, atom_feats, bond_feats)
            outputs.cpu()
            labels.cpu()
            masks.cpu()
            atom_feats.cpu()
            bond_feats.cpu()
            # loss.cpu()
            torch.cuda.empty_cache()
            eval_metric.update(outputs, labels, masks)
    if args['metric'] == 'rmse':
        rmse_score = np.mean(eval_metric.compute_metric(args['metric']))  # in case of multi-tasks
        mae_score = np.mean(eval_metric.compute_metric('mae'))  # in case of multi-tasks
        r2_score = np.mean(eval_metric.compute_metric('r2'))  # in case of multi-tasks
        return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
    else:
        roc_score = np.mean(eval_metric.compute_metric(args['metric']))  # in case of multi-tasks
        prc_score = np.mean(eval_metric.compute_metric('prc_auc'))  # in case of multi-tasks
        return {'roc_auc': roc_score, 'prc_auc': prc_score}

def get_pos_weight(my_dataset):
    num_pos = B.sum(my_dataset.labels, dim=0)
    num_indices = B.tensor(len(my_dataset.labels))
    return (num_indices - num_pos) / num_pos

def all_one_zeros(series):
    if (len(series.dropna().unique()) == 2):
        flag = False
    else:
        flag = True
    return flag

sm_num = 20
data_label = sys.argv[1] 
dataset_label = data_label
file_name='./dataset/dataset_used_for_modeling/'+dataset_label+'.csv'
model_name = sys.argv[2]  # 'gcn' or 'mpnn' or 'gat' or 'attentivefp'
CUDA = sys.argv[3] 
task_type = sys.argv[4] #cla or reg
device = torch.device('cuda:'+CUDA if torch.cuda.is_available() else 'cpu')
my_df = pd.read_csv(file_name)

AtomFeaturizer = AttentiveFPAtomFeaturizer
BondFeaturizer = AttentiveFPBondFeaturizer
epochs = 300
batch_size = 128*5
patience = 50
opt_iters = 50
repetitions = 50
num_workers = 0
args = {'device': device, 'task': data_label, 'metric': 'roc_auc' if task_type == 'cla' else 'rmse', 'model': model_name}
tasks = tasks_dic[args['task']]
Hspace = {'gcn': dict(l2=hp.choice('l2', [0, 10 ** -8, 10 ** -6, 10 ** -4]),
                      lr=hp.choice('lr', [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]),
                              atom_d_out = hp.choice('atom_d_out', [32, 64, 128]),  
                              bond_d_out = hp.choice('bond_d_out', [32, 64, 128]),                     
                      sigma = hp.loguniform('sigma', np.log(0.01), np.log(100)),
                      gcn_hidden_feats=hp.choice('gcn_hidden_feats',
                                                 [[128, 128], [256, 256], [128, 64], [256, 128]]),
                      classifier_hidden_feats=hp.choice('classifier_hidden_feats', [128, 64, 256])),
          'mpnn': dict(l2=hp.choice('l2', [0, 10 ** -8, 10 ** -6, 10 ** -4]),
                       lr=hp.choice('lr', [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]),
                        atom_d_out = hp.choice('atom_d_out', [32, 64, 128]),  
                        bond_d_out = hp.choice('bond_d_out', [32, 64, 128]),                     
                        sigma = hp.loguniform('sigma', np.log(0.01), np.log(100)),
                       node_hidden_dim=hp.choice('node_hidden_dim', [64, 32, 16]),
                       edge_hidden_dim=hp.choice('edge_hidden_dim', [64, 32, 16]),
                       num_layer_set2set=hp.choice('num_layer_set2set', [2, 3, 4])),
          'gat': dict(l2=hp.choice('l2', [0, 10 ** -8, 10 ** -6, 10 ** -4]),
                      lr=hp.choice('lr', [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]),
                        atom_d_out = hp.choice('atom_d_out', [32, 64, 128]),  
                        bond_d_out = hp.choice('bond_d_out', [32, 64, 128]),                      
                      sigma = hp.loguniform('sigma', np.log(0.01), np.log(100)),
                      gat_hidden_feats=hp.choice('gat_hidden_feats',
                                                 [[128, 128], [256, 256], [128, 64], [256, 128]]),
                      num_heads=hp.choice('num_heads', [[2, 2], [3, 3], [4, 4], [4, 3], [3, 2]]),
                      classifier_hidden_feats=hp.choice('classifier_hidden_feats', [128, 64, 256])),
          'attentivefp': dict(l2=hp.choice('l2', [0, 10 ** -8, 10 ** -6, 10 ** -4]),
                              lr=hp.choice('lr', [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]), 
                              atom_d_out = hp.choice('atom_d_out', [32, 64, 128]),  
                              bond_d_out = hp.choice('bond_d_out', [32, 64, 128]),                
                              sigma = hp.loguniform('sigma', np.log(0.01), np.log(100)),
                              num_layers=hp.choice('num_layers', [2, 3, 4, 5, 6]),
                              num_timesteps=hp.choice('num_timesteps', [1, 2, 3, 4, 5]),
                              dropout=hp.choice('dropout', [0, 0.1, 0.3, 0.5]),
                              graph_feat_size=hp.choice('graph_feat_size', [50, 100, 200, 300]))}
hyper_space = Hspace[args['model']]

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
        # x = 2*np.pi*self.coefficients[None]*x[..., None]
        x = 2*np.pi*torch.matmul(x, self.coefficients)

        return torch.cat([torch.cos(x), torch.sin(x)], -1)

class EGNN(nn.Module):
    def __init__(self, BaseGNN, atom_inputs, atom_d_out, bond_inputs, bond_d_out, sigma, args):
        super(EGNN, self).__init__()
        self.BaseGNN = BaseGNN
        self.sigma = sigma
        self.atom_embedding = PLE(atom_inputs, atom_d_out, sigma)
        self.atom_linear = nn.Linear(atom_d_out * 2, atom_inputs)
        self.bond_embedding = PLE(bond_inputs, bond_d_out, sigma)
        self.bond_linear = nn.Linear(bond_d_out * 2, bond_inputs)
        self.args = args

    def forward(self, bg, atom_feats, bond_feats):
        embed_atom_feats = self.atom_embedding(atom_feats)
        embed_bond_feats = self.bond_embedding(bond_feats)
        embed_atom_feats = F.relu(self.atom_linear(embed_atom_feats))
        embed_bond_feats = F.relu(self.bond_linear(embed_bond_feats))
        outputs = self.BaseGNN(bg, embed_atom_feats) if self.args['model'] in ['gcn', 'gat'] else self.BaseGNN(bg, embed_atom_feats, embed_bond_feats)
        return outputs

# get the df and generate graph, attention for graph generation for some bad smiles
# my_df.iloc[:, 0:-1], except with 'group'
my_dataset: MoleculeCSVDataset = csv_dataset.MoleculeCSVDataset(my_df.iloc[:, 0:-1], smiles_to_bigraph, AtomFeaturizer,
                                                                BondFeaturizer, 'cano_smiles',
                                                                file_name.replace('.csv', '.bin'))
if task_type == 'cla':
    pos_weight = get_pos_weight(my_dataset)
else:
    pos_weight = None

# get the training, validation, and test sets
dataseta=pd.read_csv(file_name)
tr_indx = my_df[dataseta.group == 'train'].index
val_indx = my_df[dataseta.group == 'valid'].index
te_indx = my_df[dataseta.group == 'test'].index
# train_set, val_set, test_set = Subset(my_dataset, tr_indx), Subset(my_dataset, val_indx), Subset(my_dataset, te_indx)
train_loader = DataLoader(Subset(my_dataset, tr_indx), batch_size=batch_size, shuffle=True,
                          collate_fn=collate_molgraphs, num_workers=num_workers)
val_loader = DataLoader(Subset(my_dataset, val_indx), batch_size=batch_size, shuffle=True,
                        collate_fn=collate_molgraphs, num_workers=num_workers)
test_loader = DataLoader(Subset(my_dataset, te_indx), batch_size=batch_size, shuffle=True,
                         collate_fn=collate_molgraphs, num_workers=num_workers)

def hyper_opt(hyper_paras):
    # get the model instance
    if model_name == 'gcn':
        BaseGNN = GCNClassifier(in_feats=AtomFeaturizer.feat_size('h'),
                                 gcn_hidden_feats=hyper_paras['gcn_hidden_feats'],
                                 n_tasks=len(tasks), classifier_hidden_feats=hyper_paras['classifier_hidden_feats'])
        my_model = EGNN(BaseGNN, AtomFeaturizer.feat_size('h'), hyper_paras['atom_d_out'], BondFeaturizer.feat_size('e'), hyper_paras['bond_d_out'], hyper_paras['sigma'], args)
        model_file_name = './saved_model/%s_%s_%s_%.6f_%s_%s_new.pth' % (args['model'], args['task'],
                                                                hyper_paras['l2'], hyper_paras['lr'],
                                                                hyper_paras['gcn_hidden_feats'],
                                                                hyper_paras['classifier_hidden_feats'])
    elif model_name == 'mpnn':
        my_model = MPNNModel(node_input_dim=AtomFeaturizer.feat_size('h'), edge_input_dim=BondFeaturizer.feat_size('e'),
                             output_dim=len(tasks), node_hidden_dim=hyper_paras['node_hidden_dim'],
                             edge_hidden_dim=hyper_paras['edge_hidden_dim'],
                             num_layer_set2set=hyper_paras['num_layer_set2set'])
        model_file_name = './saved_model/%s_%s_%s_%.6f_%s_%s_%s.pth' % (args['model'], args['task'],
                                                                        hyper_paras['l2'], hyper_paras['lr'],
                                                                        hyper_paras['node_hidden_dim'],
                                                                        hyper_paras['edge_hidden_dim'],
                                                                        hyper_paras['num_layer_set2set'])
    elif model_name == 'attentivefp':
        BaseGNN = AttentiveFP(node_feat_size=AtomFeaturizer.feat_size('h'),
                               edge_feat_size=BondFeaturizer.feat_size('e'),
                               num_layers=hyper_paras['num_layers'], num_timesteps=hyper_paras['num_timesteps'],
                               graph_feat_size=hyper_paras['graph_feat_size'], output_size=len(tasks),
                               dropout=hyper_paras['dropout'])
        my_model = EGNN(BaseGNN, AtomFeaturizer.feat_size('h'), hyper_paras['atom_d_out'], BondFeaturizer.feat_size('e'), hyper_paras['bond_d_out'], hyper_paras['sigma'], args)

        model_file_name = './saved_model/%s_%s_%s_%.6f_%s_%s_%s_%s.pth' % (args['model'], args['task'],
                                                                           hyper_paras['l2'], hyper_paras['lr'],
                                                                           hyper_paras['num_layers'],
                                                                           hyper_paras['num_timesteps'],
                                                                           hyper_paras['graph_feat_size'],
                                                                           hyper_paras['dropout'])
    else:
        BaseGNN = GATClassifier(in_feats=AtomFeaturizer.feat_size('h'),
                                 gat_hidden_feats=hyper_paras['gat_hidden_feats'],
                                 num_heads=hyper_paras['num_heads'], n_tasks=len(tasks),
                                 classifier_hidden_feats=hyper_paras['classifier_hidden_feats'])
        my_model = EGNN(BaseGNN, AtomFeaturizer.feat_size('h'), hyper_paras['atom_d_out'], BondFeaturizer.feat_size('e'), hyper_paras['bond_d_out'], hyper_paras['sigma'], args)
        model_file_name = './saved_model/%s_%s_%s_%.6f_%s_%s_%s_new.pth' % (args['model'], args['task'],
                                                                        hyper_paras['l2'], hyper_paras['lr'],
                                                                        hyper_paras['gat_hidden_feats'],
                                                                        hyper_paras['num_heads'],
                                                                        hyper_paras['classifier_hidden_feats'])
    optimizer = torch.optim.Adam(my_model.parameters(), lr=hyper_paras['lr'], weight_decay=hyper_paras['l2'])
    # print("parameters:", my_model.parameters())
    if task_type == 'reg':
        loss_func = MSELoss(reduction='none')
        stopper = EarlyStopping(mode='lower', patience=patience, filename=model_file_name)
    else:
        loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight.to(args['device']))
        stopper = EarlyStopping(mode='higher', patience=patience, filename=model_file_name)
    my_model.to(device)

    for j in range(epochs):
        # training
        run_a_train_epoch(my_model, train_loader, loss_func, optimizer, args)

        # early stopping
        val_scores = run_an_eval_epoch(my_model, val_loader, args)
        early_stop = stopper.step(val_scores[args['metric']], my_model)

        if early_stop:
            break
    stopper.load_checkpoint(my_model)
    tr_scores = run_an_eval_epoch(my_model, train_loader, args)
    val_scores = run_an_eval_epoch(my_model, val_loader, args)
    te_scores = run_an_eval_epoch(my_model, test_loader, args)
    print({'train': tr_scores, 'valid': val_scores, 'test': te_scores})
    feedback = val_scores[args['metric']] if task_type == 'reg' else (1 - val_scores[args['metric']])
    my_model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    return feedback


# start hyper-parameters optimization
print('******hyper-parameter optimization is starting now******')
trials = Trials()
opt_res = fmin(hyper_opt, hyper_space, algo=tpe.suggest, max_evals=opt_iters, trials=trials)

# hyper-parameters optimization is over
print('******hyper-parameter optimization is over******')
print('the best hyper-parameters settings for ' + args['task'] + ' ' + args['model'] + ' are:  ', opt_res)

# construct the model based on the optimal hyper-parameters
l2_ls = [0, 10 ** -8, 10 ** -6, 10 ** -4]
lr_ls = [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]
hidden_feats_ls = [(128, 128), (256, 256), (128, 64), (256, 128)]
classifier_hidden_feats_ls = [128, 64, 256]
node_hidden_dim_ls = [64, 32, 16]
edge_hidden_dim_ls = [64, 32, 16]
num_layer_set2set_ls = [2, 3, 4]
num_heads_ls = [(2, 2), (3, 3), (4, 4), (4, 3), (3, 2)]
num_layers_ls = [2, 3, 4, 5, 6]
num_timesteps_ls = [1, 2, 3, 4, 5]
graph_feat_size_ls = [50, 100, 200, 300]
dropout_ls = [0, 0.1, 0.3, 0.5]

# 50 repetitions based on the best model
tr_res = []
val_res = []
te_res = []
# regenerate the graphs
if args['task'] == 'muv' or args['task'] == 'toxcast':
    file_name = './dataset/' + args['task'] + '_new.csv'
    my_df = pd.read_csv(file_name)
    my_dataset = csv_dataset.MoleculeCSVDataset(my_df, smiles_to_bigraph, AtomFeaturizer, BondFeaturizer,
                                                'cano_smiles', file_name.replace('.csv', '.bin'))
else:
    my_df.drop(columns=['group'], inplace=True)

for split in range(1, repetitions + 1):
    # splitting the data set for classification
    if args['metric'] == 'roc_auc':
        seed = split
        while True:
            training_data, data_te = train_test_split(my_df, test_size=0.1, random_state=seed)
            # the training set was further splitted into the training set and validation set
            data_tr, data_va = train_test_split(training_data, test_size=0.1, random_state=seed)
            if np.any(data_tr[tasks].apply(all_one_zeros)) or \
                    np.any(data_va[tasks].apply(all_one_zeros)) or \
                    np.any(data_te[tasks].apply(all_one_zeros)):
                print('\ninvalid random seed {} due to one class presented in the splitted {} sets...'.format(seed,
                                                                                                              args[
                                                                                                                  'task']))
                print('Changing to another random seed...\n')
                seed = np.random.randint(50, 999999)
            else:
                print('random seed used in repetition {} is {}'.format(split, seed))
                break
    else:
        training_data, data_te = train_test_split(my_df, test_size=0.1, random_state=split)
        # the training set was further splitted into the training set and validation set
        data_tr, data_va = train_test_split(training_data, test_size=0.1, random_state=split)
    tr_indx, val_indx, te_indx = data_tr.index, data_va.index, data_te.index
    train_loader = DataLoader(Subset(my_dataset, tr_indx), batch_size=batch_size, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=num_workers)
    val_loader = DataLoader(Subset(my_dataset, val_indx), batch_size=batch_size, shuffle=True,
                            collate_fn=collate_molgraphs, num_workers=num_workers)
    test_loader = DataLoader(Subset(my_dataset, te_indx), batch_size=batch_size, shuffle=True,
                             collate_fn=collate_molgraphs, num_workers=num_workers)
    best_model_file = './saved_model/%s_%s_bst_%s.pth' % (args['model'], args['task'], split)

    if model_name == 'gcn':
        BaseGNN = GCNClassifier(in_feats=AtomFeaturizer.feat_size('h'),
                                 gcn_hidden_feats=hidden_feats_ls[opt_res['gcn_hidden_feats']],
                                 n_tasks=len(tasks), classifier_hidden_feats=classifier_hidden_feats_ls[
                                       opt_res['classifier_hidden_feats']])
        best_model = EGNN(BaseGNN, AtomFeaturizer.feat_size('h'), opt_res['atom_d_out'], BondFeaturizer.feat_size('e'), opt_res['bond_d_out'], opt_res['sigma'], args)

    elif model_name == 'gat':
        BaseGNN = GATClassifier(in_feats=AtomFeaturizer.feat_size('h'),
                                   gat_hidden_feats=hidden_feats_ls[opt_res['gat_hidden_feats']],
                                   num_heads=num_heads_ls[opt_res['num_heads']], n_tasks=len(tasks),
                                   classifier_hidden_feats=classifier_hidden_feats_ls[
                                       opt_res['classifier_hidden_feats']])
        best_model = EGNN(BaseGNN, AtomFeaturizer.feat_size('h'), opt_res['atom_d_out'], BondFeaturizer.feat_size('e'), opt_res['bond_d_out'], opt_res['sigma'], args)

    elif model_name == 'attentivefp':
        BaseGNN = AttentiveFP(node_feat_size=AtomFeaturizer.feat_size('h'),
                                 edge_feat_size=BondFeaturizer.feat_size('e'),
                                 num_layers=num_layers_ls[opt_res['num_layers']],
                                 num_timesteps=num_timesteps_ls[opt_res['num_timesteps']],
                                 graph_feat_size=graph_feat_size_ls[opt_res['graph_feat_size']], output_size=len(tasks),
                                 dropout=dropout_ls[opt_res['dropout']])
        best_model = EGNN(BaseGNN, AtomFeaturizer.feat_size('h'), opt_res['atom_d_out'], BondFeaturizer.feat_size('e'), opt_res['bond_d_out'], opt_res['sigma'], args)
    else:
        BaseGNN = MPNNModel(node_input_dim=AtomFeaturizer.feat_size('h'),
                               edge_input_dim=BondFeaturizer.feat_size('e'),
                               output_dim=len(tasks), node_hidden_dim=node_hidden_dim_ls[opt_res['node_hidden_dim']],
                               edge_hidden_dim=edge_hidden_dim_ls[opt_res['edge_hidden_dim']],
                               num_layer_set2set=num_layer_set2set_ls[opt_res['num_layer_set2set']])
        best_model = EGNN(BaseGNN, AtomFeaturizer.feat_size('h'), opt_res['atom_d_out'], BondFeaturizer.feat_size('e'), opt_res['bond_d_out'], opt_res['sigma'], args)

    best_optimizer = torch.optim.Adam(best_model.parameters(), lr=lr_ls[opt_res['lr']],
                                      weight_decay=l2_ls[opt_res['l2']])
    if task_type == 'reg':
        loss_func = MSELoss(reduction='none')
        stopper = EarlyStopping(mode='lower', patience=patience, filename=best_model_file)
    else:
        loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight.to(args['device']))
        stopper = EarlyStopping(mode='higher', patience=patience, filename=best_model_file)
    best_model.to(device)

    for j in range(epochs):
        # training
        st = time.time()
        run_a_train_epoch(best_model, train_loader, loss_func, best_optimizer, args)
        end = time.time()
        # early stopping
        train_scores = run_an_eval_epoch(best_model, train_loader, args)
        val_scores = run_an_eval_epoch(best_model, val_loader, args)
        early_stop = stopper.step(val_scores[args['metric']], best_model)
        if early_stop:
            break
        print(
            'task:{} repetition {:d}/{:d} epoch {:d}/{:d}, training {} {:.3f}, validation {} {:.3f}, time:{:.3f}S'.format(
                args['task'], split, repetitions, j + 1, epochs, args['metric'], train_scores[args['metric']],
                args['metric'],
                val_scores[args['metric']], end - st))
    stopper.load_checkpoint(best_model)
    tr_scores = run_an_eval_epoch(best_model, train_loader, args)
    val_scores = run_an_eval_epoch(best_model, val_loader, args)
    te_scores = run_an_eval_epoch(best_model, test_loader, args)
    tr_res.append(tr_scores)
    val_res.append(val_scores)
    te_res.append(te_scores)
if task_type == 'reg':
    cols = ['rmse', 'mae', 'r2']
else:
    cols = ['roc_auc', 'prc_auc']
tr = [list(item.values()) for item in tr_res]
val = [list(item.values()) for item in val_res]
te = [list(item.values()) for item in te_res]
tr_pd = pd.DataFrame(tr, columns=cols)
tr_pd['split'] = range(1, repetitions + 1)
tr_pd['set'] = 'train'
val_pd = pd.DataFrame(val, columns=cols)
val_pd['split'] = range(1, repetitions + 1)
val_pd['set'] = 'validation'
te_pd = pd.DataFrame(te, columns=cols)
te_pd['split'] = range(1, repetitions + 1)
te_pd['set'] = 'test'
sta_pd = pd.concat([tr_pd, val_pd, te_pd], ignore_index=True)
sta_pd['model'] = args['model']
sta_pd['dataset'] = args['task']
sta_pd.to_csv('./stat_res/{}_{}_statistical_results_split50.csv'.format(args['task'], args['model']), index=False)

print('training mean:', np.mean(tr, axis=0), 'training std:', np.std(tr, axis=0))
print('validation mean:', np.mean(val, axis=0), 'validation std:', np.std(val, axis=0))
print('testing mean:', np.mean(te, axis=0), 'test std:', np.std(te, axis=0))
end_time = time.time()
print('the total elapsed time is', end_time - start_time, 'S')
