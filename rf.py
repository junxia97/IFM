import time
import warnings
import sys
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectPercentile, f_classif, SelectFromModel
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc, mean_squared_error, \
    r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, XGBClassifier
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

start = time.time()
warnings.filterwarnings("ignore")


def standardize(col):
    return (col - np.mean(col)) / np.std(col)


# the metrics for classification
def statistical(y_true, y_pred, y_pro):
    c_mat = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = list(c_mat.flatten())
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tn + fp + fn + tp)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    auc_prc = auc(precision_recall_curve(y_true, y_pro, pos_label=1)[1],
                  precision_recall_curve(y_true, y_pro, pos_label=1)[0])
    auc_roc = roc_auc_score(y_true, y_pro)
    return tn, fp, fn, tp, se, sp, acc, mcc, auc_prc, auc_roc


def all_one_zeros(data):
    if (len(np.unique(data)) == 2):
        flag = False
    else:
        flag = True
    return flag


feature_selection = False
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
file_name = sys.argv[1]  # './dataset/esol_moe_pubsubfp.csv'
task_type = sys.argv[2]  # 'reg' or 'cla'
dataset_label = file_name.split('/')[-1].split('_')[0]  # dataset_label = 'esol'
tasks = tasks_dic[dataset_label]
OPT_ITERS = 50
repetitions = 50
num_pools = 5
space_ = {'n_estimators': hp.choice('n_estimators', [10, 50, 100, 200, 300, 400, 500]),
          'max_depth': hp.choice('max_depth', range(3, 12)),
          'min_samples_leaf': hp.choice('min_samples_leaf', [1, 3, 5, 10, 20, 50]),
          'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 0.01),
          'max_features': hp.choice('max_features', ['sqrt', 'log2', 0.7, 0.8, 0.9])
          }
n_estimators_ls = [10, 50, 100, 200, 300, 400, 500]
max_depth_ls = range(3, 12)
min_samples_leaf_ls = [1, 3, 5, 10, 20, 50]
max_features_ls = ['sqrt', 'log2', 0.7, 0.8, 0.9]
dataset = pd.read_csv(file_name)
pd_res = []


def hyper_runing(subtask):
    cols = [subtask]
    cols.extend(dataset.columns[(len(tasks) + 1):])
    sub_dataset = dataset[cols]

    # detect the na in the subtask (y cloumn)
    rm_index = sub_dataset[subtask][sub_dataset[subtask].isnull()].index
    sub_dataset.drop(index=rm_index, inplace=True)

    # remove the features with na
    if dataset_label != 'hiv':
        sub_dataset = sub_dataset.dropna(axis=1)
    else:
        sub_dataset = sub_dataset.dropna(axis=0)
    # *******************
    # demension reduction
    # *******************
    # Removing features with low variance
    # threshold = 0.05
    data_fea_var = sub_dataset.iloc[:, 2:].var()
    del_fea1 = list(data_fea_var[data_fea_var <= 0.05].index)
    sub_dataset.drop(columns=del_fea1, inplace=True)

    # pair correlations
    # threshold = 0.95
    data_fea_corr = sub_dataset.iloc[:, 2:].corr()
    del_fea2_col = []
    del_fea2_ind = []
    length = data_fea_corr.shape[1]
    for i in range(length):
        for j in range(i + 1, length):
            if abs(data_fea_corr.iloc[i, j]) >= 0.95:
                del_fea2_col.append(data_fea_corr.columns[i])
                del_fea2_ind.append(data_fea_corr.index[j])
    sub_dataset.drop(columns=del_fea2_ind, inplace=True)

    # standardize the features
    cols_ = list(sub_dataset.columns)[2:]
    sub_dataset[cols_] = sub_dataset[cols_].apply(standardize, axis=0)

    # get the attentivefp data splits
    data_tr = sub_dataset[sub_dataset['group'] == 'train']
    data_va = sub_dataset[sub_dataset['group'] == 'valid']
    data_te = sub_dataset[sub_dataset['group'] == 'test']

    # prepare data for training
    # training set
    data_tr_y = data_tr[subtask].values.reshape(-1, 1)
    data_tr_x = np.array(data_tr.iloc[:, 2:].values)

    # validation set
    data_va_y = data_va[subtask].values.reshape(-1, 1)
    data_va_x = np.array(data_va.iloc[:, 2:].values)

    # test set
    data_te_y = data_te[subtask].values.reshape(-1, 1)
    data_te_x = np.array(data_te.iloc[:, 2:].values)

    if feature_selection:
        # univariate feature selection
        trans1 = SelectPercentile(f_classif, percentile=80)
        trans1.fit(data_tr_x, data_tr_y)
        data_tr_x = trans1.transform(data_tr_x)
        data_va_x = trans1.transform(data_va_x)
        data_te_x = trans1.transform(data_te_x)

        # select from model
        clf = XGBClassifier(n_jobs=12, random_state=1)
        clf = clf.fit(data_tr_x, data_tr_y)
        trans2 = SelectFromModel(clf, prefit=True)

        data_tr_x = trans2.transform(data_tr_x)
        data_va_x = trans2.transform(data_va_x)
        data_te_x = trans2.transform(data_te_x)

    num_fea = data_tr_x.shape[1]
    print('the num of retained features for the ' + dataset_label + ' ' + subtask + ' is:', num_fea)

    def hyper_opt(args):
        model = RandomForestClassifier(**args, n_jobs=6, random_state=1, verbose=0, class_weight='balanced') \
            if task_type == 'cla' else RandomForestRegressor(**args, n_jobs=6, random_state=1, verbose=0)
        model.fit(data_tr_x, data_tr_y)
        val_preds = model.predict_proba(data_va_x) if task_type == 'cla' else model.predict(data_va_x)
        loss = 1 - roc_auc_score(data_va_y, val_preds[:, 1]) if task_type == 'cla' else np.sqrt(
            mean_squared_error(data_va_y, val_preds))
        return {'loss': loss, 'status': STATUS_OK}

    # start hyper-parameters optimization
    trials = Trials()
    best_results = fmin(hyper_opt, space_, algo=tpe.suggest, max_evals=OPT_ITERS, trials=trials, show_progressbar=False)
    print('the best hyper-parameters for ' + dataset_label + ' ' + subtask + ' are:  ', best_results)
    best_model = RandomForestClassifier(n_estimators=n_estimators_ls[best_results['n_estimators']],
                                        max_depth=max_depth_ls[best_results['max_depth']],
                                        min_samples_leaf=min_samples_leaf_ls[best_results['min_samples_leaf']],
                                        max_features=max_features_ls[best_results['max_features']],
                                        min_impurity_decrease=best_results['min_impurity_decrease'],
                                        n_jobs=6, random_state=1, verbose=0, class_weight='balanced') \
        if task_type == 'cla' else RandomForestRegressor(
        n_estimators=n_estimators_ls[best_results['n_estimators']],
        max_depth=max_depth_ls[best_results['max_depth']],
        min_samples_leaf=min_samples_leaf_ls[best_results['min_samples_leaf']],
        max_features=max_features_ls[best_results['max_features']],
        min_impurity_decrease=best_results['min_impurity_decrease'],
        n_jobs=6, random_state=1, verbose=0)

    best_model.fit(data_tr_x, data_tr_y)
    num_of_compounds = len(sub_dataset)

    if task_type == 'cla':
        # training set
        tr_pred = best_model.predict_proba(data_tr_x)
        tr_results = [dataset_label, subtask, 'tr', num_fea, num_of_compounds, data_tr_y[data_tr_y == 1].shape[0],
                      data_tr_y[data_tr_y == 0].shape[0],
                      data_tr_y[data_tr_y == 0].shape[0] / data_tr_y[data_tr_y == 1].shape[0],
                      n_estimators_ls[best_results['n_estimators']],
                      max_depth_ls[best_results['max_depth']],
                      min_samples_leaf_ls[best_results['min_samples_leaf']],
                      best_results['min_impurity_decrease'],
                      max_features_ls[best_results['max_features']]]
        tr_results.extend(statistical(data_tr_y, np.argmax(tr_pred, axis=1), tr_pred[:, 1]))

        # validation set
        va_pred = best_model.predict_proba(data_va_x)
        va_results = [dataset_label, subtask, 'va', num_fea, num_of_compounds, data_va_y[data_va_y == 1].shape[0],
                      data_va_y[data_va_y == 0].shape[0],
                      data_va_y[data_va_y == 0].shape[0] / data_va_y[data_va_y == 1].shape[0],
                      n_estimators_ls[best_results['n_estimators']],
                      max_depth_ls[best_results['max_depth']],
                      min_samples_leaf_ls[best_results['min_samples_leaf']],
                      best_results['min_impurity_decrease'],
                      max_features_ls[best_results['max_features']]]
        va_results.extend(statistical(data_va_y, np.argmax(va_pred, axis=1), va_pred[:, 1]))

        # test set
        te_pred = best_model.predict_proba(data_te_x)
        te_results = [dataset_label, subtask, 'te', num_fea, num_of_compounds, data_te_y[data_te_y == 1].shape[0],
                      data_te_y[data_te_y == 0].shape[0],
                      data_te_y[data_te_y == 0].shape[0] / data_te_y[data_te_y == 1].shape[0],
                      n_estimators_ls[best_results['n_estimators']],
                      max_depth_ls[best_results['max_depth']],
                      min_samples_leaf_ls[best_results['min_samples_leaf']],
                      best_results['min_impurity_decrease'],
                      max_features_ls[best_results['max_features']]]
        te_results.extend(statistical(data_te_y, np.argmax(te_pred, axis=1), te_pred[:, 1]))
    else:
        # training set
        tr_pred = best_model.predict(data_tr_x)
        tr_results = [dataset_label, subtask, 'tr', num_fea, num_of_compounds,
                      n_estimators_ls[best_results['n_estimators']],
                      max_depth_ls[best_results['max_depth']],
                      min_samples_leaf_ls[best_results['min_samples_leaf']],
                      best_results['min_impurity_decrease'],
                      max_features_ls[best_results['max_features']],
                      np.sqrt(mean_squared_error(data_tr_y, tr_pred)), r2_score(data_tr_y, tr_pred),
                      mean_absolute_error(data_tr_y, tr_pred)]

        # validation set
        va_pred = best_model.predict(data_va_x)
        va_results = [dataset_label, subtask, 'va', num_fea, num_of_compounds,
                      n_estimators_ls[best_results['n_estimators']],
                      max_depth_ls[best_results['max_depth']],
                      min_samples_leaf_ls[best_results['min_samples_leaf']],
                      best_results['min_impurity_decrease'],
                      max_features_ls[best_results['max_features']],
                      np.sqrt(mean_squared_error(data_va_y, va_pred)), r2_score(data_va_y, va_pred),
                      mean_absolute_error(data_va_y, va_pred)]

        # test set
        te_pred = best_model.predict(data_te_x)
        te_results = [dataset_label, subtask, 'te', num_fea, num_of_compounds,
                      n_estimators_ls[best_results['n_estimators']],
                      max_depth_ls[best_results['max_depth']],
                      min_samples_leaf_ls[best_results['min_samples_leaf']],
                      best_results['min_impurity_decrease'],
                      max_features_ls[best_results['max_features']],
                      np.sqrt(mean_squared_error(data_te_y, te_pred)), r2_score(data_te_y, te_pred),
                      mean_absolute_error(data_te_y, te_pred)]
    return tr_results, va_results, te_results


pool = multiprocessing.Pool(num_pools)
res = pool.starmap(hyper_runing, zip(tasks))
pool.close()
pool.join()
for item in res:
    for i in range(3):
        pd_res.append(item[i])
if task_type == 'cla':
    best_hyper = pd.DataFrame(pd_res, columns=['dataset', 'subtask', 'set',
                                               'num_of_retained_feature',
                                               'num_of_compounds', 'postives',
                                               'negtives', 'negtives/postives',
                                               'n_estimators', 'max_depth', 'min_samples_leaf',
                                               'min_impurity_decrease', 'max_features',
                                               'tn', 'fp', 'fn', 'tp', 'se', 'sp',
                                               'acc', 'mcc', 'auc_prc', 'auc_roc'])
else:
    best_hyper = pd.DataFrame(pd_res, columns=['dataset', 'subtask', 'set',
                                               'num_of_retained_feature',
                                               'num_of_compounds', 'n_estimators', 'max_depth', 'min_samples_leaf',
                                               'min_impurity_decrease', 'max_features', 'rmse', 'r2', 'mae'])
best_hyper.to_csv('./stat_res/' + dataset_label + '_moe_pubsub_rf_hyperopt_info.csv', index=0)

if task_type == 'cla':
    print('train', best_hyper[best_hyper['set'] == 'tr']['auc_roc'].mean(), best_hyper[best_hyper['set'] == 'tr']['auc_prc'].mean())
    print('valid', best_hyper[best_hyper['set'] == 'va']['auc_roc'].mean(), best_hyper[best_hyper['set'] == 'va']['auc_prc'].mean())
    print('test', best_hyper[best_hyper['set'] == 'te']['auc_roc'].mean(), best_hyper[best_hyper['set'] == 'te']['auc_prc'].mean())
else:
    print('train', best_hyper[best_hyper['set'] == 'tr']['rmse'].mean(), best_hyper[best_hyper['set'] == 'tr']['r2'].mean(), best_hyper[best_hyper['set'] == 'tr']['mae'].mean())
    print('valid', best_hyper[best_hyper['set'] == 'va']['rmse'].mean(), best_hyper[best_hyper['set'] == 'va']['r2'].mean(), best_hyper[best_hyper['set'] == 'va']['mae'].mean())
    print('test', best_hyper[best_hyper['set'] == 'te']['rmse'].mean(), best_hyper[best_hyper['set'] == 'te']['r2'].mean(), best_hyper[best_hyper['set'] == 'te']['mae'].mean())

# 50 repetitions based on thr best hypers
dataset.drop(columns=['group'], inplace=True)
pd_res = []


def best_model_runing(split):
    seed = split
    if task_type == 'cla':
        while True:
            training_data, data_te = train_test_split(sub_dataset, test_size=0.1, random_state=seed)
            # the training set was further splited into the training set and validation set
            data_tr, data_va = train_test_split(training_data, test_size=0.1, random_state=seed)
            if (all_one_zeros(data_tr[subtask]) or all_one_zeros(data_va[subtask]) or all_one_zeros(data_te[subtask])):
                print(
                    '\ninvalid random seed {} due to one class presented in the {} splitted sets...'.format(seed,
                                                                                                            subtask))
                print('Changing to another random seed...\n')
                seed = np.random.randint(50, 999999)
            else:
                print('random seed used in repetition {} is {}'.format(split, seed))
                break
    else:
        training_data, data_te = train_test_split(sub_dataset, test_size=0.1, random_state=seed)
        # the training set was further splited into the training set and validation set
        data_tr, data_va = train_test_split(training_data, test_size=0.1, random_state=seed)

    # prepare data for training
    # training set
    data_tr_y = data_tr[subtask].values.reshape(-1, 1)
    data_tr_x = np.array(data_tr.iloc[:, 1:].values)

    # validation set
    data_va_y = data_va[subtask].values.reshape(-1, 1)
    data_va_x = np.array(data_va.iloc[:, 1:].values)

    # test set
    data_te_y = data_te[subtask].values.reshape(-1, 1)
    data_te_x = np.array(data_te.iloc[:, 1:].values)

    if feature_selection:
        # univariate feature selection
        trans1 = SelectPercentile(f_classif, percentile=80)
        trans1.fit(data_tr_x, data_tr_y)
        data_tr_x = trans1.transform(data_tr_x)
        data_va_x = trans1.transform(data_va_x)
        data_te_x = trans1.transform(data_te_x)

        # select from model
        clf = XGBClassifier(n_jobs=6, random_state=1)
        clf = clf.fit(data_tr_x, data_tr_y)
        trans2 = SelectFromModel(clf, prefit=True)

        data_tr_x = trans2.transform(data_tr_x)
        data_va_x = trans2.transform(data_va_x)
        data_te_x = trans2.transform(data_te_x)

    num_fea = data_tr_x.shape[1]
    model = RandomForestClassifier(n_estimators=best_hyper[best_hyper.subtask == subtask].iloc[0,]['n_estimators'],
                                   max_depth=best_hyper[best_hyper.subtask == subtask].iloc[0,]['max_depth'],
                                   min_samples_leaf=best_hyper[best_hyper.subtask == subtask].iloc[0,][
                                       'min_samples_leaf'],
                                   max_features=best_hyper[best_hyper.subtask == subtask].iloc[0,]['max_features'],
                                   min_impurity_decrease=best_hyper[best_hyper.subtask == subtask].iloc[0,][
                                       'min_impurity_decrease'],
                                   n_jobs=6, random_state=1, verbose=0, class_weight='balanced') \
        if task_type == 'cla' else RandomForestRegressor(
        n_estimators=best_hyper[best_hyper.subtask == subtask].iloc[0,]['n_estimators'],
        max_depth=best_hyper[best_hyper.subtask == subtask].iloc[0,]['max_depth'],
        min_samples_leaf=best_hyper[best_hyper.subtask == subtask].iloc[0,]['min_samples_leaf'],
        max_features=best_hyper[best_hyper.subtask == subtask].iloc[0,]['max_features'],
        min_impurity_decrease=best_hyper[best_hyper.subtask == subtask].iloc[0,]['min_impurity_decrease'],
        n_jobs=6, random_state=1, verbose=0)

    model.fit(data_tr_x, data_tr_y)
    num_of_compounds = sub_dataset.shape[0]
    if task_type == 'cla':
        # training set
        tr_pred = model.predict_proba(data_tr_x)
        tr_results = [split, dataset_label, subtask, 'tr', num_fea, num_of_compounds,
                      data_tr_y[data_tr_y == 1].shape[0],
                      data_tr_y[data_tr_y == 0].shape[0],
                      data_tr_y[data_tr_y == 0].shape[0] / data_tr_y[data_tr_y == 1].shape[0]]
        tr_results.extend(statistical(data_tr_y, np.argmax(tr_pred, axis=1), tr_pred[:, 1]))

        # validation set
        va_pred = model.predict_proba(data_va_x)
        va_results = [split, dataset_label, subtask, 'va', num_fea, num_of_compounds,
                      data_va_y[data_va_y == 1].shape[0],
                      data_va_y[data_va_y == 0].shape[0],
                      data_va_y[data_va_y == 0].shape[0] / data_va_y[data_va_y == 1].shape[0]]
        va_results.extend(statistical(data_va_y, np.argmax(va_pred, axis=1), va_pred[:, 1]))

        # test set
        te_pred = model.predict_proba(data_te_x)
        te_results = [split, dataset_label, subtask, 'te', num_fea, num_of_compounds,
                      data_te_y[data_te_y == 1].shape[0],
                      data_te_y[data_te_y == 0].shape[0],
                      data_te_y[data_te_y == 0].shape[0] / data_te_y[data_te_y == 1].shape[0]]
        te_results.extend(statistical(data_te_y, np.argmax(te_pred, axis=1), te_pred[:, 1]))
    else:
        # training set
        tr_pred = model.predict(data_tr_x)
        tr_results = [split, dataset_label, subtask, 'tr', num_fea, num_of_compounds,
                      np.sqrt(mean_squared_error(data_tr_y, tr_pred)), r2_score(data_tr_y, tr_pred),
                      mean_absolute_error(data_tr_y, tr_pred)]

        # validation set
        va_pred = model.predict(data_va_x)
        va_results = [split, dataset_label, subtask, 'va', num_fea, num_of_compounds,
                      np.sqrt(mean_squared_error(data_va_y, va_pred)), r2_score(data_va_y, va_pred),
                      mean_absolute_error(data_va_y, va_pred)]

        # test set
        te_pred = model.predict(data_te_x)
        te_results = [split, dataset_label, subtask, 'te', num_fea, num_of_compounds,
                      np.sqrt(mean_squared_error(data_te_y, te_pred)), r2_score(data_te_y, te_pred),
                      mean_absolute_error(data_te_y, te_pred)]
    return tr_results, va_results, te_results


for subtask in tasks:
    cols = [subtask]
    cols.extend(dataset.columns[(len(tasks) + 1):])
    # cols.extend(dataset.columns[(617+1):])
    sub_dataset = dataset[cols]

    # detect the NA in the subtask (y cloumn)
    rm_index = sub_dataset[subtask][sub_dataset[subtask].isnull()].index
    sub_dataset.drop(index=rm_index, inplace=True)

    # remove the features with na
    if dataset_label != 'hiv':
        sub_dataset = sub_dataset.dropna(axis=1)
    else:
        sub_dataset = sub_dataset.dropna(axis=0)

    # *******************
    # demension reduction
    # *******************
    # Removing features with low variance
    # threshold = 0.05
    data_fea_var = sub_dataset.iloc[:, 1:].var()
    del_fea1 = list(data_fea_var[data_fea_var <= 0.05].index)
    sub_dataset.drop(columns=del_fea1, inplace=True)

    # pair correlations
    # threshold = 0.95
    data_fea_corr = sub_dataset.iloc[:, 1:].corr()
    del_fea2_col = []
    del_fea2_ind = []
    length = data_fea_corr.shape[1]
    for i in range(length):
        for j in range(i + 1, length):
            if abs(data_fea_corr.iloc[i, j]) >= 0.95:
                del_fea2_col.append(data_fea_corr.columns[i])
                del_fea2_ind.append(data_fea_corr.index[j])
    sub_dataset.drop(columns=del_fea2_ind, inplace=True)

    # standardize the features
    cols_ = list(sub_dataset.columns)[1:]
    sub_dataset[cols_] = sub_dataset[cols_].apply(standardize, axis=0)

    # for split in range(1, splits+1):
    pool = multiprocessing.Pool(num_pools)
    res = pool.starmap(best_model_runing, zip(range(1, repetitions + 1)))
    pool.close()
    pool.join()
    for item in res:
        for i in range(3):
            pd_res.append(item[i])
if task_type == 'cla':
    stat_res = pd.DataFrame(pd_res, columns=['split', 'dataset', 'subtask', 'set',
                                             'num_of_retained_feature',
                                             'num_of_compounds', 'postives',
                                             'negtives', 'negtives/postives',
                                             'tn', 'fp', 'fn', 'tp', 'se', 'sp',
                                             'acc', 'mcc', 'auc_prc', 'auc_roc'])
else:
    stat_res = pd.DataFrame(pd_res, columns=['split', 'dataset', 'subtask', 'set',
                                             'num_of_retained_feature',
                                             'num_of_compounds', 'rmse', 'r2', 'mae'])
stat_res.to_csv('./stat_res/' + dataset_label + '_rf_statistical_results_split50_20200622.csv', index=0)
# single tasks
if len(tasks) == 1:
    args = {'data_label': dataset_label, 'metric': 'auc_roc' if task_type == 'cla' else 'rmse', 'model': 'RF'}
    print('{}_{}: the mean {} for the training set is {:.3f} with std {:.3f}'.format(args['data_label'], args['model'],
                                                                                     args['metric'], np.mean(
            stat_res[stat_res['set'] == 'tr'][args['metric']]), np.std(
            stat_res[stat_res['set'] == 'tr'][args['metric']])))
    print(
        '{}_{}: the mean {} for the validation set is {:.3f} with std {:.3f}'.format(args['data_label'], args['model'],
                                                                                     args['metric'], np.mean(
                stat_res[stat_res['set'] == 'va'][args['metric']]), np.std(
                stat_res[stat_res['set'] == 'va'][args['metric']])))
    print('{}_{}: the mean {} for the test set is {:.3f} with std {:.3f}'.format(args['data_label'], args['model'],
                                                                                 args['metric'], np.mean(
            stat_res[stat_res['set'] == 'te'][args['metric']]), np.std(
            stat_res[stat_res['set'] == 'te'][args['metric']])))
# multi-tasks
else:
    args = {'data_label': dataset_label, 'metric': 'auc_roc' if dataset_label != 'muv' else 'auc_prc', 'model': 'RF'}
    tr_acc = np.zeros(repetitions)
    va_acc = np.zeros(repetitions)
    te_acc = np.zeros(repetitions)
    for subtask in tasks:
        tr = stat_res[stat_res['set'] == 'tr']
        tr_acc = tr_acc + tr[tr['subtask'] == subtask][args['metric']].values

        va = stat_res[stat_res['set'] == 'va']
        va_acc = va_acc + va[va['subtask'] == subtask][args['metric']].values

        te = stat_res[stat_res['set'] == 'te']
        te_acc = te_acc + te[te['subtask'] == subtask][args['metric']].values
    tr_acc = tr_acc / len(tasks)
    va_acc = va_acc / len(tasks)
    te_acc = te_acc / len(tasks)
    print('{}_{}: the mean {} for the training set is {:.3f} with std {:.3f}'.format(args['data_label'], args['model'],
                                                                                     args['metric'], np.mean(tr_acc),
                                                                                     np.std(tr_acc)))
    print(
        '{}_{}: the mean {} for the validation set is {:.3f} with std {:.3f}'.format(args['data_label'], args['model'],
                                                                                     args['metric'], np.mean(va_acc),
                                                                                     np.std(va_acc)))
    print('{}_{}: the mean {} for the test set is {:.3f} with std {:.3f}'.format(args['data_label'], args['model'],
                                                                                 args['metric'], np.mean(te_acc),
                                                                                 np.std(te_acc)))
end = time.time()  # get the end time
print('the elapsed time is:', (end - start), 'S')
