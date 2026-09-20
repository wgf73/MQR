import os
import logging
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import warnings
import dill
from collections import Counter
from rdkit import Chem
from collections import defaultdict
import torch
import random
from scipy.stats import linregress
warnings.filterwarnings('ignore')

def get_model_path(log_directory_path, log_dir_prefix):
    """
    Given a path to a directory containing log files, return the path to the most recent model file.

    Parameters:
    log_directory_path (str): Path to the directory containing log files.
    log_dir_prefix (str): Prefix of the log directory containing the desired model file.

    Returns:
    str: Path to the most recent model file.
    """

    # Get a list of all logs in the directory
    log_files = os.listdir(log_directory_path)

    # Select the log directory that starts with the specified prefix
    log_dir_prefix += '_'
            avg_pop = medicine_pop[out_list].mean()
            pop.append(avg_pop)
        return np.mean(pop)

    medicine_pop = pd.read_csv(medicine_pop_path, header=None, index_col=0)  
    return ja, np.mean(avg_prc), np.mean(avg_recall), avg_pop


def ddi_rate_score(record, path=None):
    N_fingerprint = len(fingerprint_dict)
    # transform into projection matrix
    n_col = sum(average_index)
    n_row = len(average_index)

    average_projection = np.zeros((n_row, n_col))
    col_counter = 0
    for i, item in enumerate(average_index):
        if item > 0:
            average_projection[i, col_counter : col_counter + item] = 1 / item
        col_counter += item

    return MPNNSet, N_fingerprint, torch.FloatTensor(average_projection)

# COGNet
def sequence_metric_v2(y_gt, y_pred, y_label):
    def average_prc(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b]==1)[0]
            out_list = y_label[b]
            inter = list(set(out_list) & set(target))
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score


    def average_recall(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = list(set(out_list) & set(target))
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score


    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if (average_prc[idx] + average_recall[idx]) == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score


    def jaccard(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = list(set(out_list) & set(target))
            union = list(set(out_list) | set(target))
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_pred_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_pred_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob_label, k):
        precision = 0
        for i in range(len(y_gt)):
            TP = 0
            for j in y_prob_label[i][:k]:
                if y_gt[i, j] == 1:
                    TP += 1
            precision += TP / k
        return precision / len(y_gt)
    # try:
    #     auc = roc_auc(y_gt, y_prob)
    # except ValueError:
    #     auc = 0
    # p_1 = precision_at_k(y_gt, y_label, k=1)
    # p_3 = precision_at_k(y_gt, y_label, k=3)
    # p_5 = precision_at_k(y_gt, y_label, k=5)
    f1 = f1(y_gt, y_pred)
    # prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def output_flatten(labels, logits, seq_length, m_length_matrix, med_num, END_TOKEN, device, training=True, testing=False, max_len=20):
    '''
    labels: [batch_size, visit_num, medication_num]
    logits: [batch_size, visit_num, max_med_num, medication_vocab_size]
    '''
    # 将最终多个维度的结果展开
    batch_size, max_seq_length = labels.size()[:2]
    assert max_seq_length == max(seq_length)
    whole_seqs_num = seq_length.sum().item()
    if training:
        whole_med_sum = sum([sum(buf) for buf in m_length_matrix]) + whole_seqs_num # 因为每一个seq后面会多一个END_TOKEN

        # 将结果展开，然后用库函数进行计算
        labels_flatten = torch.empty(whole_med_sum).to(device)
        logits_flatten = torch.empty(whole_med_sum, med_num).to(device)

        start_idx = 0
        for i in range(batch_size): # 每个batch
            for j in range(seq_length[i]):  # seq_length[i]指这个batch对应的seq数目
                for k in range(m_length_matrix[i][j]+1):  # m_length_matrix[i][j]对应seq中med的数目
                    if k==m_length_matrix[i][j]:    # 最后一个label指定为END_TOKEN
                        labels_flatten[start_idx] = END_TOKEN
                    else:
                        labels_flatten[start_idx] = labels[i, j, k]
                    logits_flatten[start_idx, :] = logits[i, j, k, :]
                    start_idx += 1
        return labels_flatten, logits_flatten
    else:
        # 将结果按照adm展开，然后用库函数进行计算
        labels_flatten = []
        logits_flatten = []

        start_idx = 0
        for i in range(batch_size): # 每个batch
            for j in range(seq_length[i]):  # seq_length[i]指这个batch对应的seq数目
                labels_flatten.append(labels[i,j,:m_length_matrix[i][j]].detach().cpu().numpy())
                
                if testing:
                    logits_flatten.append(logits[j])  # beam search目前直接给出了预测结果
                else:
                    logits_flatten.append(logits[i,j,:max_len,:].detach().cpu().numpy())     # 注意这里手动定义了max_len
                # cur_label = []
                # cur_seq_length = []
                # for k in range(m_length_matrix[i][j]+1):  # m_length_matrix[i][j]对应seq中med的数目
                #     if k==m_length_matrix[i][j]:    # 最后一个label指定为END_TOKEN
                #         continue
                #     else:
                #         labels_flatten[start_idx] = labels[i, j, k]
                #     logits_flatten[start_idx, :] = logits[i, j, k, :med_num]
                #     start_idx += 1
        return labels_flatten, logits_flatten


def print_result(label, prediction):
    '''
    label: [real_med_num, ]
    logits: [20, med_vocab_size]
    '''
    label_text = " ".join([str(x) for x in label])
    predict_text = " ".join([str(x) for x in prediction])
    
    return "[GT]\t{}\n[PR]\t{}\n\n".format(label_text, predict_text)


# MoleRec
def buildPrjSmiles(molecule, med_voc, device="cpu:0"):

    average_index, smiles_all = [], []

    print(len(med_voc.items()))  # 131
    for index, ndc in med_voc.items():

        smilesList = list(molecule[ndc])

        """Create each data with the above defined functions."""
        counter = 0  # counter how many drugs are under that ATC-3
        for smiles in smilesList:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                smiles_all.append(smiles)
                counter += 1
            else:
                print('[SMILES]', smiles)
                print('[Error] Invalid smiles')
        average_index.append(counter)

        """Transform the above each data of numpy
        to pytorch tensor on a device (i.e., CPU or GPU).
        """
    # transform into projection matrix
    n_col = sum(average_index)
    n_row = len(average_index)

    average_projection = np.zeros((n_row, n_col))
    col_counter = 0
    for i, item in enumerate(average_index):
        average_projection[i, col_counter: col_counter + item] = 1 / item
        col_counter += item

    print("Smiles Num:{}".format(len(smiles_all)))
    print("n_col:{}".format(n_col))
    print("n_row:{}".format(n_row))

    return torch.FloatTensor(average_projection), smiles_all