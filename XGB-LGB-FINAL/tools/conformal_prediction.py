import sys
from typing import Dict, List
import numpy as np


def build_target_quantiles(target_alpha: List):
    """
    Build target quantiles from the list of alpha, including the median
    """
    target_quantiles = [0.5]
    for alpha in target_alpha:
        target_quantiles.append(alpha / 2)
        target_quantiles.append(1 - alpha / 2)
    target_quantiles.sort()
    return target_quantiles


def build_cp_pis(preds_cali: np.array, y_cali: np.array, preds_test: np.array,
                 settings: Dict, method: str='higher'):
    """
    Compute PIs at the different alpha levels using conformal prediction
    """
    preds_cali = np.squeeze(preds_cali, axis=-1)
    if preds_test.shape[0]>1:
        sys.exit('ERROR: exec_cup supports single test samples')
    # Compute conformity score (absolute residual)
    conf_score = np.abs(preds_cali - y_cali)
    #print(conf_score)

    n=conf_score.shape[0]



    # Stack the quantiles to the point pred for each alpha
    preds_test_q=[preds_test]
    for alpha in settings['target_alpha']:
        q = np.ceil((n + 1) * (1 - alpha)) / n
        
        Q_1_alpha= np.expand_dims(np.quantile(a=conf_score, q=q, axis=0, method=method),
                                  axis=(0,-1))
        # Append lower/upper PIs for the current alpha
        
        preds_test_q.append(preds_test - Q_1_alpha)
        preds_test_q.append(preds_test + Q_1_alpha)
    preds_test_q = np.concatenate(preds_test_q, axis=2)
    # Fix quantile crossing by sorting and return prediction flattened in temporal dimension (sample over pred horizon)
    return np.sort(preds_test_q.reshape(-1, preds_test_q.shape[-1]), axis=-1)

def build_cp_pis_weigthed(preds_cali: np.array, y_cali: np.array, preds_test: np.array,
                 settings: Dict, method: str='higher'):
    """
    Compute PIs at the different alpha levels using conformal prediction
    """
    preds_cali = np.squeeze(preds_cali, axis=-1)
    if preds_test.shape[0]>1:
        sys.exit('ERROR: exec_cup supports single test samples')
    # Compute conformity score (absolute residual)
    conf_score = np.abs(preds_cali - y_cali)
    #print(conf_score)
    n=conf_score.shape[0]
    # Stack the quantiles to the point pred for each alpha
    preds_test_q=[preds_test]
    decay=0.99
    num_cali=preds_cali.size
    #print("Num cali:", num_cali, "n:", n)
    #print("controllare se l'ordine dei pesi è giusto o inverso")
    decay_weights = np.power(decay, np.arange(n))[::-1]
    

    decay_weights = np.tile(decay_weights, (conf_score.shape[1], 1)).T



    sorted_indices = np.argsort(conf_score, axis=0)
    #sorted_values = conf_score[sorted_indices]
    sorted_values = np.empty_like(conf_score)
    sorted_weights = np.empty_like(decay_weights)
    for i in range(conf_score.shape[1]):
        sorted_values[:, i] = conf_score[sorted_indices[:, i], i]
        sorted_weights[:, i] = decay_weights[sorted_indices[:, i], i]
    
    #sorted_weights = decay_weights[sorted_indices]

    
    # Compute the cumulative sum of the weights
    cumulative_weights = np.cumsum(sorted_weights, axis=0)

    
    # Normalize the cumulative weights by the total weight
    total_weight = cumulative_weights[-1, :]
    #print("Total weight:", total_weight)
   
    normalized_cumulative_weights = cumulative_weights / total_weight

    
    # Find the index where the normalized cumulative weight crosses alpha
   
    #quantile_index = np.searchsorted(normalized_cumulative_weights, alpha, side='right')
    for alpha in settings['target_alpha']:
        print("Alpha:", alpha)
        q = np.ceil((n + 1) * (1 - alpha)) / n
        #print("check se ci va effettivamente 1-alpha etc etc")
        #print("Q:", q)
        #quantile_index = np.searchsorted(normalized_cumulative_weights ,q , side='right')

        quantile_index = np.apply_along_axis(
            lambda a: np.searchsorted(a, q, side='right'),
            axis=0,
            arr=normalized_cumulative_weights
        )
        quantile_index = np.clip(quantile_index, 0, n-1) 



        quantile_index = quantile_index.flatten()

        Q_1_alpha= np.expand_dims(sorted_values[quantile_index, np.arange(sorted_values.shape[1])], axis=(0,-1))

        # Append lower/upper PIs for the current alpha
        #print( alpha, Q_1_alpha.shape)
        preds_test_q.append(preds_test - Q_1_alpha)
        preds_test_q.append(preds_test + Q_1_alpha)
    preds_test_q = np.concatenate(preds_test_q, axis=2)
    # Fix quantile crossing by sorting and return prediction flattened in temporal dimension (sample over pred horizon)
    return np.sort(preds_test_q.reshape(-1, preds_test_q.shape[-1]), axis=-1)



def compute_cp(recalib_preds, settings: Dict):
    """
    Reshape recalibration predictions and execute conformal prediciton for each test sample
    """
    settings['target_quantiles'] = build_target_quantiles(settings['target_alpha'])
    ens_p = recalib_preds.loc[:,0.5].to_numpy()
    ens_p_d = ens_p.reshape(-1, settings['pred_horiz'], 1)
    target_d = recalib_preds.filter([settings['task_name']], axis=1).to_numpy().reshape(-1, settings['pred_horiz'])

    num_test_samples = ens_p_d.shape[0] - settings['num_cali_samples']
    
    test_PIs=[]
    for t_s in range(num_test_samples):
        preds_cali = ens_p_d[t_s:settings['num_cali_samples'] + t_s]
        preds_test = ens_p_d[settings['num_cali_samples'] + t_s:settings['num_cali_samples'] + t_s+1]
        y_cali = target_d[t_s:settings['num_cali_samples'] + t_s]
    
        test_PIs.append(build_cp_pis(preds_cali=preds_cali,
                                     y_cali=y_cali,
                                     preds_test=preds_test,
                                     settings=settings))
    test_PIs=np.concatenate(test_PIs, axis=0)
    # Build updated dataframe
    aggr_df=recalib_preds.filter([settings['task_name']], axis=1)
    aggr_df=aggr_df.iloc[settings['pred_horiz'] * settings['num_cali_samples']:]
    for j in range(len(settings['target_quantiles'])):
        aggr_df[settings['target_quantiles'][j]]=test_PIs[:,j]
    return aggr_df

def compute_weighted_cp(recalib_preds, settings: Dict):
    """
    Reshape recalibration predictions and execute conformal prediciton for each test sample
    """
    settings['target_quantiles'] = build_target_quantiles(settings['target_alpha'])
    ens_p = recalib_preds.loc[:,0.5].to_numpy()
    ens_p_d = ens_p.reshape(-1, settings['pred_horiz'], 1)
    target_d = recalib_preds.filter([settings['task_name']], axis=1).to_numpy().reshape(-1, settings['pred_horiz'])
    num_test_samples = ens_p_d.shape[0] - settings['num_cali_samples']
    
    test_PIs=[]
    for t_s in range(num_test_samples):
        preds_cali = ens_p_d[t_s:settings['num_cali_samples'] + t_s]
        preds_test = ens_p_d[settings['num_cali_samples'] + t_s:settings['num_cali_samples'] + t_s+1]
        y_cali = target_d[t_s:settings['num_cali_samples'] + t_s]
        test_PIs.append(build_cp_pis_weigthed(preds_cali=preds_cali,
                                     y_cali=y_cali,
                                     preds_test=preds_test,
                                     settings=settings))
    test_PIs=np.concatenate(test_PIs, axis=0)
    # Build updated dataframe
    aggr_df=recalib_preds.filter([settings['task_name']], axis=1)
    aggr_df=aggr_df.iloc[settings['pred_horiz'] * settings['num_cali_samples']:]
    for j in range(len(settings['target_quantiles'])):
        aggr_df[settings['target_quantiles'][j]]=test_PIs[:,j]
    return aggr_df
