"""
Main script to run the recalibration experiments
"""


import os
import pandas as pd
import numpy as np
os.environ["TF_USE_LEGACY_KERAS"]="1"
from tools.PrTSF_Recalib_tools import PrTsfRecalibEngine, load_data_model_configs
from tools.prediction_quantiles_tools import plot_quantiles

#--------------------------------------------------------------------------------------------------------------------
def compute_pinball_scores(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the pinball score on the test results
    return: pinball scores computed for each quantile level and each step in the pred horizon
    """
    score = []
    for i, q in enumerate(quantiles_levels):
        error = np.subtract(y_true, pred_quantiles[:, :, i])
        loss_q = np.maximum(q * error, (q - 1) * error)
        score.append(np.expand_dims(loss_q,-1))
    score = np.mean(np.concatenate(score, axis=-1), axis=0)
    return score
#--------------------------------------------------------------------------------------------------------------------
def compute_winkler_scores(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the Winkler score on the test results
    return: Winkler scores computed for each quantile level and each step in the pred horizon
    """
    score = []
    for i, q in enumerate(quantiles_levels):
        if q >= 0.5:
            break
        Upper = (pred_quantiles[:, :, -i - 1])
        Lower = (pred_quantiles[:, :, i])

        delta_n = Upper - Lower
        idx_up = np.greater(y_true, Upper)
        idx_down = np.greater(Lower, y_true)
        winkler = (idx_down * (delta_n + 2 / (1 - q * 2) * (Lower - y_true)) + idx_up * (
                delta_n + 2 / (1 - q * 2) * (y_true - Upper)) + (~(idx_up | idx_down)) * delta_n)

        loss_q = np.mean(winkler, axis=0)
        score.append(loss_q)
    return score


def compute_delta_cov(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the delta coverage on the test results
    return: delta coverage computed for each quantile level and each step in the pred horizon
    """
    print(quantiles_levels)
    score = []
    EC = []
    #quantile levels must have symmetric quantiles and also the 0.5 quantile
    useful_quantiles = quantiles_levels[-(np.size(quantiles_levels)-1)/2:]
    print(useful_quantiles)
    for i, q in enumerate(quantiles_levels):
        if q >= 0.5:
            break       
        Upper = (pred_quantiles[:, :, -i - 1])
        Lower = (pred_quantiles[:, :, i])

        idx_up = np.greater( Upper, y_true)
        idx_down = np.greater( y_true, Lower)
        EC_alpha = np.mean(idx_up & idx_down, axis=0)#quale asse
        #score.append(delta_cov)
        EC.append(np.abs(EC_alpha-1+q))#check
    
    score = 1/(useful_quantiles[-1]-useful_quantiles[0])*np.sum(EC)



    return score



#--------------------------------------------------------------------------------------------------------------------
# Set PEPF task to execute
PF_task_name = 'EM_price'
# Set Model setup to execute
#exper_setup = 'N-DNN'
exper_setup = 'N-BiTCN'

#---------------------------------------------------------------------------------------------------------------------
# Set run configs
run_id = 'recalib_opt_grid_1_1'
# Load hyperparams from file (select: load_tuned or optuna_tuner)
hyper_mode = 'load_tuned'
# Plot train history flag
plot_train_history=False
plot_weights=False

#---------------------------------------------------------------------------------------------------------------------
# Load experiments configuration from json file
configs=load_data_model_configs(task_name=PF_task_name, exper_setup=exper_setup, run_id=run_id)

# Load dataset
dir_path = os.getcwd()
ds = pd.read_csv(os.path.join(dir_path, 'data', 'datasets', configs['data_config'].dataset_name))
ds.set_index(ds.columns[0], inplace=True)

#---------------------------------------------------------------------------------------------------------------------
# Instantiate recalibratione engine
PrTSF_eng = PrTsfRecalibEngine(dataset=ds,
                               data_configs=configs['data_config'],
                               model_configs=configs['model_config'])

# Get model hyperparameters (previously saved or by tuning)
model_hyperparams = PrTSF_eng.get_model_hyperparams(method=hyper_mode, optuna_m=configs['model_config']['optuna_m'])

# Exec recalib loop over the test_set samples, using the tuned hyperparams
test_predictions = PrTSF_eng.run_recalibration(model_hyperparams=model_hyperparams,
                                               plot_history=plot_train_history,
                                               plot_weights=plot_weights)

#--------------------------------------------------------------------------------------------------------------------
# Compute pinball score
quantiles_levels = PrTSF_eng.model_configs['target_quantiles']
pred_steps = configs['model_config']['pred_horiz']

pinball_scores = compute_pinball_scores(y_true=test_predictions[PF_task_name].to_numpy().reshape(-1,pred_steps),
                                        pred_quantiles=test_predictions.loc[:,test_predictions.columns != PF_task_name].
                                        to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                                        quantiles_levels=quantiles_levels)

#--------------------------------------------------------------------------------------------------------------------
# Plot test predictions
plot_quantiles(test_predictions, target=PF_task_name)

#--------------------------------------------------------------------------------------------------------------------
print('Done!')


def compute_delta_cov(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the delta coverage on the test results
    return: delta coverage computed for each quantile level and each step in the pred horizon
    """
    print(quantiles_levels)
    score = []
    EC = []

    #print(pred_quantiles)

    #quantile levels must have symmetric quantiles and also the 0.5 quantile
    useful_quantiles = quantiles_levels[-int((np.size(quantiles_levels)-1)/2):]
    
    print(useful_quantiles)
    for i, q in enumerate(quantiles_levels):
        if q >= 0.5:
            break       
        Upper = (pred_quantiles[:, :, -i - 1])
        Lower = (pred_quantiles[:, :, i])


        idx_up = np.greater_equal( Upper, y_true)
        idx_down = np.greater_equal( y_true, Lower)
       
        EC_alpha = np.mean(idx_up & idx_down)#quale asse

   

        #score.append(delta_cov)
        EC.append(np.abs(EC_alpha-(1-q*2)))#check
    

    
    score = 1/(2*(useful_quantiles[-1]-useful_quantiles[0]))*np.sum(EC)

    return score


delta_cov2 = compute_delta_cov(y_true=test_predictions[PF_task_name].to_numpy().reshape(-1,pred_steps),
                                        pred_quantiles=test_predictions.loc[:,test_predictions.columns != PF_task_name].
                                        to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                                        quantiles_levels=quantiles_levels)
print(delta_cov2)
