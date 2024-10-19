"""
Main script to run the recalibration experiments
"""
# Author: Alessandro Brusaferri
# License: Apache-2.0 license

import os
import pandas as pd
import numpy as np
os.environ["TF_USE_LEGACY_KERAS"]="1"
from tools.PrTSF_Recalib_tools import PrTsfRecalibEngine, load_data_model_configs
from tools.prediction_quantiles_tools import plot_quantiles
from tools.conformal_prediction import compute_cp
from tools.conformal_prediction import compute_weighted_cp

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
# Set PEPF task to execute
PF_task_name = 'EM_price'
# Set Model setup to execute
exper_setup = 'point-XGB'

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
test_predictions3 = PrTSF_eng.run_recalibration(model_hyperparams=model_hyperparams,
                                               plot_history=plot_train_history,
                                               plot_weights=plot_weights)
print(test_predictions3.shape)
print(test_predictions3.head())


#--------------------------------------------------------------------------------------------------------------------
# Conformal prediction settings
exec_CP = True 
# set the size of the calibration set sufficiently large to cover the target alpha (tails)
cp_settings={'target_alpha':[0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]}
num_cali_samples = 365
#num_cali_samples = 730
#cp_settings={'target_alpha':[0.10]}
#num_cali_samples = 31

if exec_CP:
    if exper_setup[:5]=='point':
        # build the settings to build PF from point using CP
        cp_settings['pred_horiz']=configs['data_config'].pred_horiz
        cp_settings['task_name']=configs['data_config'].task_name
        cp_settings['num_cali_samples']=num_cali_samples
        # exec conformal prediction
        print(cp_settings)
        test_predictions4 = compute_cp(test_predictions3,cp_settings)
        test_predictions5 = compute_weighted_cp(test_predictions3,cp_settings)
    else:
        print('conformal prediction implemented on point predictions')

#--------------------------------------------------------------------------------------------------------------------
# Plot test predictions
plot_quantiles(test_predictions4, target=PF_task_name)
plot_quantiles(test_predictions5, target=PF_task_name)

#--------------------------------------------------------------------------------------------------------------------
# Compute pinball score
#quantiles_levels = PrTSF_eng.model_configs['target_quantiles'] #vecchio 
quantiles_levels = PrTSF_eng.__build_target_quantiles__(cp_settings['target_alpha'])
pred_steps = configs['model_config']['pred_horiz']

pinball_scores1 = compute_pinball_scores(y_true=test_predictions4[PF_task_name].to_numpy().reshape(-1,pred_steps),
                                        pred_quantiles=test_predictions4.loc[:,test_predictions4.columns != PF_task_name].
                                        to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                                        quantiles_levels=quantiles_levels)


pinball_scores2 = compute_pinball_scores(y_true=test_predictions5[PF_task_name].to_numpy().reshape(-1,pred_steps),
                                        pred_quantiles=test_predictions5.loc[:,test_predictions5.columns != PF_task_name].
                                        to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                                        quantiles_levels=quantiles_levels)

#print(pinball_scores1, pinball_scores2)

#--------------------------------------------------------------------------------------------------------------------
print('Done!')


##IMPLEMENTED FUNCTIONS
from cProfile import label
from pickle import TRUE
from matplotlib.pyplot import plot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Flag è stringa per dire che metrica usare
#average è un boleano, dice se calcolare hourly o mean, di default calcola hourly
def compute_metrics(predicted, true, flag, average=False):
    if average:    
        if flag == 'RMSE':
            error=  np.sqrt(np.mean(np.square(true - predicted)))    
        elif flag == 'MAE':
            error=np.mean((np.abs(true-predicted)))
        elif flag == 'sMAPE':
            error= 100*2*np.mean(np.abs(np.abs(true-predicted))/(np.abs(true)+np.abs(predicted)))
        else:
            print("Not a valid metric")
            return 0
    else:
        error=np.zeros(24)   
        idx2=predicted.index            
        if flag == 'RMSE':
            for i in np.arange(24):
                indices= idx2.hour==idx2[i].hour
                error[i]=  np.sqrt(np.mean(np.square(true[indices] - predicted[indices])))    
        elif flag == 'MAE':
            for i in np.arange(24):
                indices= idx2.hour==idx2[i].hour
                error[i]= np.mean((np.abs(true[indices]-predicted[indices])))
        elif flag == 'sMAPE':
            for i in np.arange(24):
                indices= idx2.hour==idx2[i].hour
                error[i]= 100*2*np.mean(np.abs(true[indices]-predicted[indices])/(np.abs(true[indices])+np.abs(predicted[indices])))
        else:
            print("Not a valid metric")
            return error 
    return error
  

def full_evaluation(predicted, true):
    metrics={"RMSE", "MAE", "sMAPE" }
    hours = [f"{hour:02}:00" for hour in range(24)]
    # Add an extra 'Average' row
    hours.append('Average')
    # Initialize the DataFrame with zeros
    full_evaluation=pd.DataFrame(0, index=hours, columns=['RMSE', 'MAE', 'sMAPE'])
    for st in metrics:
        full_evaluation.loc[hours[:-1],st]=compute_metrics(true, predicted, st)
        full_evaluation.loc[hours[-1],st]=compute_metrics(true, predicted, st, TRUE)

    fig1, ax1 = plt.subplots()
    idx=hours[:-1]
    ones_vec=np.ones(24)
    for st in metrics:
        ax1.plot(idx,full_evaluation.loc[idx,st], linestyle="-", linewidth=0.9, label=st)
        ax1.plot(idx, ones_vec*full_evaluation.loc['Average',st], linestyle="-", linewidth=0.9, label=st+"_average")
    tics=hours[0:24:3]
    ax1.set_xticks(tics)
    #fig1.xticks(ticks=tics)
    #fig1.gca().xaxis.set_ticks([tick for tick in ax1.gca().xaxis.get_ticks() if tick in tics])

    ax1.grid()
    ax1.legend()
    ax1.set_title("Error Metrics")
    fig1.show()
    return full_evaluation

def compute_delta_cov(y_true, pred_quantiles, quantiles_levels):
    """
    Utility function to compute the delta coverage on the test results
    return: delta coverage computed for each quantile level and each step in the pred horizon
    """
  
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
        #print(idx_up, idx_down)
        
        EC_alpha = np.mean(idx_up & idx_down)#quale asse



        #score.append(delta_cov)
        EC.append(np.abs(EC_alpha-(1-q*2)))#check
    

    
    score = 1/(2*(useful_quantiles[-1]-useful_quantiles[0]))*np.sum(EC)

    return score






delta_cov1 = compute_delta_cov(y_true=test_predictions4[PF_task_name].to_numpy().reshape(-1,pred_steps),
                                        pred_quantiles=test_predictions4.loc[:,test_predictions4.columns != PF_task_name].
                                        to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                                        quantiles_levels=quantiles_levels)
print(delta_cov1)
delta_cov2 = compute_delta_cov(y_true=test_predictions5[PF_task_name].to_numpy().reshape(-1,pred_steps),
                                        pred_quantiles=test_predictions5.loc[:,test_predictions5.columns != PF_task_name].
                                        to_numpy().reshape(-1, pred_steps, len(quantiles_levels)),
                                        quantiles_levels=quantiles_levels)
print(delta_cov1,delta_cov2)

full_metrics=full_evaluation(test_predictions4[0.5], test_predictions4[PF_task_name])