"""
DNN model class
"""

# Author: 
# License: Apache-2.0 license

from tools.data_utils import features_keys
import numpy as np
import tensorflow as tf
#from tensorflow import keras as tfk
#from tensorflow.keras import layers as tfkl

from typing import List
import sys
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import os
import shutil
import matplotlib.pyplot as plt


class LSTMRegressor:
    def __init__(self, settings, loss):
        self.settings = settings
        self.__build_model__(loss)

    def __build_model__(self, loss):

        #print(self.settings)
        x_in = tf.keras.layers.Input(shape=(self.settings['input_size'],1,))#1
        x_in = tf.keras.layers.BatchNormalization()(x_in)#1

        x_in1 = tf.keras.layers.Lambda(lambda x: x[:, :-78, :])(x_in)
        x_in2 = tf.keras.layers.Lambda(lambda x: x[:, -78:, :])(x_in)
        x_in1 = tf.keras.layers.BatchNormalization()(x_in1)
        x_in2 = tf.keras.layers.BatchNormalization()(x_in2)
        x_in1 = tf.keras.layers.GaussianNoise(0.1)(x_in1)
        x = tf.keras.layers.LSTM(160, activation='tanh')(x_in1)
        x_in2 = tf.keras.layers.Flatten()(x_in2)
        x2= tf.keras.layers.Dense(256, activation='gelu')(x_in2)
        x2 = tf.keras.layers.Dropout(0.15)(x2)
        x = tf.keras.layers.Dropout(0.15)(x)
        x = tf.keras.layers.Concatenate()([x, x2])




        print((self.settings['input_size'],1))
        
        #(self.settings['input_size'],) è la input_shape
        #x= tf.keras.layers.Masking(mask_value = 0, input_shape = (self.settings['input_size'],))(x_in)
         # Add a Bidirectional LSTM layer with 64 units
        
        #x = tf.keras.layers.GRU(64, activation="relu", return_sequences=True, name= 'gru', kernel_regularizer= tf.keras.regularizers.L2(0.001),recurrent_regularizer=tf.keras.regularizers.L2(0.001))(x_in)
        #x = tf.keras.layers.Dropout(0.1)(x)
        #x = tf.keras.layers.GRU(128, activation="relu", return_sequences=True, name= 'gru2', kernel_regularizer= tf.keras.regularizers.L2(0.001),recurrent_regularizer=tf.keras.regularizers.L2(0.001))(x)
        
        
        
        #x = tf.keras.layers.LSTM(320, activation='tanh')(x_in)
        #x = tf.keras.layers.Dropout(0.15)(x)
        
        #LSTM ultimo layer che usavo


        


        #x = tf.keras.layers.SimpleRNN(128, activation="tanh", name= 'gru1')(x_in1)
       
        #x = tf.keras.layers.Reshape((128,1))(x)
        #x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False, name='lstm'), name='bidirectional_lstm')(x_in1)
        
        #x = tf.keras.layers.LSTM(64, activation='tanh', return_sequences=True,kernel_regularizer= tf.keras.regularizers.L2(0.001),recurrent_regularizer=tf.keras.regularizers.L2(0.001))(x_in1)
        n_lstm=128
        # x = tf.keras.layers.SimpleRNN(n_lstm, activation='gelu', return_sequences=False)(x)
        # x = tf.keras.layers.Dropout(0.15)(x)
        # #x_in2 = tf.keras.layers.Flatten()(x_in2)
        # x_in2 = tf.keras.layers.Conv1D(128,3, activation='gelu')(x_in2)
        # #x_in2 = tf.keras.layers.MaxPooling1D(2)(x_in2)
        # x_in2 = tf.keras.layers.SimpleRNN(128, activation='gelu', return_sequences=False)(x_in2)
        # #x_in2 = tf.keras.layers.Flatten()(x_in2)
        

        ##opzione 1
        #x = tf.keras.layers.Reshape((n_lstm+78, 1))(x)
        # Add a 1D Convolution layer with 128 filters and a kernel size of 3
        #x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu', name='conv')(x)
        #x = tf.keras.layers.Dropout(0.1)(x)
        # Add a final Convolution layer to match the desired output shape
        #lo metto piccolo, prima aveva un dense di 512, così è 12x246
        #x = tf.keras.layers.Conv1D(1, 3, padding='same', name='output_layer')(x)
        #   PROVARE A USARE SOLO FLATTEN DOPO IL CONVOLUZIONALE, SENZA IL DENSE O CONV DA 1
        #x = tf.keras.layers.Dense(1, activation="sigmoid", name='output_layer')(x) 
        #x = tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_regularizer=tf.keras.regularizers.l2(2e-5))(x)
        #x = tf.keras.layers.Reshape((1,x.shape[2]*x.shape[1]))(x)
        #x = tf.keras.layers.Flatten()(x)
        #print("prova:", x.shape[2]*x.shape[1])

        ##opzione 2
        #x = tf.keras.layers.GaussianNoise(0.1)(x)
        #x = tf.keras.layers.Dense(256, activation='leaky_relu')(x)
        #prima era 128
        #x = tf.keras.layers.Dropout(0.15)(x)

        if self.settings['PF_method'] == 'point':
            out_size = 1
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                          activation='linear',
                                          )(x)
            output = tf.keras.layers.Reshape((self.settings['pred_horiz'], 1))(logit)

        elif self.settings['PF_method'] == 'qr':
            out_size = len(self.settings['target_quantiles'])
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                            activation='linear',
                                            )(x)
            output = tf.keras.layers.Reshape((self.settings['pred_horiz'], out_size))(logit)
            output = tf.keras.layers.Lambda(lambda x: tf.sort(x, axis=-1))(output)
        
        elif self.settings['PF_method'] == 'Normal':
            out_size = 2
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                            activation='linear',
                                            )(x)
            output = tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(
                    loc=t[..., :self.settings['pred_horiz']],
                    scale=1e-3 + 3 * tf.math.softplus(0.05 * t[..., self.settings['pred_horiz']:])))(logit)
        
        elif self.settings['PF_method'] == 'JSU':  ##da controllare
            out_size = 4 #qui sarà 4 per me
            #x = tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_regularizer=tf.keras.regularizers.l2(2e-5))(x)
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                            activation='sigmoid',#qui era linear
                                            kernel_regularizer=tf.keras.regularizers.l2(2e-5),
                                            )(x)
            output = tfp.layers.DistributionLambda(
                lambda t: tfd.JohnsonSU(
                    skewness= t[..., self.settings['pred_horiz']*3:],
                    tailweight=1 + 3 * tf.math.softplus(0.05 * t[..., self.settings['pred_horiz']*2:self.settings['pred_horiz']*3 ]),
                    loc=t[..., :self.settings['pred_horiz']],
                    scale=1e-3 + 3 * tf.math.softplus(0.05 * t[..., self.settings['pred_horiz']:self.settings['pred_horiz']*2 ])))(logit)
            
        elif self.settings['PF_method'] == 't-student':
            out_size = 3
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size, activation='linear')(x)
            output = tfp.layers.DistributionLambda(
                lambda t: tfd.StudentT(
                    df=1 + tf.math.softplus(0.05 * t[..., self.settings['pred_horiz'] * 2:self.settings['pred_horiz'] * 3]),
                    loc=t[..., :self.settings['pred_horiz']],
                    scale=1e-3 + 3 * tf.math.softplus(0.05 * t[..., self.settings['pred_horiz']:self.settings['pred_horiz'] * 2])
                )
            )(logit)

       
        else:
            sys.exit('ERROR: unknown PF_method config!')

        # Create model
        self.model= tf.keras.Model(inputs=[x_in], outputs=[output])
        # Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.settings['lr']),
                           loss=loss)
        
        self.model.summary()

    def fit(self, train_x, train_y, val_x, val_y, verbose=0, pruning_call=None):
        # Convert the data into the input format using the internal converter
        train_x = self.build_model_input_from_series(x=train_x,
                                                     col_names=self.settings['x_columns_names'],
                                                     pred_horiz=self.settings['pred_horiz'])
        val_x = self.build_model_input_from_series(x=val_x,
                                                   col_names=self.settings['x_columns_names'],
                                                   pred_horiz=self.settings['pred_horiz'])
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                              patience=self.settings['patience'],
                                              restore_best_weights=False)

        # Create folder to temporally store checkpoints
        checkpoint_path = os.path.join(os.getcwd(), 'tmp_checkpoints', 'cp.weights.h5')
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                monitor="val_loss", mode="min",
                                                save_best_only=True,
                                                save_weights_only=True, verbose=0)
        if pruning_call==None:
            callbacks = [es, cp]
        else:
            callbacks = [es, cp, pruning_call]

        print("Per ricordare dove è il verbose")

        history = self.model.fit(train_x,
                                 train_y,
                                 validation_data=(val_x, val_y),
                                 epochs=self.settings['max_epochs'],
                                 batch_size=self.settings['batch_size'],
                                 callbacks=callbacks,
                                 verbose=1)#verbose=verbose) #qui era verbose
        # Load best weights: do not use restore_best_weights from early stop since works only in case it stops training
        self.model.load_weights(checkpoint_path)
        # delete temporary folder
        shutil.rmtree(checkpoint_dir)
        return history

    def predict(self, x):
        x = self.build_model_input_from_series(x=x,
                                               col_names=self.settings['x_columns_names'],
                                               pred_horiz=self.settings['pred_horiz'])
        return self.model(x)

    def evaluate(self, x, y):
        x = self.build_model_input_from_series(x=x,
                                               col_names=self.settings['x_columns_names'],
                                               pred_horiz=self.settings['pred_horiz'])
        return self.model.evaluate(x=x, y=y)

    @staticmethod
    def build_model_input_from_series(x, col_names: List, pred_horiz: int):
        # get index of target and past features
        past_col_idxs = [index for (index, item) in enumerate(col_names)
                         if features_keys['target'] in item or features_keys['past'] in item]

        # get index of const features
        const_col_idxs = [index for (index, item) in enumerate(col_names)
                          if features_keys['const'] in item]

        # get index of futu features
        futu_col_idxs = [index for (index, item) in enumerate(col_names)
                         if features_keys['futu'] in item]

        # build conditioning variables for past features
        past_feat = [x[:, :-pred_horiz, feat_idx] for feat_idx in past_col_idxs]
        # build conditioning variables for futu features
        futu_feat = [x[:, -pred_horiz:, feat_idx] for feat_idx in futu_col_idxs]
        # build conditioning variables for cal features
        c_feat = [x[:, -pred_horiz:-pred_horiz + 1, feat_idx] for feat_idx in const_col_idxs]

        # return flattened input
        return np.concatenate(past_feat + futu_feat + c_feat, axis=1)

    @staticmethod
    def get_hyperparams_trial(trial, settings):
        settings['hidden_size'] = trial.suggest_int('hidden_size', 64, 960, step=64)
        settings['n_hidden_layers'] = 2  # trial.suggest_int('n_hidden_layers', 1, 3)
        settings['lr'] = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        settings['activation'] = 'softplus'
        return settings

    @staticmethod
    def get_hyperparams_searchspace():
        return {'hidden_size': [128, 512],
                'lr': [1e-4, 1e-3]}

    @staticmethod
    def get_hyperparams_dict_from_configs(configs):
        model_hyperparams = {
            'hidden_size': configs['hidden_size'],
            'n_hidden_layers': configs['n_hidden_layers'],
            'lr': configs['lr'],
            'activation': configs['activation']
        }
        return model_hyperparams

    def plot_weights(self):
        w_b = self.model.layers[1].get_weights()
        plt.imshow(w_b[0].T)
        plt.title('DNN input weights')
        plt.show()
