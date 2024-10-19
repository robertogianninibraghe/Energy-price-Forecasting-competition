# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import tensorflow as tf
# import tensorflow_probability as tfp
# from tensorflow_probability import distributions as tfd
# import numpy as np
# import os
# import shutil
# import matplotlib.pyplot as plt
# from typing import List
# from tools.data_utils import features_keys


# # class deepar(nn.Module):
# #     def __init__(self, d_lag, d_cov, d_output, d_hidden, dropout, N):
# #         super(deepar, self).__init__()
# #         lstm = [nn.LSTM(d_lag + d_cov, d_hidden)]
# #         for i in range(N - 1):
# #             lstm += [nn.LSTM(d_hidden, d_hidden)]
# #         self.lstm = nn.ModuleList(lstm)
# #         self.drop = nn.ModuleList([nn.Dropout(dropout) for _ in range(N)])
# #         self.loc = nn.Linear(d_hidden * N, d_output)
# #         self.scale = nn.Linear(d_hidden * N, d_output)
# #         self.epsilon = 1e-6

# #     def forward(self, x_lag, x_cov, d_outputseqlen):
# #         inputs = torch.cat((x_lag, x_cov), dim=-1)
# #         h = []
# #         for i, layer in enumerate(self.lstm):
# #             outputs, _ = layer(inputs)
# #             outputs = self.drop[i](outputs)
# #             inputs = outputs
# #             h.append(outputs)
# #         h = torch.cat(h, -1)
# #         loc = self.loc(h[-d_outputseqlen:])
# #         scale = F.softplus(self.scale(h[-d_outputseqlen:]))
# #         return loc, scale + self.epsilon

# # class TorchModelWrapper(tf.keras.layers.Layer):
# #     def __init__(self, settings):
# #         print(settings)
# #         super(TorchModelWrapper, self).__init__()
# #         #self.settings = settings
# #         self.pytorch_model = deepar(
# #             d_lag=settings['input_size']-78,
# #             d_cov=78,
# #             d_output=settings['pred_horiz'],
# #             d_hidden=settings['hidden_size'],
# #             dropout=0.1,
# #             N=settings['n_hidden_layers']
# #         )

# #     # def call(self, inputs):
# #     #     x_lag, x_cov = inputs
# #     #     #x_lag = torch.tensor(x_lag.numpy(), dtype=torch.float32)
# #     #     #x_cov = torch.tensor(x_cov.numpy(), dtype=torch.float32)
# #     #     x_lag = torch.tensor(x_lag, dtype=torch.float32)
# #     #     x_cov = torch.tensor(x_cov, dtype=torch.float32)
# #     #     loc, scale = self.pytorch_model(x_lag, x_cov, self.settings['pred_horiz'])
# #     #     loc = loc.detach().numpy()
# #     #     scale = scale.detach().numpy()
# #     #     return loc, scale

# #     def call(self, inputs):
# #         x_lag, x_cov = inputs

# #         def to_numpy(t):
# #             return t.numpy()

# #         x_lag = tf.py_function(func=to_numpy, inp=[x_lag], Tout=tf.float32)
# #         x_cov = tf.py_function(func=to_numpy, inp=[x_cov], Tout=tf.float32)

# #         def forward_fn(x_lag, x_cov):
# #             x_lag = torch.tensor(x_lag, dtype=torch.float32)
# #             x_cov = torch.tensor(x_cov, dtype=torch.float32)
# #             loc, scale = self.pytorch_model(x_lag, x_cov, self.settings['pred_horiz'])
# #             return loc.detach().numpy(), scale.detach().numpy()

# #         loc, scale = tf.py_function(func=forward_fn, inp=[x_lag, x_cov], Tout=[tf.float32, tf.float32])
# #         return loc, scale


# class DeepARRegressor:
#     def __init__(self, settings, loss):
#         self.settings = settings
#         self.__build_model__(loss)

#     def __build_model__(self, loss):
#         #x_lag = tf.keras.layers.Input(shape=(self.settings['input_size']-78,))
#         #x_cov = tf.keras.layers.Input(shape=(78,))

#         x__in = tf.keras.layers.Input(shape=(self.settings['input_size'],1,))

#         #x_lag = tf.keras.layers.Lambda(lambda x: x[:, :-78])(x_in)
#         #x_cov = tf.keras.layers.Lambda(lambda x: x[:, -78:])(x_in)
#         # if self.settings['PF_method'] == 'Normal':
#         #     torch_layer = TorchModelWrapper(self.settings)
#         #     # Assuming x_lag and x_cov are your input tensors
#         #     #x_lag_np = x_lag.numpy()
#         #     #x_cov_np = x_cov.numpy()
#         #     # Now you can pass these numpy arrays to your model
#         #     #loc, scale = torch_layer([x_lag_np, x_cov_np])

#         #     loc, scale = torch_layer([x_lag, x_cov])
#         #     output = tfp.layers.DistributionLambda(
#         #         lambda t: tfd.Normal(loc=t[0], scale=t[1])
#         #     )([loc, scale])
        
#         # else:
#         #     # Handle other PF_method cases (point, qr, JSU, etc.)
#         #     pass
#         # self.model = tf.keras.Model(inputs=[x_lag, x_cov], outputs=[output])
#         # Concatenate inputs
#         #x = tf.concat([x_lag, x_cov], axis=-1) 
#         # LSTM layers with dropout
#         # for _ in range(self.settings['n_hidden_layers']):
#         #     x = tf.keras.layers.LSTM(self.settings['hidden_size'], return_sequences=True)(x)
#         #     x = tf.keras.layers.Dropout(0.1)(x)

#         x = tf.keras.layers.LSTM(self.settings['hidden_size'], return_sequences=True)(x__in)
#         x = tf.keras.layers.Dropout(0.1)(x)
#         x = tf.keras.layers.LSTM(self.settings['hidden_size'], return_sequences=True)(x)
#         x = tf.keras.layers.Dropout(0.1)(x)
        
#         # Fully connected layers for location and scale
#         x = tf.keras.layers.LSTM(self.settings['hidden_size'], return_sequences=False)(x)
#         x = tf.keras.layers.Dropout(0.1)(x)
#         loc = tf.keras.layers.Dense(self.settings['pred_horiz'])(x)
#         scale = tf.keras.layers.Dense(self.settings['pred_horiz'], activation='softplus')(x)
        
#         # Add epsilon for numerical stability
#         scale = scale + 1e-6
        
#         # Distribution layer
#         output = tfp.layers.DistributionLambda(
#             lambda t: tfd.Normal(loc=t[0], scale=t[1])
#         )([loc, scale])

#         # Compile model
#         self.model = tf.keras.Model(inputs=[x], outputs=[output])
#         self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.settings['lr']),
#                            loss=loss)
#         self.model.summary()


#     def fit(self, train_x, train_y, val_x, val_y, verbose=0, pruning_call=None):
#         # Convert the data into the input format using the internal converter
#         train_x = self.build_model_input_from_series(x=train_x,
#                                                      col_names=self.settings['x_columns_names'],
#                                                      pred_horiz=self.settings['pred_horiz'])
#         val_x = self.build_model_input_from_series(x=val_x,
#                                                    col_names=self.settings['x_columns_names'],
#                                                    pred_horiz=self.settings['pred_horiz'])
#         es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
#                                               patience=self.settings['patience'],
#                                               restore_best_weights=False)

#         # Create folder to temporally store checkpoints
#         checkpoint_path = os.path.join(os.getcwd(), 'tmp_checkpoints', 'cp.weights.h5')
#         checkpoint_dir = os.path.dirname(checkpoint_path)
#         if not os.path.exists(checkpoint_dir):
#             os.makedirs(checkpoint_dir)

#         cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                 monitor="val_loss", mode="min",
#                                                 save_best_only=True,
#                                                 save_weights_only=True, verbose=0)
#         if pruning_call==None:
#             callbacks = [es, cp]
#         else:
#             callbacks = [es, cp, pruning_call]

#         history = self.model.fit(train_x,
#                                  train_y,
#                                  validation_data=(val_x, val_y),
#                                  epochs=self.settings['max_epochs'],
#                                  batch_size=self.settings['batch_size'],
#                                  callbacks=callbacks,
#                                  verbose=0)
#         # Load best weights: do not use restore_best_weights from early stop since works only in case it stops training
#         self.model.load_weights(checkpoint_path)
#         # delete temporary folder
#         shutil.rmtree(checkpoint_dir)
#         return history

#     def predict(self, x):
#         x = self.build_model_input_from_series(x=x,
#                                                col_names=self.settings['x_columns_names'],
#                                                pred_horiz=self.settings['pred_horiz'])
#         return self.model(x)

#     def evaluate(self, x, y):
#         x = self.build_model_input_from_series(x=x,
#                                                col_names=self.settings['x_columns_names'],
#                                                pred_horiz=self.settings['pred_horiz'])
#         return self.model.evaluate(x=x, y=y)

#     @staticmethod
#     def build_model_input_from_series(x, col_names: List, pred_horiz: int):
#         # get index of target and past features
#         past_col_idxs = [index for (index, item) in enumerate(col_names)
#                          if features_keys['target'] in item or features_keys['past'] in item]

#         # get index of const features
#         const_col_idxs = [index for (index, item) in enumerate(col_names)
#                           if features_keys['const'] in item]

#         # get index of futu features
#         futu_col_idxs = [index for (index, item) in enumerate(col_names)
#                          if features_keys['futu'] in item]

#         # build conditioning variables for past features
#         past_feat = [x[:, :-pred_horiz, feat_idx] for feat_idx in past_col_idxs]
#         # build conditioning variables for futu features
#         futu_feat = [x[:, -pred_horiz:, feat_idx] for feat_idx in futu_col_idxs]
#         # build conditioning variables for cal features
#         c_feat = [x[:, -pred_horiz:-pred_horiz + 1, feat_idx] for feat_idx in const_col_idxs]

#         print(pred_horiz)
#         print(past_col_idxs)
#         print(const_col_idxs)
#         print(futu_col_idxs)

#         # return flattened input
#         return np.concatenate(past_feat + futu_feat + c_feat, axis=1)

#     @staticmethod
#     def get_hyperparams_trial(trial, settings):
#         settings['hidden_size'] = trial.suggest_int('hidden_size', 64, 960, step=64)
#         settings['n_hidden_layers'] = 2  # trial.suggest_int('n_hidden_layers', 1, 3)
#         settings['lr'] = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
#         settings['activation'] = 'softplus'
#         return settings

#     @staticmethod
#     def get_hyperparams_searchspace():
#         return {'hidden_size': [128, 512],
#                 'lr': [1e-4, 1e-3]}

#     @staticmethod
#     def get_hyperparams_dict_from_configs(configs):
#         model_hyperparams = {
#             'hidden_size': configs['hidden_size'],
#             'n_hidden_layers': configs['n_hidden_layers'],
#             'lr': configs['lr'],
#             'activation': configs['activation']
#         }
#         return model_hyperparams

#     def plot_weights(self):
#         w_b = self.model.layers[1].get_weights()
#         plt.imshow(w_b[0].T)
#         plt.title('DNN input weights')
#         plt.show()


#     # def fit(self, train_x, train_y, val_x, val_y, verbose=0, pruning_call=None):
#     #     train_x_lag, train_x_cov = self.build_model_input_from_series(train_x, self.settings['x_columns_names'], self.settings['pred_horiz'])
#     #     val_x_lag, val_x_cov = self.build_model_input_from_series(val_x, self.settings['x_columns_names'], self.settings['pred_horiz'])
        
#     #     es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",
#     #                                           patience=self.settings['patience'],
#     #                                           restore_best_weights=False)

#     #     checkpoint_path = os.path.join(os.getcwd(), 'tmp_checkpoints', 'cp.weights.h5')
#     #     checkpoint_dir = os.path.dirname(checkpoint_path)
#     #     if not os.path.exists(checkpoint_dir):
#     #         os.makedirs(checkpoint_dir)

#     #     cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#     #                                             monitor="val_loss", mode="min",
#     #                                             save_best_only=True,
#     #                                             save_weights_only=True, verbose=0)
#     #     callbacks = [es, cp] if pruning_call is None else [es, cp, pruning_call]

#     #     history = self.model.fit([train_x_lag, train_x_cov], train_y,
#     #                              validation_data=([val_x_lag, val_x_cov], val_y),
#     #                              epochs=self.settings['max_epochs'],
#     #                              batch_size=self.settings['batch_size'],
#     #                              callbacks=callbacks,
#     #                              verbose=0)
#     #     self.model.load_weights(checkpoint_path)
#     #     shutil.rmtree(checkpoint_dir)
#     #     return history

#     # def predict(self, x):
#     #     x_lag, x_cov = self.build_model_input_from_series(x, self.settings['x_columns_names'], self.settings['pred_horiz'])
#     #     return self.model([x_lag, x_cov])

#     # def evaluate(self, x, y):
#     #     x_lag, x_cov = self.build_model_input_from_series(x, self.settings['x_columns_names'], self.settings['pred_horiz'])
#     #     return self.model.evaluate([x_lag, x_cov], y)

#     # @staticmethod
#     # def build_model_input_from_series(x, col_names: List, pred_horiz: int):
#     #     #past_col_idxs = [index for (index, item) in enumerate(col_names)
#     #     #                 if 'target' in item or 'past' in item]

#     #     #const_col_idxs = [index for (index, item) in enumerate(col_names)
#     #      #                 if 'const' in item]

#     #     #futu_col_idxs = [index for (index, item) in enumerate(col_names)
#     #      #                if 'futu' in item]

#     #     past_col_idxs = [index for (index, item) in enumerate(col_names)
#     #                      if features_keys['target'] in item or features_keys['past'] in item]

#     #     # get index of const features
#     #     const_col_idxs = [index for (index, item) in enumerate(col_names)
#     #                       if features_keys['const'] in item]

#     #     # get index of futu features
#     #     futu_col_idxs = [index for (index, item) in enumerate(col_names)
#     #                      if features_keys['futu'] in item]

#     #     past_feat = [x[:, :-pred_horiz, feat_idx] for feat_idx in past_col_idxs]
#     #     futu_feat = [x[:, -pred_horiz:, feat_idx] for feat_idx in futu_col_idxs]
#     #     c_feat = [x[:, -pred_horiz:-pred_horiz + 1, feat_idx] for feat_idx in const_col_idxs]
        
        
#     #     print(pred_horiz)
#     #     print(col_names)
#     #     print(past_col_idxs)
#     #     print(const_col_idxs)
#     #     print(futu_col_idxs)
#     #     print(past_feat)
#     #     print(futu_feat)
#     #     print(c_feat)
#     #     x_lag = np.concatenate(past_feat, axis=1)
#     #     x_cov = np.concatenate(futu_feat + c_feat, axis=1)
        
#     #     return x_lag, x_cov

#     # @staticmethod
#     # def get_hyperparams_trial(trial, settings):
#     #     settings['hidden_size'] = trial.suggest_int('hidden_size', 64, 960, step=64)
#     #     settings['n_hidden_layers'] = 2  # trial.suggest_int('n_hidden_layers', 1, 3)
#     #     settings['lr'] = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
#     #     settings['activation'] = 'softplus'
#     #     return settings

#     # @staticmethod
#     # def get_hyperparams_searchspace():
#     #     return {'hidden_size': [128, 512],
#     #             'lr': [1e-4, 1e-3]}

#     # @staticmethod
#     # def get_hyperparams_dict_from_configs(configs):
#     #     model_hyperparams = {
#     #         'hidden_size': configs['hidden_size'],
#     #         'n_hidden_layers': configs['n_hidden_layers'],
#     #         'lr': configs['lr'],
#     #         'activation': configs['activation']
#     #     }
#     #     return model_hyperparams

#     # def plot_weights(self):
#     #     w_b = self.model.layers[1].get_weights()
#     #     plt.imshow(w_b[0].T)
#     #     plt.title('Deep AR Weights')
#     #     plt.show()



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
from tcn import TCN


class DeepARRegressor:
    def __init__(self, settings, loss):
        self.settings = settings
        self.__build_model__(loss)

    def __build_model__(self, loss):

        # #INPUTS
        # x_in = tf.keras.layers.Input(shape=(self.settings['input_size'],1,))#1
        # print((self.settings['input_size'],1))
        # x_in1 = tf.keras.layers.Lambda(lambda x: x[:, :-78, :])(x_in)
        # x_in2 = tf.keras.layers.Lambda(lambda x: x[:, -78:, :])(x_in)
        # x_in1 = tf.keras.layers.BatchNormalization()(x_in1)
        # x_in2 = tf.keras.layers.BatchNormalization()(x_in2)
        # print((self.settings['input_size'],1))
        # x_in1 = tf.keras.layers.GaussianNoise(0.1)(x_in1)
        # n_lstm=256
        # #x = tf.keras.layers.SimpleRNN(n_lstm, activation='gelu', return_sequences=True)(x_in1)
        # #x = tf.keras.layers.Dropout(0.15)(x)
        # #x = tf.keras.layers.LSTM(n_lstm, activation='tanh', return_sequences=True)(x)
        # #x = tf.keras.layers.Dropout(0.1)(x)
        # #RECURRENT E DENSE
        # x = tf.keras.layers.SimpleRNN(n_lstm*2, activation='gelu', return_sequences=False)(x_in1)  
        # x = tf.keras.layers.Dropout(0.15)(x)
        # x_in2 = tf.keras.layers.Flatten()(x_in2)
        # x2= tf.keras.layers.Dense(256, activation='gelu')(x_in2)
        # x2 = tf.keras.layers.Dropout(0.15)(x2)
        # x2= tf.keras.layers.Dense(256, activation='gelu')(x2)
        # x2 = tf.keras.layers.Dropout(0.15)(x2)
        # #CONCATENATE
        # x = tf.keras.layers.Concatenate()([x, x2])


        # ultimo modello inputs
        x_in = tf.keras.layers.Input(shape=(self.settings['input_size'],1,))#1
        print((self.settings['input_size'],1))
        x_in1 = tf.keras.layers.Lambda(lambda x: x[:, :-78, :])(x_in)
        x_in2 = tf.keras.layers.Lambda(lambda x: x[:, -78:, :])(x_in)
        x_in1 = tf.keras.layers.BatchNormalization()(x_in1)
        x_in2 = tf.keras.layers.BatchNormalization()(x_in2)

        # x_in1 = TCN(nb_filters=256, kernel_size=5, nb_stacks=1, dilations=[1], padding='same', use_skip_connections=True, dropout_rate=0.1, return_sequences=False)(x_in1)
        # x_in2 = tf.keras.layers.Flatten()(x_in2)
        # x_in2 = tf.keras.layers.Dense(256)(x_in2)
        # x_in2 = tf.keras.layers.Dropout(0.15)(x_in2)
        # x = tf.keras.layers.Concatenate()([x_in1, x_in2])


       # ultimo modello continuo parametri
        print((self.settings['input_size'],1))
        x_in1 = tf.keras.layers.LSTM(256, activation='tanh', return_sequences=False)(x_in1)
        x_in1 = tf.keras.layers.Dropout(0.2)(x_in1)
        x_in2= tf.keras.layers.Flatten()(x_in2)
        x_in22 = tf.keras.layers.Dense(128, activation='tanh')(x_in2)
        x_in22 = tf.keras.layers.Dropout(0.15)(x_in22)
        x = tf.keras.layers.Concatenate()([x_in1, x_in2, x_in22])




        #x = tf.keras.layers.GaussianNoise(0.1)(x_in1)
         # Padding on the both ends of time series
        # kernel_size = 3
        # num_of_pads = (kernel_size - 1) // 2
        # front = tf.keras.layers.Lambda(lambda x: tf.repeat(x[:, 0:1, :], num_of_pads, axis=1))(x)
        # end = tf.keras.layers.Lambda(lambda x: tf.repeat(x[:, -1:, :], num_of_pads, axis=1))(x)
        # x_padded = tf.keras.layers.Concatenate(axis=1)([front, x, end])
        # # Calculate the trend and seasonal part of the series
        # x_trend = tf.keras.layers.AveragePooling1D(pool_size=kernel_size, strides=1, padding='valid')(x_padded)
        # #x_trend = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.keras.layers.AveragePooling1D(pool_size=kernel_size, strides=1, padding='valid')(tf.transpose(x, perm=[0, 2, 1])), perm=[0, 2, 1]))(x_padded)
        # x_seasonal = tf.keras.layers.Subtract()([x, x_trend])
        # x_trend = tf.keras.layers.SimpleRNN(128, activation='gelu', return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_trend)
        # x_trend = tf.keras.layers.Dropout(0.2)(x_trend)
        # x_seasonal = tf.keras.layers.SimpleRNN(128, activation='gelu', return_sequences=False, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_seasonal)
        # x_seasonal = tf.keras.layers.Dropout(0.2)(x_seasonal)
        # x_cov = tf.keras.layers.Flatten()(x_in2)
        # x_cov = tf.keras.layers.Dense(256, activation='gelu')(x_cov)
        # x_cov = tf.keras.layers.Dropout(0.2)(x_cov)
        # #x_cov = tf.keras.layers.Dense(256, activation='gelu')(x_cov)
        # #x_cov = tf.keras.layers.Dropout(0.15)(x_cov)
        #x = tf.keras.layers.Concatenate()([x_trend, x_seasonal, x_cov])






        if self.settings['PF_method'] == 'point':
            out_size = 1
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                          activation='linear',
                                          kernel_regularizer=tf.keras.regularizers.l2(0.001),
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
            #print("sto facendo t-student")
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                            activation='linear',
                                            )(x)
            output = tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(
                    loc=t[..., :self.settings['pred_horiz']],
                    scale=1e-3 + 3 * tf.math.softplus(0.05 * t[..., self.settings['pred_horiz']:])))(logit)
        
        elif self.settings['PF_method'] == 'JSU':
            out_size = 4 
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
            print("sto facendo t-student")
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                            activation='linear',
                                            )(x)
            #df_reg= tf.keras.layers.Dense(self.settings['pred_horiz'] ,
            #                                activation='softplus',
            #                                )(x)
            output = tfp.layers.DistributionLambda(
                lambda t : tfd.StudentT(
                    df=1+ tf.math.softplus(0.05 * t[..., self.settings['pred_horiz']*2:self.settings['pred_horiz']*3 ]),
                    loc=t[..., :self.settings['pred_horiz']],
                    scale=1e-3 + 3 * tf.math.softplus(0.05 * t[..., self.settings['pred_horiz']: self.settings['pred_horiz']*2 ])))(logit)

       
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
