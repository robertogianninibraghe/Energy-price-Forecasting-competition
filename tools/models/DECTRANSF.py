

from tools.data_utils import features_keys
import numpy as np
import tensorflow as tf
from typing import List
import sys
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import os
import shutil
import matplotlib.pyplot as plt


def transformer_encoder(inputs, head_size, num_heads, ff_dim,
                    dropout=0):
    """
    Creates a single transformer block.
    """
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(
    key_dim=head_size, num_heads=num_heads, dropout=dropout,
    attention_axes=1
    )(x, x)
    x = tf.keras.layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


class DECTRANSFRegressor:

    
    def __init__(self, settings, loss):
        self.settings = settings
        self.__build_model__(loss)

    def __build_model__(self, loss):

        pred_horiz = self.settings['pred_horiz']
        d_hidden = self.settings['hidden_size']
        x_in = tf.keras.layers.Input(shape=(self.settings['input_size'],1,))
        x_in = tf.keras.layers.BatchNormalization()(x_in)
        #x_in = tf.keras.layers.GaussianNoise(0.05)(x_in)
        x = tf.keras.layers.Lambda(lambda x: x[:, :-78, :])(x_in)
        x_in2 = tf.keras.layers.Lambda(lambda x: x[:, -78:, :])(x_in)
        # Padding on the both ends of time series
        kernel_size = 3
        num_of_pads = (kernel_size - 1) // 2
        front = tf.keras.layers.Lambda(lambda x: tf.repeat(x[:, 0:1, :], num_of_pads, axis=1))(x)
        end = tf.keras.layers.Lambda(lambda x: tf.repeat(x[:, -1:, :], num_of_pads, axis=1))(x)
        x_padded = tf.keras.layers.Concatenate(axis=1)([front, x, end])
        # Calculate the trend and seasonal part of the series
        x_trend = tf.keras.layers.AveragePooling1D(pool_size=kernel_size, strides=1, padding='valid')(x_padded)
        x_seasonal = tf.keras.layers.Subtract()([x, x_trend])
        x_trend = transformer_encoder(x_trend, head_size = 128, num_heads = 5, ff_dim = 128, dropout = 0.01)
        x_seasonal= transformer_encoder(x_seasonal, head_size = 128, num_heads = 5, ff_dim = 128, dropout = 0.01)
        x_cov = transformer_encoder(x_in2, head_size = 256, num_heads = 5, ff_dim = 128, dropout = 0.01)
        x_trend = tf.keras.layers.Dropout(0.15)(x_trend)
        x_seasonal = tf.keras.layers.Dropout(0.15)(x_seasonal)
        x_cov = tf.keras.layers.Dropout(0.15)(x_cov)
        x_trend = tf.keras.layers.Flatten()(x_trend)
        x_seasonal = tf.keras.layers.Flatten()(x_seasonal)
        x_cov = tf.keras.layers.Flatten()(x_cov)

        x = tf.keras.layers.Concatenate()([x_trend, x_seasonal, x_cov])


        #tried separating also the constant features, but performed worse
        #x_const_feat = tf.keras.layers.Flatten()(x_const_feat)
        #x_const_feat =transformer_encoder(x_const_feat, head_size = 32, num_heads = 5, ff_dim = 16, dropout = 0.01)
        #x_const_feat = tf.keras.layers.Dropout(0.15)(x_const_feat)

        ###Tried adding a rnn layer after the encoder, but model was way  to complex
        #x_trend = tf.keras.layers.SimpleRNN(128, return_sequences=False)(x_trend)
        #x_seasonal = tf.keras.layers.SimpleRNN(128, return_sequences=False)(x_seasonal)


        if self.settings['PF_method'] == 'point':
            out_size = 1
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                          activation='linear',
                                          #kernel_regularizer=tf.keras.regularizers.l1(0.001),
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
        
        elif self.settings['PF_method'] == 'JSU':  
            out_size = 4 
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                            activation='linear',
                                            #kernel_regularizer=tf.keras.regularizers.l2(2e-5),
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
        self.model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=self.settings['lr']),
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
                                              restore_best_weights=True)
        

        # Create folder to temporally store checkpoints
        checkpoint_path = os.path.join(os.getcwd(), 'tmp_checkpoints', 'cp.weights.h5')
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                monitor="val_loss", mode="min",
                                                save_best_only=True,
                                                save_weights_only=True, verbose=0)
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25,
                              patience=6, min_lr=0.0001)

        if pruning_call==None:
            callbacks = [es, cp]
        else:
            callbacks = [es, cp, pruning_call]

        history = self.model.fit(train_x,
                                 train_y,
                                 validation_data=(val_x, val_y),
                                 epochs=self.settings['max_epochs'],
                                 batch_size=self.settings['batch_size'],
                                 callbacks=callbacks,
                                 verbose=1)
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
