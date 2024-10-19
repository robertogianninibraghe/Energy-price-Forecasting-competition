
from tools.data_utils import features_keys
import numpy as np
import tensorflow as tf
from typing import List
import os
import shutil
import matplotlib.pyplot as plt




class DECRegressor:
    def __init__(self, settings, loss):
        self.settings = settings
        self.__build_model__(loss)

    def __build_model__(self, loss):
        # Here tried some combinations of trend and seasonality layers to see the performances
        #mainly reported the 

        x_in = tf.keras.layers.Input(shape=(self.settings['input_size'],1,))
        x_in = tf.keras.layers.BatchNormalization()(x_in)
        #split in prices historical data and covariates
        x = tf.keras.layers.Lambda(lambda x: x[:, :-78, :])(x_in)
        x_in2 = tf.keras.layers.Lambda(lambda x: x[:, -78:, :])(x_in)
        #x_in1 = tf.keras.layers.BatchNormalization()(x_in1)
        #x_in2 = tf.keras.layers.BatchNormalization()(x_in2)

        d_hidden = self.settings['d_hidden']

        # Padding on the both ends of time series
        kernel_size = 3
        num_of_pads = (kernel_size - 1) // 2
        front = tf.keras.layers.Lambda(lambda x: tf.repeat(x[:, 0:1, :], num_of_pads, axis=1))(x)
        end = tf.keras.layers.Lambda(lambda x: tf.repeat(x[:, -1:, :], num_of_pads, axis=1))(x)
        x_padded = tf.keras.layers.Concatenate(axis=1)([front, x, end])

        # Calculate the trend and seasonal part of the series
        x_trend = tf.keras.layers.AveragePooling1D(pool_size=kernel_size, strides=1, padding='valid')(x_padded)
        #x_trend = tf.keras.layers.Lambda(lambda x: tf.transpose(tf.keras.layers.AveragePooling1D(pool_size=kernel_size, strides=1, padding='valid')(tf.transpose(x, perm=[0, 2, 1])), perm=[0, 2, 1]))(x_padded)
        x_seasonal = tf.keras.layers.Subtract()([x, x_trend])

        # Flatten the trend and seasonal part
        x_trend = tf.keras.layers.Flatten()(x_trend)
        x_seasonal = tf.keras.layers.Flatten()(x_seasonal)
        x_in2 = tf.keras.layers.Flatten()(x_in2)

        x_trend = tf.keras.layers.Dense(d_hidden, activation='relu')(x_trend)
        x_seasonal = tf.keras.layers.Dense(d_hidden, activation='relu')(x_seasonal)
        x_in2 = tf.keras.layers.Dense(d_hidden, activation='relu')(x_in2)
        x_trend = tf.keras.layers.Dropout(0.2)(x_trend)
        x_seasonal = tf.keras.layers.Dropout(0.2)(x_seasonal)
        x_in2 = tf.keras.layers.Dropout(0.2)(x_in2)
        x_final = tf.keras.layers.Concatenate()([x_trend,x_seasonal, x_in2])
        
        #try without regularization
        # logit = tf.keras.layers.Dense(self.settings['pred_horiz'],
        #                               activation='linear',
        #                               
        #                               )(x_final)
        logit = tf.keras.layers.Dense(self.settings['pred_horiz'],
                                      activation='linear',
                                      kernel_regularizer=tf.keras.regularizers.l1(self.settings['l1']))(x_final)
        



        #output = tf.reshape(logit, (-1, self.settings['pred_horiz'], 1))
        output = tf.keras.layers.Reshape((self.settings['pred_horiz'], 1))(logit)

        # Create model
        self.model= tf.keras.Model(inputs=[x_in], outputs=[output])
        # Compile the model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.settings['lr']),
                           loss=loss
                           )
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

        history = self.model.fit(train_x,
                                 train_y,
                                 validation_data=(val_x, val_y),
                                 epochs=self.settings['max_epochs'],
                                 batch_size=self.settings['batch_size'],
                                 callbacks=callbacks,
                                 verbose=0)

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
        settings['l1'] = trial.suggest_float('l1', 1e-7, 1e-1)
        settings['lr'] = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
        return settings

    @staticmethod
    def get_hyperparams_searchspace():
        return {'l1': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
                'lr': [1e-4, 1e-3, 1e-2]}

    @staticmethod
    def get_hyperparams_dict_from_configs(configs):
        model_hyperparams = {
            'l1': configs['l1'],
            'lr': configs['lr']
        }
        return model_hyperparams

    def plot_weights(self):
        w_b = self.model.layers[1].get_weights()
        plt.imshow(w_b[0].T)
        l1=str(self.settings['l1'])
        plt.title('ARX Weights - l1:' + l1)
        plt.show()

