
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


class TRANSFRegressor:

    
    def __init__(self, settings, loss):
        self.settings = settings
        self.__build_model__(loss)

    def __build_model__(self, loss):

        pred_horiz = self.settings['pred_horiz']
        d_hidden = self.settings['hidden_size']
        x_in0 = tf.keras.layers.Input(shape=(self.settings['input_size'], 1,))
        x_in0 = tf.keras.layers.BatchNormalization()(x_in0)
        past_size= self.settings['input_size'] - 6 - pred_horiz*3
        past_size = int(past_size/4)
        x_in1 = tf.keras.layers.Lambda(lambda x: x[:, :past_size, :], name='time_series')(x_in0)
        #x_cov_in = tf.keras.layers.Input(shape=(self.settings['input_size'] + pred_horiz, d_cov))

        # Split covariates into historical and future
        x_const_feat = tf.keras.layers.Lambda(lambda x: x[:, past_size+pred_horiz*3:past_size+pred_horiz*3 +6, :], name='const')(x_in0)
        
        x_cov_hist = tf.keras.layers.Lambda(lambda x: x[:, - past_size*3:, :])(x_in0)
        x_cov_hist = tf.keras.layers.Reshape((past_size, 3,))(x_cov_hist)
        x_cov_future = tf.keras.layers.Lambda(lambda x: x[:, past_size:past_size+pred_horiz*3, :])(x_in0)
        x_cov_future = tf.keras.layers.Reshape((pred_horiz, 3,))(x_cov_future)

        #print(self.settings)
        #x_in = tf.keras.layers.Input(shape=(self.settings['input_size'],1,))#1
        # x_in1 = tf.keras.layers.Lambda(lambda x: x[:, :-78, :])(x_in)
        # x_in2 = tf.keras.layers.Lambda(lambda x: x[:, -78:, :])(x_in)
        # x_in1 = tf.keras.layers.BatchNormalization()(x_in1)
        # x_in2 = tf.keras.layers.BatchNormalization()(x_in2)
        # print((self.settings['input_size'],1))

        h_cov = x_cov_hist
        #h_lag = tf.concat((x_in1, h_cov), axis=-1)
        h_lag=tf.keras.layers.Concatenate(axis=-1, name="lag")([x_in1, h_cov])
        h_cov = tf.keras.layers.Concatenate(axis=1, name= "cov")([h_cov, x_cov_future])

        x = transformer_encoder(h_lag, head_size = 128, num_heads = 5, ff_dim = 128, dropout = 0.01)
        x_in2 = transformer_encoder(h_cov, head_size = 128, num_heads = 5, ff_dim = 128, dropout = 0.01)
        #x_const_feat = tf.keras.layers.Flatten()(x_const_feat)
        
        #x_const_feat =transformer_encoder(x_const_feat, head_size = 32, num_heads = 5, ff_dim = 16, dropout = 0.01)
        #x_const_feat = tf.keras.layers.Dropout(0.15)(x_const_feat)
        
        #x = tf.keras.layers.SimpleRNN(n_lstm, activation='gelu')(x)
        #x_in2 = tf.keras.layers.SimpleRNN(n_lstm, activation='gelu')(x_in2)
        x = tf.keras.layers.Dropout(0.15)(x)
        x_in2 = tf.keras.layers.Dropout(0.15)(x_in2)
        

        x = tf.keras.layers.Flatten()(x)
        x_in2 = tf.keras.layers.Flatten()(x_in2)
        x_const_feat = tf.keras.layers.Flatten()(x_const_feat)



        #x_in2 = tf.keras.layers.Flatten()(x_in2)
        x = tf.keras.layers.Concatenate()([x, x_in2, x_const_feat])
        #x = tf.keras.layers.Dense(d_hidden, kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

        ##opzione 2

        #x = tf.keras.layers.Dropout(0.01)(x)

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
            out_size = 4 
            #x = tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_regularizer=tf.keras.regularizers.l2(2e-5))(x)
            logit = tf.keras.layers.Dense(self.settings['pred_horiz'] * out_size,
                                            activation='sigmoid',#qui era linear
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
        self.model= tf.keras.Model(inputs=[x_in0], outputs=[output])
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
                                              restore_best_weights=True)#era false
        

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
                              patience=5, min_lr=0.0001)
        
        if pruning_call==None:
            callbacks = [es,reduce_lr, cp]
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

    # @staticmethod
    # def build_model_input_from_series(x, col_names: List, pred_horiz: int):
    #     # get index of target and past features
    #     past_col_idxs = [index for (index, item) in enumerate(col_names)
    #                      if features_keys['target'] in item or features_keys['past'] in item]

    #     # get index of const features
    #     const_col_idxs = [index for (index, item) in enumerate(col_names)
    #                       if features_keys['const'] in item]

    #     # get index of futu features
    #     futu_col_idxs = [index for (index, item) in enumerate(col_names)
    #                      if features_keys['futu'] in item]

    #     # build conditioning variables for past features
    #     past_feat = [x[:, :-pred_horiz, feat_idx] for feat_idx in past_col_idxs]
    #     # build conditioning variables for futu features
    #     futu_feat = [x[:, -pred_horiz:, feat_idx] for feat_idx in futu_col_idxs]
    #     # build conditioning variables for cal features
    #     c_feat = [x[:, -pred_horiz:-pred_horiz + 1, feat_idx] for feat_idx in const_col_idxs]

    #     # return flattened input
    #     return np.concatenate(past_feat + futu_feat + c_feat, axis=1)

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
        past_cov_feat = [x[:, :-pred_horiz, feat_idx] for feat_idx in futu_col_idxs]

        # return flattened input
        return np.concatenate(past_feat + futu_feat + c_feat+ past_cov_feat, axis=1)

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
