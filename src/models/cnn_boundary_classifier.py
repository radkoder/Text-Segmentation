import tensorflow as tf
import numpy as np
import datasets
from datasets import windowed

class CNNAttention1D(tf.keras.layers.Layer):
    def __init__(self,inner_dim=512, outer_dim = 512, padding = 'same',**kwargs):
        super().__init__(**kwargs)
        self.inner_dim = inner_dim
        self.outer_dim = outer_dim
        self.padding = padding

        self.qconv = tf.keras.layers.Conv1D(filters=inner_dim,kernel_size=3,padding='same',activation='relu',bias_initializer='ones')
        self.kconv = tf.keras.layers.Conv1D(filters=inner_dim,kernel_size=3,padding='same',activation='relu',bias_initializer='ones')
        self.vconv = tf.keras.layers.Conv1D(filters=inner_dim,kernel_size=3,padding='same',activation='relu',bias_initializer='ones')
        self.outconv = tf.keras.layers.Conv1D(filters=outer_dim,kernel_size=1,padding=padding,activation='relu',bias_initializer='ones')
        self.troughconv = tf.keras.layers.Conv1D(filters=outer_dim,kernel_size=1,padding=padding,activation='tanh')
        self.scaler = tf.constant(1/np.sqrt(inner_dim),dtype=tf.float32)

    def call(self, inputs):
        q = self.qconv(inputs)
        k = self.kconv(inputs)
        v = self.vconv(inputs)
        scores =tf.nn.softmax(tf.matmul(q,k,transpose_b=True) * self.scaler)
        scores = tf.matmul(scores,v)
        return self.troughconv(inputs) + self.outconv(scores)
    def get_config(self):
        cfg = super().get_config()
        cfg["inner_dim"] = self.inner_dim
        cfg["outer_dim"] = self.outer_dim
        cfg["padding"] = self.padding
        return cfg
class PureAttention1D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.scaler = tf.constant(1/np.sqrt(512),dtype=tf.float32)
    def call(self, inputs):
        scores =tf.nn.softmax(tf.matmul(inputs,inputs,transpose_b=True) * self.scaler)
        scores = tf.matmul(scores,inputs)
        return scores

def make_attention_cnn(cnn_layers = [(64,128),(32,64),(16,32),(8,16)], dense_layers=[128,64], cnn_dropout = 0.2, dense_dropout=0.5, input_shape = (1,10,512)):
    model = tf.keras.Sequential()
    for cnn_args in cnn_layers:
        model.add(CNNAttention1D(*cnn_args))
        model.add(tf.keras.layers.Dropout(cnn_dropout))
    model.add(tf.keras.layers.Flatten())
    for dense_dim in dense_layers:
        model.add(tf.keras.layers.Dense(dense_dim,activation='relu'))
        model.add(tf.keras.layers.Dropout(dense_dropout))
    model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

    # build model
    model(np.zeros(shape=input_shape))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss='binary_crossentropy',metrics=['accuracy'])
    return model

def get_callbacks():
    return [tf.keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(patience=5,factor=0.1,verbose=1),
           tf.keras.callbacks.ModelCheckpoint('models/checkpoints/cnn_attention.{epoch:02d}-{val_loss:.2f}.hdf5')]


def train(model : tf.keras.Model, ds : datasets.EmbeddedDataset, val_split = 0.8,batch_size = 16,name='cnn_model', **kwargs):
    val_bound = int(len(ds)*val_split)
    wds_train = windowed.WindowedDataset(ds,batch_size=batch_size,limits=(0,val_bound))
    wds_val = windowed.WindowedDataset(ds,batch_size=batch_size, limits=(val_bound, None))

    model.fit(wds_train, validation_data=wds_val,**kwargs)
    model.save(name)
    return model

def load(name):
    return tf.keras.models.load_model(filepath=name, custom_objects={"CNNAttention1D":CNNAttention1D, "PureAttention1D":PureAttention1D})




