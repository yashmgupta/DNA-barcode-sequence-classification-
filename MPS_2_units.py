

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensornetwork as tn


# Def TN layer
class TNLayer(tf.keras.layers.Layer):
    def __init__(self, units=3,in_dim=28, out_dim=10,bond_dim=10):
        '''
        units = number of tensors
        in_dim = dimension of input (physical leg) of each tensor
        out_dim = Number of classes
        '''
        
        super(TNLayer, self).__init__()
        # Create the variables for the layer.
        
        self.units = units
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bond_dim = bond_dim
        
        self.bias = tf.Variable(tf.zeros(shape=tuple([self.out_dim])), name="bias", trainable=True)
        
        self.tensor_list = [tf.Variable(tf.random.uniform(shape =(in_dim,bond_dim)),trainable=True)] + [tf.Variable(tf.random.uniform(shape =(in_dim,bond_dim,out_dim)),trainable=True)]
        
        
    def call(self, inputs):
        def TN_contraction(input_i, tensor_list, bias, units, in_dim, out_dim, bond_dim):
            
            
            x_vec = tf.reshape(input_i, (units,in_dim))

            

            mpo = [tn.Node(tensor_list[i],name =str(i)) for i in range(len(tensor_list))]

            x_node = [tn.Node(x_vec[i], name ="x"+str(i)) for i in range(len(x_vec))]


            mpo[0][1]^mpo[1][1]

            for i in range(units):
                mpo[i][0]^x_node[i][0]

#             result = tn.contractors.auto(mpo+x_node, ignore_edge_order=True)
            tmp = []
            for i in range(units):
                tmp.append(mpo[i]@x_node[i])

            c = tmp[0]@tmp[1]
            
                       

            return c.tensor+bias
                
        result = tf.vectorized_map(lambda vec:TN_contraction(vec, self.tensor_list, self.bias, self.units, self.in_dim, self.out_dim, self.bond_dim), inputs)

        return tf.nn.sigmoid(tf.reshape(result,(-1,self.out_dim)))

    


