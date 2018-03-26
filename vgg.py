from enum import Enum
import tensorflow as tf
import scipy.io
import numpy as np

class VggModel:
        
    def __init__(self, data_path, img):
        self.raw_data = scipy.io.loadmat(data_path)
        self.raw_layers = self.raw_data['layers'][0]
        self.__build_net(img)

    def __build_net(self, img):
        class Type(Enum):
            Conv = 1
            Relu = 2
            Pool = 3
            
        net_config = (('conv1_1', Type.Conv, 0), ('relu1_1', Type.Relu, 0),
                        ('conv1_2', Type.Conv, 2), ('relu1_2', Type.Relu, 2),
                        ('pool1', Type.Pool),
                        ('conv2_1', Type.Conv, 5), ('relu2_1', Type.Relu, 5),
                        ('conv2_2', Type.Conv, 7), ('relu2_2', Type.Relu, 7),
                        ('pool2', Type.Pool),
                        ('conv3_1', Type.Conv, 10), ('relu3_1', Type.Relu, 10),
                        ('conv3_2', Type.Conv, 12), ('relu3_2', Type.Relu, 12),
                        ('conv3_3', Type.Conv, 14), ('relu3_3', Type.Relu, 14),
                        ('conv3_4', Type.Conv, 16), ('relu3_4', Type.Relu, 16),
                        ('pool3', Type.Pool),
                        ('conv4_1', Type.Conv, 19), ('relu4_1', Type.Relu, 19),
                        ('conv4_2', Type.Conv, 21), ('relu4_2', Type.Relu, 21),
                        ('conv4_3', Type.Conv, 23), ('relu4_3', Type.Relu, 23),
                        ('conv4_4', Type.Conv, 25), ('relu4_4', Type.Relu, 25),
                        ('pool4', Type.Pool),
                        ('conv5_1', Type.Conv, 28), ('relu5_1', Type.Relu, 28),
                        ('conv5_2', Type.Conv, 30), ('relu5_2', Type.Relu, 30),
                        ('conv5_3', Type.Conv, 32), ('relu5_3', Type.Relu, 32),
                        ('conv5_4', Type.Conv, 34), ('relu5_4', Type.Relu, 34),
                        ('pool5', Type.Pool))
    
        self.net = {}
        #self.net['input'] = tf.placeholder(tf.float32)
        self.net['input'] = tf.Variable(tf.zeros(img.shape, dtype=np.float32))

        layer_input = self.net['input']        
        for layer_config in net_config:
            if layer_config[1] == Type.Conv:
                weights = tf.constant(self.raw_layers[layer_config[2]][0][0][2][0][0])
                self.net[layer_config[0]] = tf.nn.conv2d(layer_input, weights, strides=[1, 1, 1, 1], padding='SAME')    
            elif layer_config[1] == Type.Relu:
                bias = self.raw_layers[layer_config[2]][0][0][2][0][1]
                bias = tf.constant(np.reshape(bias, (bias.size)))
                self.net[layer_config[0]] = tf.nn.relu(layer_input + bias)
            elif layer_config[1] == Type.Pool:
                self.net[layer_config[0]] = tf.nn.avg_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            else:
                raise Exception('invalid layer type!')
        
            layer_input = self.net[layer_config[0]]
            
