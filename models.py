from enum import Enum
import tensorflow as tf
import scipy.io
import numpy as np

MEAN_PIXEL = np.array([123.68,  116.779,  103.939])
_raw_data = 0

def vgg_preprocess(image):
	image = image - MEAN_PIXEL 
	image = np.transpose(image, (0, 3, 1, 2))
	return image

def vgg_postprocess(image):
	image = np.transpose(image, (0, 2, 3, 1))
	image = image + MEAN_PIXEL
	return image

def load_vgg_data(data_path):
	global _raw_data
	_raw_data = scipy.io.loadmat(data_path)

def load_vgg_net(net_name, input):
	global _raw_data
	raw_layers = _raw_data['layers'][0]
		
	net = {}
	
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

    #net['input'] = tf.Variable(tf.zeros(input.shape, dtype=np.float32))
	net['input'] = input
	
	layer_input = net['input']
	for layer_config in net_config:
		node_name = net_name + '_' + layer_config[0]
		if layer_config[1] == Type.Conv:
			weights = tf.constant(raw_layers[layer_config[2]][0][0][2][0][0], name=node_name + '_weight_const')
			conv = tf.nn.conv2d(layer_input, weights, strides=[1, 1, 1, 1], padding='SAME', name=node_name, data_format='NCHW')
			net[layer_config[0]] = conv

		elif layer_config[1] == Type.Relu:
			bias = raw_layers[layer_config[2]][0][0][2][0][1]
			bias = np.reshape(bias, (1, bias.size, 1, 1))
			bias = tf.constant(bias, name=node_name + '_bias_const')
			net[layer_config[0]] = tf.nn.relu(tf.add(layer_input, bias, name=node_name + '_add'), name=node_name)
		elif layer_config[1] == Type.Pool:
			net[layer_config[0]] = tf.nn.avg_pool(layer_input, ksize=[1, 1, 2, 2], strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW', name=node_name)
		else:
			raise Exception('invalid layer type!')
	
		layer_input = net[layer_config[0]]
	
	return net


def load_mixer_net(name, input):

	def _conv_init_vars(input, num_filters, filter_size, transpose=False):
		_, channels, h, w = input.get_shape()
		if not transpose:
			weights_shape = [filter_size, filter_size, channels.value, num_filters]
		else:
			weights_shape = [filter_size, filter_size, num_filters, channels.value]

		#print('filter: ' + str(weights_shape))
		return tf.Variable(tf.truncated_normal(weights_shape, stddev=.1, seed=1), dtype=tf.float32)

	def _instance_norm(net):
		#_, h, w, channels = [i.value for i in net.get_shape()]
		#var_shape = [channels]
		mu, sigma_sq = tf.nn.moments(net, [2, 3], keep_dims=True)
		shift = tf.Variable(tf.zeros(net.get_shape()))
		scale = tf.Variable(tf.ones(net.get_shape()))
		# summary these variables, see their change
		epsilon = 1e-3
		normalized = (net - mu) / (sigma_sq + epsilon) ** 0.5
		return scale * normalized + shift

	def _conv_layer(name, net, num_filters, filter_size, strides, relu=True):
		#print('conv layer: ' + str(net.get_shape()))
		weights_init = _conv_init_vars(net, num_filters, filter_size)
		strides_shape = [1, 1, strides, strides]
		net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME', data_format='NCHW', name=name)
		net = _instance_norm(net)
		if relu:
			net = tf.nn.relu(net)

		print('conv: ' + str(net.get_shape()))
		return net

	def _residual_block(name, net, filter_size):
		_, channels, h, w = net.get_shape()
		#print('residual layer: ' + str(net.get_shape()))
		return tf.add(net, _conv_layer(name + '_conv', net, channels.value, filter_size, 1, relu=False), name=name)

	def _conv_transpose_layer(net, num_filters, filter_size, strides):
		weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

		batch_size, in_channels, rows, cols = [i.value for i in net.get_shape()]
		new_rows, new_cols = int(rows * strides), int(cols * strides)
		# new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

		new_shape = [batch_size, num_filters, new_rows, new_cols]
		tf_shape = tf.stack(new_shape)
		strides_shape = [1, 1, strides,strides]

		net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME', data_format='NCHW')
		net = _instance_norm(net)
		print('trans conv: ' + str(net.get_shape()))
		return tf.nn.relu(net)
	
	conv1 = _conv_layer(name + '_conv1', input, 32, 9, 1)
	conv2 = _conv_layer(name + '_conv2', conv1, 64, 3, 2)
	conv3 = _conv_layer(name + '_conv3', conv2, 128, 3, 2)
	resid1 = _residual_block(name + '_res1', conv3, 3)
	resid2 = _residual_block(name + '_res2', resid1, 3)
	resid3 = _residual_block(name + '_res3', resid2, 3)
	resid4 = _residual_block(name + '_res4', resid3, 3)
	resid5 = _residual_block(name + '_res5', resid4, 3)
	conv_t1 = _conv_transpose_layer(resid5, 64, 3, 2)
	conv_t2 = _conv_transpose_layer(conv_t1, 32, 3, 2)
	conv_t3 = _conv_layer(name + 'convend', conv_t2, 3, 9, 1, relu=False)
	# print the shape of each layer.
	preds = tf.nn.tanh(conv_t3) * 150 + 255./2
	# why scale is 150
	return preds
