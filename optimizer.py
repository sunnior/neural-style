import tensorflow as tf

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'


def create_content_loss(vgg_mixer_net, vgg_content_net):
	# todo assert the shape
	_, h, w, d = vgg_mixer_net[CONTENT_LAYER].get_shape()
	m = h.value * w.value
	n = d.value
	k = 1. / (2. * n**0.5 * m**0.5)
	
	content_loss = k * tf.reduce_sum(tf.pow(vgg_mixer_net[CONTENT_LAYER] - vgg_content_net[CONTENT_LAYER], 2))
	return content_loss

def create_style_loss(sess, vgg_mixer_net, vgg_style_net):
	def _gram_matrix(net):
		b, h, w, d = net.get_shape()
		size = h * w * d
		F = tf.reshape(net, (b, h * w, d))
		F_t = tf.transpose(F, perm=[0, 2, 1])
		gram = tf.matmul(F_t, F) / size.value
		print(gram.get_shape())
		return gram
	
	style_loss = 0
	for style_layer in STYLE_LAYERS:
		activation = sess.run(vgg_style_net[style_layer])
		#gram_style = _gram_matrix(tf.convert_to_tensor(activation))
		#gram_mixer = _gram_matrix(vgg_mixer_net[style_layer])

		#_, w, h = gram_style.get_shape()
		#size = w * h
		#style_loss += tf.reduce_sum(tf.pow(gram_style - gram_mixer, 2)) / size.value
		
	style_loss /= float(len(style_layers_weights))
	
	return style_loss    
