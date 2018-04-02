import tensorflow as tf

STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'conv4_2'


def create_content_loss(sess, net, input):
	# todo assert the shape
	_, h, w, d = net[CONTENT_LAYER].get_shape()
	m = h.value * w.value
	n = d.value
	k = 1. / (2. * n**0.5 * m**0.5)

	sess.run(net['input'].assign(input))
	activation = sess.run(net[CONTENT_LAYER])
	content_loss = k * tf.reduce_sum(tf.pow(net[CONTENT_LAYER] - activation, 2))
	return content_loss

def create_content_loss_single_image(sess, vgg_mixer_net, vgg_content_net):
	_, h, w, d = vgg_mixer_net[CONTENT_LAYER].get_shape()
	s = (h * w * d).value
	k = 1. / (2. * s**0.5)
	activation = sess.run(vgg_content_net[CONTENT_LAYER])
	content_loss = k * tf.reduce_sum(tf.pow(vgg_mixer_net[CONTENT_LAYER] - activation, 2))
	return content_loss

def create_style_loss(sess, net, input):
	def _gram_matrix(layer):
		b, d, h, w = layer.get_shape()
		size = h * w * d
		F = tf.reshape(layer, (b, d, h * w))
		F_t = tf.transpose(F, perm=[0, 2, 1])
		gram = tf.matmul(F, F_t) / size.value
		return gram
	
	sess.run(net['input'].assign(input))
	
	style_loss = 0
	for style_layer in STYLE_LAYERS:
		activation = sess.run(net[style_layer])
		gram_style = _gram_matrix(tf.constant(activation))
		gram_mixer = _gram_matrix(net[style_layer])

		_, w, h = gram_style.get_shape()
		s = (w * h).value
		k = 1. / (2. * s**0.5)
		style_loss += k * tf.reduce_sum(tf.pow(gram_style - gram_mixer, 2))
		
	style_loss /= float(len(STYLE_LAYERS))
	
	return style_loss    
