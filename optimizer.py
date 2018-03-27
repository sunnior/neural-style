STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
CONTENT_LAYER = 'relu4_2'

def create_content_loss(vgg_mixer_net, vgg_content_net):
	#todo assert the shape
	_, h, w, d = vgg_mixer_net.get_shape()
    m = h.value * w.value
    n = d.value
    k = 1. / (2. * n**0.5 * m**0.5)
      
    content_loss = k * tf.reduce_sum(tf.pow(vgg_mixer_net[CONTENT_LAYER] - vgg_content_net[CONTENT_LAYER], 2))
	return content_loss

def create_style_loss(sess, vgg_mixer_net, vgg_style_net):

	def _gram_matrix(x, area, depth):
		F = tf.reshape(x, (area, depth))
		G = tf.matmul(tf.transpose(F), F)
		return G

	def _style_layer_loss(a, x):
		_, h, w, d = a.get_shape()
		M = h.value * w.value
		N = d.value
		A = gram_matrix(a, M, N)
		G = gram_matrix(x, M, N)
		loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
		return loss
	    
    style_loss = 0
    for style_layer in STYLE_LAYERS:
        activation = sess.run(vgg_style_net[style_layer])
        style_loss += style_layer_loss(tf.convert_to_tensor(activation), vgg_mixer_net[style_layer])
    
    style_loss /= float(len(style_layers_weights))

    return style_loss    