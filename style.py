import tensorflow as tf

def get_content_loss(sess, vgg_net, content_img):
    content_layers_weights = [['conv4_3', 1.0]]
    
    sess.run(vgg_net['input'].assign(content_img))
    
    content_loss = 0
    for content_layer_weight in content_layers_weights:
        activation = sess.run(vgg_net[content_layer_weight[0]])

        _, h, w, d = activation.shape
        m = h * w
        n = d
        k = 1. / (2. * n**0.5 * m**0.5)
        
        content_loss += content_layer_weight[1] * k * tf.reduce_sum(tf.pow(vgg_net[content_layer_weight[0]] - tf.convert_to_tensor(activation), 2))
        
    content_loss /= float(len(content_layers_weights))

    return content_loss
    
def gram_matrix(x, area, depth):
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)
    return G

def style_layer_loss(a, x):
    _, h, w, d = a.get_shape()
    M = h.value * w.value
    N = d.value
    A = gram_matrix(a, M, N)
    G = gram_matrix(x, M, N)
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss

def get_style_loss(sess, vgg_net, style_img):
    style_layers_weights=[['relu1_1', 0.2], ['relu2_1', 0.2], ['relu3_1', 0.2], ['relu4_1', 0.2], ['relu5_1', 0.2]]

    sess.run(vgg_net['input'].assign(style_img))
    
    style_loss = 0
    for style_layer_weight in style_layers_weights:
        activation = sess.run(vgg_net[style_layer_weight[0]])

        style_loss += style_layer_weight[1] * style_layer_loss(tf.convert_to_tensor(activation), vgg_net[style_layer_weight[0]])
    
    style_loss /= float(len(style_layers_weights))

    return style_loss    