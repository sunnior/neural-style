import data
import models
import optimizer
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline

content_loss_weight = 1e0
style_loss_weight = 1e3
tv_loss_weight = 1e-1

vgg_data_path = '../models/imagenet-vgg-verydeep-19.mat'
content_image_path = '../content/tubingen.jpg'
style_image_path = '../style/kandinsky.jpg'
output_path = '../output/'

content_img = data.get_img(content_image_path)
content_img = data.img_fit_to(content_img)
image_shape = content_img.shape

content_img = np.expand_dims(content_img, axis=0)
content_img = models.vgg_preprocess(content_img)

style_img = np.expand_dims(data.get_img(style_image_path, img_size=image_shape), axis=0)
style_img = models.vgg_preprocess(style_img)

net_input = tf.Variable(tf.random_uniform(content_img.shape, minval=-20., maxval=20., dtype=tf.float32))

#style_input_vgg = tf.constant(style_input, dtype=tf.float32, name='style_input')

models.load_vgg_data(vgg_data_path)
vgg_net = models.load_vgg_net('vgg', net_input)

sess = tf.Session()

content_loss = optimizer.create_content_loss(sess, vgg_net, content_img)
style_loss = optimizer.create_style_loss(sess, vgg_net, style_img)
tv_loss = tf.image.total_variation(net_input)

loss = content_loss_weight * content_loss + style_loss_weight * style_loss + tv_loss_weight * tv_loss

run_metadata = tf.RunMetadata()
options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
train = tf.train.AdamOptimizer(1).minimize(loss)

sess.run(tf.global_variables_initializer())

#sess.run(train, options=options, run_metadata=run_metadata)

for i in range(10001):
    sess.run(train)
    
    if i % 20 == 0:
        closs, sloss, tloss = sess.run([content_loss, style_loss, tv_loss])
        print("iter: " + str(i) + " closs: " + str(content_loss_weight * closs) + " sloss: " + str(style_loss_weight * sloss) + " tloss: " + str(tv_loss_weight * tloss))
        image = sess.run(vgg_net['input'])
        image = np.squeeze(models.vgg_postprocess(image))
        data.save_img(output_path + format(i, '04d') + '.jpg', np.squeeze(image))

'''       
tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format(show_memory=True)
with open('../timeline.json', 'w') as f:
    f.write(ctf)
'''