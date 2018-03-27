import models
import optimizer
import data

vgg_data_path = ''
dataset_path = ''
style_image_path = ''

batch_size = 4
batch_shape = (batch_size, 256, 256, 3)

content_loss_weight = 7.5e0
style_loss_weight = 1e2
tv_loss_weight = 2e2

learning_rate = 1e-3
epoches = 2

data.init_dataset(dataset_path, batch_shape)

batch_input = tf.placeholder(tf.float32, shape=batch_shape)
batch_input_vgg = models.vgg_preprocess(batch_input)
style_image = data.get_image(style_image_path)
style_input = tf.constants(style_image)
style_input_vgg = models.vgg_preprocess(style_input)

mixer_net = models.load_mixer_net(batch_input)
mixer_net_vgg = models.vgg_preprocess(mixer_net)

vgg_mixer_net = models.load_vgg_net(vgg_data_path, mixer_net_vgg)
vgg_style_net = models.load_vgg_net(vgg_data_path, style_input_vgg)
vgg_content_net = models.load_vgg_net(vgg_data_path, batch_input_vgg)

sess = tf.Session()

content_loss = optimizer.create_content_loss(vgg_mixer_net, vgg_content_net)
style_loss = optimizer.create_style_loss(sess, vgg_mixer_net, vgg_style_net)
tv_loss = optimizer.create_tv_loss()
#how to get tv loss what is _tensor_size

loss = content_loss_weight * content_loss + style_loss_weight * style_loss + tv_loss_weight * tv_loss

train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
sess.run(tf.global_variables_initializer())

data.set_batch_size(batch_size)
for epoch in range(epoches):
	while True:
		x_batch = data.get_next_batch()
		if x_batch.shape[0] == 0:
			break
		
		sess.run(train_step, feed_dict={batch_input:x_batch})

sess.close()
#I see they creat 3 vgg net. Can we just create one? all vgg nets are constant how it get optimized?

#the style net may have different size, but gram matrix can handle it. how? print the shape.

