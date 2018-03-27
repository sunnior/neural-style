import models
import optimizer

batch_size = 4
batch_shape = (batch_size, 256, 256, 3)
style_shape = 
vgg_data_path = ''

content_loss_weight = 
style_loss_weight = 
tv_loss_weight = 

learning_rate =
epoches = 

batch_input = tf.placeholder(tf.float32, shape=batch_shape)
batch_input_vgg = models.vgg_preprocess(batch_input)
style_input = tf.placeholder(tf.float32, shape=style_shape)
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

for epoch in range(epoches):
	while True:
		has_data, x_batch = get_next_batch(batch_size)
		if not has_data:
			break
		
		sess.run(train_step, feed_dict={batch_input:x_batch})
#I see they creat 3 vgg net. Can we just create one? all vgg nets are constant how it get optimized?

#the style net may have different size, but gram matrix can handle it. how? print the shape.

