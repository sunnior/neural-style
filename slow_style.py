import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import data
import models
import optimizer

vgg_data_path = '../models/imagenet-vgg-verydeep-19.mat'
dataset_path = '../dataset/train2017'
style_image_path = '../style/kandinsky.jpg'
content_image_path = '../content/tubingen.jpg'
output_path = '../output/'

batch_size = 1
batch_shape = (batch_size, 256, 256, 3)

content_loss_weight = 1e0
style_loss_weight = 1e3
tv_loss_weight = 2e2

learning_rate = 1
epoches = 2

data.init_dataset(dataset_path, batch_shape)

content_img = data.get_img(content_image_path)
content_img = data.img_fit_to(content_img)

content_input = np.expand_dims(content_img, axis=0)
content_input = models.vgg_preprocess(content_input)
content_input_vgg = tf.constant(content_input, dtype=tf.float32, name='content_input')
style_input = np.expand_dims(data.get_img(style_image_path), axis=0)
style_input = models.vgg_preprocess(style_input)
style_input_vgg = tf.Variable(style_input, dtype=tf.float32, name='style_input')



#mixer_net = models.load_mixer_net(batch_input)
#image_input = mixer_net
image_input_vgg = tf.Variable(tf.random_uniform(content_input.shape, minval=-20., maxval=20., dtype=tf.float32))

vgg_mixer_net = models.load_vgg_net(vgg_data_path, image_input_vgg)
vgg_style_net = models.load_vgg_net(vgg_data_path, style_input_vgg)
vgg_content_net = models.load_vgg_net(vgg_data_path, content_input_vgg)

#config.gpu_options.per_process_gpu_memory_fraction = 
sess = tf.Session()
sess.run(style_input_vgg.initializer)
sess.run(content_input_vgg.initializer)

style_loss = optimizer.create_style_loss(sess, vgg_mixer_net, vgg_style_net)
#content_loss = optimizer.create_content_loss(vgg_mixer_net, vgg_content_net)
content_loss = optimizer.create_content_loss_single_image(sess, vgg_mixer_net, vgg_content_net)
tv_loss = tf.image.total_variation(image_input_vgg)

loss = style_loss * style_loss_weight + content_loss * content_loss_weight# + tv_loss * tv_loss_weight
train_step = tf.train.AdamOptimizer(1).minimize(loss)


sess.run(tf.global_variables_initializer())
options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, report_tensor_allocations_upon_oom=True)
run_metadata = tf.RunMetadata()
sess.run(train_step, options=options, run_metadata=run_metadata)

with open('../run.txt', 'w') as out:
	out.write(str(run_metadata))

tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
with open('../timeline.json', 'w') as f:
    f.write(ctf)
'''
builder = tf.profiler.ProfileOptionBuilder
opts = builder(builder.time_and_memory()).order_by('micro').build()
run_metadata = tf.RunMetadata()

pctx = tf.contrib.tfprof.ProfileContext('../profile', trace_steps=[], dump_steps=[])
sess.run(tf.global_variables_initializer())

#sess.run(train_step)

for idx in range(10001):
	pctx.trace_next_step()
	pctx.dump_next_step()
	sess.run(train_step, run_metadata=run_metadata)	
	pctx.profiler.profile_operations(options=opts)
	if idx % 20 == 0:
		lossc, losss = sess.run([content_loss * content_loss_weight, style_loss * style_loss_weight])
		print("iter: " + str(idx) + " c_loss: " + str(lossc) + " s_loss: " + str(losss))

		image = sess.run(image_input_vgg)
		image = np.squeeze(models.vgg_postprocess(image))
		data.save_img(output_path + format(idx, '04d') + '.jpg', np.squeeze(image))

'''
'''
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

'''

'''
test = tf.constant(np.zeros((4, 128, 128, 3))+1, dtype=tf.float32)
test = tf.nn.conv2d(test, np.zeros((3, 3, 3, 64), dtype=float)+1, strides=[1, 1, 1, 1], padding='SAME')
test = tf.nn.avg_pool(test, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
test = tf.nn.conv2d(test, np.zeros((3, 3, 64, 64), dtype=float)+1, strides=[1, 1, 1, 1], padding='SAME')

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(test)
'''