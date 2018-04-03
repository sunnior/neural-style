import models
import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline

vgg_data_path = '../models/imagenet-vgg-verydeep-19.mat'

input = tf.placeholder(dtype=tf.float32, shape=(4, 3, 256, 256))
test = np.zeros((4, 3, 256, 256)) + 1

models.load_vgg_data(vgg_data_path)
net = models.load_mixer_net('mixer', input)
vgg_mixer_net = models.load_vgg_net('vgg', net)

options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
run_metadata = tf.RunMetadata()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(vgg_mixer_net['conv5_1'], feed_dict={input:test}, options=options, run_metadata=run_metadata)
sess.close()
       
tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format(show_memory=True)
with open('timeline_mixer.json', 'w') as f:
    f.write(ctf)

