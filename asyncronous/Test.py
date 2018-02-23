import tensorflow as tf


config=tf.ConfigProto(log_device_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "7"
config.gpu_options.visible_device_list = "0, 1"

# Creates a graph.
with tf.device('/device:GPU:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with allow_soft_placement and log_device_placement set
# to True.
sess = tf.Session(config=config)
# Runs the op.
print(sess.run(c))

