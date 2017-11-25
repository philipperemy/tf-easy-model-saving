import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from easy_model_saving import model_saver

CHECKPOINT_DIR = 'chkpt'


def graph():
    x_ = tf.placeholder(tf.float32, [None, 20])
    slim.fully_connected(inputs=x_, num_outputs=1)


with tf.Session() as sess:
    graph()

    last_step = model_saver.restore_graph_variables(CHECKPOINT_DIR)
    if last_step == 0:
        print('Did not find any weights.')
        model_saver.initialize_graph_variables()
    else:
        print('Restore successful.')

    trainable_variables = tf.trainable_variables()
    print(trainable_variables)
    slim_fc_weights = trainable_variables[0]

    sum_weights = np.sum(sess.run(slim_fc_weights).flatten())

    print('Original sum of weights = {0}'.format(sum_weights))

    saver = model_saver.Saver(CHECKPOINT_DIR)

    new_weight_values = tf.constant(value=np.random.uniform(size=(20, 1)),
                                    dtype=tf.float32)

    sess.run(tf.assign(slim_fc_weights, new_weight_values))

    new_sum_weights = np.sum(sess.run(slim_fc_weights).flatten())

    print('Sum of weights after update = {0}'.format(new_sum_weights))
    print('Values should be different.')

    saver.save(global_step=1)

    print('Restart this script now that CHECKPOINT_DIR contains a checkpoint.')
