import tensorflow as tf

# 1.7159 tanh(2/3 * x)

# batch size, time frame, feature
x = tf.placeholder(dtype = tf.float32, shape = [200, 6, 10])
batch_size = x.get_shape()[0]

# feature 10 to 64
w1 = tf.get_variable('w1',
                     shape = [10, 64],
                     dtype = tf.float32,
                     initializer = tf.contrib.layers.xavier_initializer(),
                     regularizer = tf.contrib.layers.l2_regularizer(0.0001),
                     trainable = True
                     )

b1 = tf.get_variable('b1',
                     shape = [64, ],
                     dtype = tf.float32,
                     initializer = tf.zeros_initializer(),
                     trainable = True)

# [None, 6, 64] + [64]
l1 = tf.tensordot(x, w1, axes = [2, 0]) + 100 * b1
l1 = 1.7159 * tf.tanh((2/3) * l1)

# layer 2
w2 = tf.get_variable('w2',
                     shape = [64, 64],
                     dtype = tf.float32,
                     initializer = tf.contrib.layers.xavier_initializer(),
                     regularizer = tf.contrib.layers.l2_regularizer(0.0001),
                     trainable = True
                     )

b2 = tf.get_variable('b2',
                     shape = [64, ],
                     dtype = tf.float32,
                     initializer = tf.zeros_initializer(),
                     trainable = True)

l2 = tf.tensordot(l1, w2, axes = [2, 0]) + 100 * b2
l2 = 1.7159 * tf.tanh((2/3) * l2)

# layer 3
w3 = tf.get_variable('w3',
                     shape = [64, 64],
                     dtype = tf.float32,
                     initializer = tf.contrib.layers.xavier_initializer(),
                     regularizer = tf.contrib.layers.l2_regularizer(0.0001),
                     trainable = True
                     )

b3 = tf.get_variable('b3',
                     shape = [64, ],
                     dtype = tf.float32,
                     initializer = tf.zeros_initializer(),
                     trainable = True)

l3 = tf.tensordot(l2, w3, axes = [2, 0]) + 100 * b3
l3 = 1.7159 * tf.tanh((2/3) * l3)

# layer mu
w_mu = tf.get_variable('w_mu',
                     shape = [64, 64],
                     dtype = tf.float32,
                     initializer = tf.contrib.layers.xavier_initializer(),
                     regularizer = tf.contrib.layers.l2_regularizer(0.0001),
                     trainable = True
                     )

b_mu = tf.get_variable('b_mu',
                     shape = [64, ],
                     dtype = tf.float32,
                     initializer = tf.zeros_initializer(),
                     trainable = True)

mu = tf.tensordot(l3, w_mu, axes = [2, 0]) + 100 * b_mu

# layer sigma
w_sigma = tf.get_variable('w_sigma',
                     shape = [64, 64],
                     dtype = tf.float32,
                     initializer = tf.contrib.layers.xavier_initializer(),
                     regularizer = tf.contrib.layers.l2_regularizer(0.0001),
                     trainable = True
                     )

b_sigma = tf.get_variable('b_sigma',
                     shape = [64, ],
                     dtype = tf.float32,
                     initializer = tf.zeros_initializer(),
                     trainable = True)

sigma = tf.tensordot(l3, w_sigma, axes = [2, 0]) + 100 * b_sigma
sigma = tf.nn.softplus(sigma)

# lstm_mu
with tf.variable_scope('lstm_mu'):
    lstm_mu = tf.contrib.rnn.LSTMCell(64)
    lstm_mu_zero_state = lstm_mu.zero_state(batch_size, tf.float32)
    lstm_mu_hs, lstm_mu_last = tf.nn.dynamic_rnn(lstm_mu,
                                                 mu,
                                                 dtype = tf.float32,
                                                 initial_state = lstm_mu_zero_state)

    # lstm_mu_out
    lstm_mu_w = tf.get_variable('lstm_mu_w',
                                shape = [64, 64],
                                dtype = tf.float32,
                                initializer = tf.contrib.layers.xavier_initializer(),
                                regularizer = tf.contrib.layers.l2_regularizer(0.0001),
                                trainable = True
                                )

    lstm_mu_b = tf.get_variable('lstm_mu_b',
                                shape = [64, ],
                                dtype = tf.float32,
                                initializer = tf.zeros_initializer(),
                                trainable = True)

    lstm_mu_out = tf.tensordot(lstm_mu_hs, lstm_mu_w, axes = [2, 0]) + 100 * lstm_mu_b

# lstm_sigma
with tf.variable_scope('lstm_sigma'):
    lstm_sigma = tf.contrib.rnn.LSTMCell(64)
    lstm_sigma_zero_state = lstm_sigma.zero_state(batch_size, tf.float32)
    lstm_sigma_hs, lstm_sigma_last = tf.nn.dynamic_rnn(lstm_sigma,
                                                       sigma,
                                                       dtype = tf.float32,
                                                       initial_state = lstm_sigma_zero_state)

    # lstm_sigma_out
    lstm_sigma_w = tf.get_variable('lstm_sigma_w',
                                shape = [64, 64],
                                dtype = tf.float32,
                                initializer = tf.contrib.layers.xavier_initializer(),
                                regularizer = tf.contrib.layers.l2_regularizer(0.0001),
                                trainable = True
                                )

    lstm_sigma_b = tf.get_variable('lstm_sigma_b',
                                shape = [64, ],
                                dtype = tf.float32,
                                initializer = tf.zeros_initializer(),
                                trainable = True)

    lstm_sigma_out = tf.tensordot(lstm_sigma_hs, lstm_sigma_w, axes = [2, 0]) + 100 * lstm_sigma_b
    lstm_sigma_out = tf.nn.softplus(lstm_sigma_out)

# assume cov = 0 and set reserve term to avoid divergence
# and get kl_divergence sym cost
reserve = 1e-8

vae_lstm_loss1 = (1/2) * (tf.log(reserve + tf.square(lstm_sigma_out[:-1,:,:] / (reserve + sigma[1:,:,:])))
                          + (tf.square(sigma[1:,:,:]) + tf.square(mu[1:,:,:] - lstm_mu_out[:-1,:,:]))
                          / (reserve + tf.square(lstm_sigma_out[:-1,:,:])))

vae_lstm_loss2 = (1/2) * (tf.log(reserve + tf.square(sigma[1:,:,:] / (reserve + lstm_sigma_out[:-1,:,:])))
                          + (tf.square(lstm_sigma_out[:-1,:,:]) + tf.square(lstm_mu_out[:-1,:,:] - mu[1:,:,:]))
                          / (reserve + tf.square(sigma[1:,:,:])))

vae_lstm_loss = (1/2) * tf.reduce_mean(vae_lstm_loss1 + vae_lstm_loss2)


# dist [None, 6, 64]
z = mu + tf.sqrt(sigma) * tf.random_normal(shape = sigma.get_shape(), mean = 0, stddev = 1.0, dtype = tf.float32)
lstm_z = lstm_mu_out + tf.sqrt(lstm_sigma_out) * tf.random_normal(shape = lstm_sigma_out.get_shape(), mean = 0, stddev = 1.0, dtype = tf.float32)
lstm_z = tf.concat([tf.expand_dims(z[0,:,:], 0), lstm_z[:-1,:,:]], axis = 0)
comp_z = (z + lstm_z)/2

# layer reverse 3
w3_reverse = tf.get_variable('w3_reverse',
                             shape = [64, 64],
                             dtype = tf.float32,
                             initializer = tf.contrib.layers.xavier_initializer(),
                             regularizer = tf.contrib.layers.l2_regularizer(0.0001),
                             trainable = True
                             )

b3_reverse = tf.get_variable('b3_reverse',
                             shape = [64, ],
                             dtype = tf.float32,
                             initializer = tf.zeros_initializer(),
                             trainable = True)

l3_reverse = tf.tensordot(comp_z, w3_reverse, axes = [2, 0]) + 100 * b3_reverse
l3_reverse = 1.7159 * tf.tanh((2/3) * l3_reverse)

# layer reverse 2
w2_reverse = tf.get_variable('w2_reverse',
                             shape = [64, 64],
                             dtype = tf.float32,
                             initializer = tf.contrib.layers.xavier_initializer(),
                             regularizer = tf.contrib.layers.l2_regularizer(0.0001),
                             trainable = True
                             )

b2_reverse = tf.get_variable('b2_reverse',
                             shape = [64, ],
                             dtype = tf.float32,
                             initializer = tf.zeros_initializer(),
                             trainable = True)

l2_reverse = tf.tensordot(l3_reverse, w2_reverse, axes = [2, 0]) + 100 * b2_reverse
l2_reverse = 1.7159 * tf.tanh((2/3) * l2_reverse)

# layer reverse 1
w1_reverse = tf.get_variable('w1_reverse',
                             shape = [64, 10],
                             dtype = tf.float32,
                             initializer = tf.contrib.layers.xavier_initializer(),
                             regularizer = tf.contrib.layers.l2_regularizer(0.0001),
                             trainable = True
                             )

b1_reverse = tf.get_variable('b1_reverse',
                             shape = [10, ],
                             dtype = tf.float32,
                             initializer = tf.zeros_initializer(),
                             trainable = True)

# [None, 6, 10]
x_hat = tf.tensordot(l2_reverse, w1_reverse, axes = [2, 0]) + 100 * b1_reverse

# costs
reconstruct_error = (1/2) * tf.reduce_mean(tf.square(x_hat - x))
kl_divergence = (-1/2) * tf.reduce_mean(1 + tf.log(1e-8 + tf.square(sigma)) - tf.square(mu) - tf.square(sigma))
reg_loss = tf.losses.get_regularization_loss()
total_cost = reconstruct_error + vae_lstm_loss + kl_divergence + reg_loss

optimizer = tf.train.AdamOptimizer(0.001)
train = optimizer.minimize(total_cost)


import numpy as np

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

feed_x = np.ones([200, 6, 10])
for i in range(1000):
    _, cost = sess.run([train, total_cost], feed_dict = {x: feed_x})
    print(i, ':', '{:.5f}'.format(cost))