import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization

in_1 = Input(shape=[10])
out_1 = Dense(5, activation='elu')(in_1)
out_1 = BatchNormalization()(out_1)
out_1 = Dense(1, activation='sigmoid')(out_1)

in_2 = Input(shape=[10])
out_2 = Dense(5, activation='elu')(in_2)
out_2 = BatchNormalization()(out_2)
out_2 = Dense(1, activation='sigmoid')(out_2)

model_1 = Model(inputs=[in_1], outputs=[out_1])
model_2 = Model(inputs=[in_2], outputs=[out_2])


data = np.random.random_sample(size=[5, 10])
p_1 = tf.placeholder(dtype=tf.float32, shape=[None, 10])
p_2 = tf.placeholder(dtype=tf.float32, shape=[None, 10])

r_1 = model_1(p_1)
r_2 = model_2(p_2)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model_2.set_weights(model_1.get_weights())
    print('---------------------------------------------------------')
    for w1, w2 in zip(model_1.get_weights(), model_2.get_weights()):
        print(np.array_equiv(w1, w2))
    print(sess.run(r_1, feed_dict={p_1:data}))
    print(sess.run(r_2, feed_dict={p_2:data}))
