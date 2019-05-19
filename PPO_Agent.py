import itertools
import random
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, \
    Conv2D, TimeDistributed, Flatten, GRU, MaxPooling2D, Concatenate, Lambda, ELU, Activation, Add
from tensorflow.keras.backend import clip, constant
import matplotlib.pyplot as plt
from copy import deepcopy
SAMPLE_SIGMA = 0.1
PLAY_SIGMA = 0.1
CAL_SIGMA = 0.1
CAL_COV = CAL_SIGMA**2
SAMPLE_COV = SAMPLE_SIGMA**2
PLAY_COV = PLAY_SIGMA**2
EPSILON = 0.2
batch_size = 64
TRAIN_NUM = 1024
reward_list = []

SMOOTH_NUM = 100
def smooth(l):
    if len(l) < SMOOTH_NUM:
        return l
    tmp = []
    current_sum = 0
    for i in range(len(l)):
        current = l[i]
        current_sum += current
        tmp.append(current_sum/(i+1))
        if i == SMOOTH_NUM-2:
            break
    for i in range(SMOOTH_NUM-1, len(l)):
        tmp.append(sum(l[i-(SMOOTH_NUM-1):i+1])/SMOOTH_NUM)
    l = tmp
    return l

def plotting(l):
    length = len(l)
    index = list(range(length))
    plt.plot(index, smooth(l))
    plt.savefig('stat.png')


class Memory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def append(self, memory_sample):
        #0: prev_s, 1: action, 2: r
        self.states.append(memory_sample[0])
        self.actions.append(memory_sample[1])
        self.rewards.append(memory_sample[2])

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []

class Replay():
    def __init__(self):
        self.max_len = 1100
        self.action_size = 3
        self.states = []
        self.actions = []
        self.action_probs = []
        self.gae = []
        self.oracle_values = []

    def calculate_log_prob(self, m, x):
        d = self.action_size
        loga = (-1 / 2) *(1/CAL_COV)* np.sum((x - m) ** 2, axis=1)
        logb = np.log(np.sqrt(((2 * np.pi) ** d)*(CAL_COV)**d))
        return loga-logb

    def append_gae(self, l):
        if len(l) + len(self.gae) > self.max_len:
            self.gae = self.gae[len(l):]
            self.gae += l
        else:
            self.gae += l
        return

    def append_oracle(self, l):
        if len(l) + len(self.oracle_values) > self.max_len:
            self.oracle_values = self.oracle_values[len(l):]
            self.oracle_values += l
        else:
            self.oracle_values += l
        return

    def append_states(self, l):
        if len(l) + len(self.states) > self.max_len:
            self.states = self.states[len(l):]
            self.states += l
        else:
            self.states += l
        return

    def append_actions(self, l):
        if len(l) + len(self.actions) > self.max_len:
            self.actions = self.actions[len(l):]
            self.actions += l
        else:
            self.actions += l
        return

    def refresh(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.gae = []
        self.oracle_values = []
        return


class PPO_Functions():
    def calculate_value(self, memory, model):
        states = memory.states
        values = np.array(model.critic_forward(states))
        values = values.reshape((-1))
        return values

    def calculate_delta(self, memory, values, gamma, done):
        delta = []
        rewards = memory.rewards
        length = len(values)
        for i in range(length):
            if i < length -1:
                delta.append(rewards[i] + gamma*values[i+1] - values[i])
            else:
                if done:
                    delta.append(rewards[i] - values[i])
        return delta

    def calculate_gae(self, delta, r):
        length = len(delta)
        gae = []
        running = 0
        for i in reversed(range(length)):
            running = delta[i] + r*running
            gae.append(running)
        gae.reverse()
        return gae

    def calculate_oracle_values(self, values, gae):
        length = len(gae)
        oracle_values = np.zeros((length))
        for i in range(length):
            oracle_values[i] = values[i] + gae[i]
        return list(oracle_values)

    def sample(self, m):
        return np.random.normal(m, SAMPLE_SIGMA)

    def play_sample(self, m):
        return np.random.normal(m, PLAY_SIGMA)

    def random_sample(self):
        m = np.random.uniform(low=-1, high=1, size=[3])
        m = m + np.array([0, 2, 0])
        return np.random.normal(m, SAMPLE_SIGMA)

class PPO_Network():
    def __init__(self, seq_size, w, h, action_size):
        print('init called')
        self.seq_size = seq_size
        self.w = w
        self.h = h
        self.action_size = action_size
        self.epsilon = EPSILON

        self.history_input = tf.placeholder(dtype=tf.float32, shape=[None, seq_size, self.h, self.w, 1])
        self.vel_input = tf.placeholder(dtype=tf.float32, shape=[None, 3])

        self.old_history_input = tf.placeholder(dtype=tf.float32, shape=[None, seq_size, self.h, self.w, 1])
        self.old_vel_input = tf.placeholder(dtype=tf.float32, shape=[None, 3])

        self.actor, self.critic, self.whole = self.build_models()

        self.old_actor, _, self.old_whole = self.build_models()

        self.action_out = self.actor([self.history_input, self.vel_input])
        self.value_out = self.critic([self.history_input, self.vel_input])

        self.old_action_out = self.old_actor([self.old_history_input, self.old_vel_input])

        self.actor_loss, self.opt_actor = self.build_actor_optimizer()
        self.critic_loss, self.opt_critic = self.build_critic_optimizer()
        self.saver = tf.train.Saver()
        self.name = "model.ckpt"
        return

    def save_weights(self):
        self.saver.save(self.sess, self.name)

    def save_best_weights(self):
        self.saver.save(self.sess, 'best_model.ckpt')

    def set_session(self, sess, resume=True):
        self.sess = sess
        if resume:
            self.saver.restore(sess, self.name)
            print('successfully  restored')
        else:
            if os.path.isfile("./output_true.csv"):
                os.remove("./output_true.csv")
            sess.run(tf.global_variables_initializer())
            self.update_weights()
            self.whole.summary()
            print('initialized')

    def set_playing_session(self, sess):
        self.sess = sess
        self.saver.restore(sess, 'best_model.ckpt')
        print('successfully restored best weight')

    def build_models(self):
        min_action = -1.5
        max_action = 1.5
        initializer_1 = tf.keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
        action_constant_1 = constant([0, 1, 0])
        action_constant_2 = constant([1, 0.5, 1])
        action_constant_3 = constant([0,0.5,0])
        in_history = Input(shape=[self.seq_size, self.h, self.w, 1]) # batch, sequence_size, h, w, 1
        in_vel = Input(shape=[self.action_size,]) # batch, action_size
        image_process = BatchNormalization()(in_history)
        image_process = TimeDistributed(
            Conv2D(16, (3, 3), activation='elu', padding='same', kernel_initializer='he_normal'))(image_process)
        #72 128
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #70 126
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #68 124
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        #34 62
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #32 60
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #30 58
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        #15 29
        image_process = TimeDistributed(Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #13 27
        image_process = TimeDistributed(Conv2D(32, (4, 4), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #10 24
        image_process = TimeDistributed(MaxPooling2D((2, 2)))(image_process)
        #5 12
        image_process = TimeDistributed(Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        #3 10
        image_process = TimeDistributed(Conv2D(8, (1, 1), activation='elu', kernel_initializer='he_normal'))(
            image_process)
        image_process = TimeDistributed(Flatten())(image_process)
        image_process = GRU(48, kernel_initializer='he_normal', use_bias=False)(image_process)
        image_process = BatchNormalization()(image_process)
        image_process = Activation('tanh')(image_process)

        # vel process
        vel_process = Dense(48, kernel_initializer='he_normal', use_bias=False)(in_vel)
        vel_process = BatchNormalization()(vel_process)
        vel_process = Activation('tanh')(vel_process)



        #add
        shared = Add()([image_process, vel_process])

        #actor
        action = Dense(32, kernel_initializer='he_normal')(shared)
        action = BatchNormalization()(action)
        action = ELU()(action)
        action = Dense(32, kernel_initializer='he_normal')(action)
        action = BatchNormalization()(action)
        action = ELU()(action)
        action = Dense(self.action_size, kernel_initializer=initializer_1)(action)
        action = Lambda(lambda x:clip(x, min_action, max_action))(action)
        action = Lambda(lambda x:x+action_constant_1)(action)
        action = Lambda(lambda x:x*action_constant_2)(action)
        action = Lambda(lambda x:x+action_constant_3)(action)


        #critic
        value = Dense(32,kernel_initializer='he_normal')(shared)
        value = BatchNormalization()(value)
        value = ELU()(value)
        value = Dense(32, kernel_initializer='he_normal')(value)
        value = BatchNormalization()(value)
        value = ELU()(value)
        value = Dense(1)(value)

        actor = Model(inputs=[in_history, in_vel], outputs=action)
        critic = Model(inputs=[in_history, in_vel], outputs=value)
        whole = Model(inputs=[in_history, in_vel], outputs=[action, value])

        return actor, critic, whole

    def calculate_log_prob(self, m, x):
        loga = (-1 / (2*(CAL_SIGMA**2))) * ((x - m) ** 2)
        logb = tf.log(tf.sqrt(2 * np.pi*((CAL_SIGMA)**2)))
        return loga - logb

    def build_actor_optimizer(self):
        actions = tf.placeholder(dtype=tf.float32, shape=(None, self.action_size))
        self.actions = actions
        p_olds = tf.placeholder(dtype=tf.float32, shape=(None, 3))
        self.p_olds = p_olds
        gae = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.gae = gae
        p_nows = self.calculate_log_prob(self.action_out, actions)
        r = tf.exp(p_nows - p_olds)
        self.r = r # for debug
        a = gae * r
        b = gae * tf.clip_by_value(r, 1 - self.epsilon, 1 + self.epsilon)
        tmp = tf.minimum(a, b)
        loss = -tf.reduce_mean(tmp)
        opt = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(loss, var_list=self.actor.trainable_variables)
        return loss, opt

    def build_critic_optimizer(self):
        oracle_values = tf.placeholder(dtype=tf.float32, shape=[None, ])
        self.oracle_values = oracle_values
        predicted_values = tf.reshape(self.value_out, [-1, ])
        loss = tf.reduce_mean((predicted_values - oracle_values) ** 2)
        opt = tf.train.AdamOptimizer(learning_rate=2.5e-4).minimize(loss, var_list=self.actor.trainable_variables)
        return loss, opt

    def action_forward(self, state):
        sess = self.sess
        state = np.array(state)
        history = np.stack(state[:, 0], axis=0)
        vel = np.stack(state[:, 1], axis=0)
        action = sess.run(self.action_out, feed_dict={self.history_input:history, self.vel_input:vel})
        return action

    def old_action_forward(self, state):
        sess = self.sess
        state = np.array(state)
        history = np.stack(state[:, 0], axis=0)
        vel = np.stack(state[:, 1], axis=0)
        action = sess.run(self.old_action_out, feed_dict={self.old_history_input:history, self.old_vel_input:vel})
        return action

    def critic_forward(self, state):
        sess = self.sess
        state = np.array(state)
        history = np.stack(state[:, 0], axis=0)
        vel = np.stack(state[:, 1], axis=0)
        value = sess.run(self.value_out, feed_dict={self.history_input:history, self.vel_input:vel})
        return value

    def _calculate_log_prob(self, m, x):
        loga = (-1 / (2 * (CAL_SIGMA ** 2))) * ((x - m) ** 2)
        logb = np.log(np.sqrt(2 * np.pi * ((CAL_SIGMA) ** 2)))
        return loga - logb

    def calculate_action_probs(self, states, actions):
        old_action_mean = None
        for i in range(len(states)//batch_size):
            if i < (len(states)//batch_size -1):
                tmp = self.old_action_forward(states[i*batch_size:(i+1)*batch_size])
            else:
                tmp = self.old_action_forward(states[i * batch_size:])
            if old_action_mean is None:
                old_action_mean = tmp
            else:
                old_action_mean = np.concatenate((old_action_mean, tmp), axis=0)
        return self._calculate_log_prob(old_action_mean, actions)


    def optimize(self, replay, k):
        states = replay.states
        actions = replay.actions
        gae = replay.gae
        oracle_values = replay.oracle_values
        oracle_values = np.array(oracle_values, dtype=np.float32)
        states = np.array(states)
        actions = np.array(actions, dtype=np.float32)
        gae = np.array(gae, dtype=np.float32)
        length = gae.shape[0]
        if length > 1:
            m = np.mean(gae)
            s = np.std(gae)
            gae = (gae - m) / s
        else:
            m = np.mean(gae)
            gae = gae - m

        gae = np.reshape(gae, [-1, 1])
        idx_arr = np.arange(length)
        np.random.shuffle(idx_arr)
        sum_actor_loss = 0
        sum_critic_loss = 0
        if length > TRAIN_NUM:
            train_num = TRAIN_NUM
        else:
            train_num = length
        for i in range(train_num//batch_size):
            _oracle_values = oracle_values[idx_arr[i*batch_size:(i+1)*batch_size]]
            _states = states[idx_arr[i*batch_size:(i+1)*batch_size]]
            _actions = actions[idx_arr[i*batch_size:(i+1)*batch_size]]
            _p_olds = self.calculate_action_probs(_states, _actions)
            _gae = gae[idx_arr[i*batch_size:(i+1)*batch_size]]
            _history = np.float32(np.stack(_states[:, 0], axis=0))
            _vel = np.float32(np.stack(_states[:, 1], axis=0))
            #actor update
            a = self.sess.run([self.actor_loss, self.opt_actor, self.r],
                              feed_dict={self.history_input:_history, self.vel_input:_vel,
                                         self.actions:_actions, self.p_olds:_p_olds, self.gae:_gae})
            #critic update
            c = self.sess.run([self.critic_loss, self.opt_critic],
                              feed_dict={self.history_input:_history, self.vel_input:_vel,
                                         self.oracle_values:_oracle_values})
            actor_loss = a[0]
            critic_loss = c[0]
            print('r: ', a[2][:10])
            if sum_actor_loss == 0:
                sum_actor_loss = actor_loss
            else:
                sum_actor_loss = sum_actor_loss + actor_loss
            if sum_critic_loss == 0:
                sum_critic_loss = critic_loss
            else:
                sum_critic_loss = sum_critic_loss + critic_loss
        sum_actor_loss = sum_actor_loss/(length//batch_size)
        sum_critic_loss = sum_critic_loss/(length//batch_size)
        print('actor loss, critic loss: ', sum_actor_loss, sum_critic_loss)
        return sum_actor_loss

    def update_weights(self):
        for i, j in zip(self.old_whole.trainable_variables, self.whole.trainable_variables):
            self.sess.run(tf.assign(i, j))
        return


def preprocess(obj):
    return np.array(obj, dtype=np.float32)

def sample_random_action(action_size):
    return np.random.uniform(-1, 1, (action_size, ))


