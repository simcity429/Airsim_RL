import tensorflow as tf
import numpy as np
from PIL import Image
from copy import deepcopy
from PPO_Agent import PPO_Network
from PPO_Agent import PPO_Functions
from PPO_Agent import Replay
from PPO_Agent import Memory
from airsim_env import Env
from numpy.random import randint
#hyperparameters
TRAIN_NUM = 1024
RANDCONST = 100000
SEQUENCE_SIZE = 5
W = 128
H = 72
ACTION_SIZE = 3
targetY = 58
gamma = 0.99
lamb = 0.90
max_step = 600
epochs = 10
cooltime = 1024
cooltime_cnt = 0
time_horizon = 10
rand_prob = 0.0
episode = 0
score_bank = []
best_score = 0

def is_done():
    if len(score_bank) < 15:
        return False
    else:
        return (np.mean(score_bank[-10:]) >56.0)

def save_best_weight(model):
    global best_score
    if len(score_bank) > 40:
        if np.mean(score_bank[-20:]) > best_score:
            best_score = np.mean(score_bank[-20:])
            print('best score: ',best_score)
            print('saving best weight')
            model.save_best_weights()

def transform_input(responses):
    w = W
    h = H
    img1d = np.array(responses[0].image_data_float, dtype=np.float)
    img1d = np.array(np.clip(255*3*img1d, 0, 255), dtype=np.uint8)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
    image = Image.fromarray(img2d).resize((w, h)).convert('L')
    image.save("tmp.jpg")
    im_final = np.array(image, dtype=np.float32)
    im_final = ((im_final)/128)-1
    return np.float32(im_final.reshape((1, h, w, 1)))

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Make RL agent
    model = PPO_Network(SEQUENCE_SIZE, W, H, ACTION_SIZE)
    functions = PPO_Functions()
    env = Env()
    replay = Replay()
    memory = Memory()
    #set session, and initialize
    model.set_session(tf.Session(config=config), resume=True)
    try:
        for e in range(5000000):
            rand_flag = False
            done = False
            bug = False
            level = -1
            reward_sum = 0
            ayachan = randint(RANDCONST)
            if ayachan < rand_prob*RANDCONST:
                rand_flag = True
            t = 0
            time_horizon_cnt = 0
            score = 0
            observation = env.reset()
            responses = observation[0]
            quad_vel = observation[1]
            # stack history here
            try:
                img = transform_input(responses)
            except:
                print('bug')
                continue
            history = np.copy(img)
            for _ in range(SEQUENCE_SIZE - 1):
                history = np.append(history, img, axis=0)
                img = np.copy(img)
            history = np.reshape(history, (SEQUENCE_SIZE, H, W, 1))

            while not done:
                t += 1
                time_horizon_cnt += 1
                cooltime_cnt += 1
                state = [[history, quad_vel]]
                action_mean = model.action_forward(state)[0]
                tmp = model.old_action_forward(state)[0]
                if rand_flag:
                    action = functions.random_sample()
                else:
                    action = functions.sample(action_mean)
                observation, reward, done, info = env.step(action, t)
                level = info['level']
                reward_sum += reward
                if t>max_step:
                    done = True
                score += reward

                # stack history here
                responses = observation[0]
                quad_vel = observation[1]
                try:
                    img = transform_input(responses)
                except:
                    bug = True
                    break
                next_history = np.append(history[1:, :, :, :], img, axis=0)
                state = state[0]

                memory_sample = state, list(action), reward
                memory.append(memory_sample)
                print('Step %d Action %s Reward %.2f Info %s:' % (t, action, reward, info))
                print('Action mean: ', action_mean)
                history = next_history

                if done or time_horizon_cnt == time_horizon:
                    quad_pos = env.client.getMultirotorState().kinematics_estimated.position
                    if quad_pos.y_val < 1:
                        bug = True
                        break
                    time_horizon_cnt = 0
                    values = functions.calculate_value(memory, model)
                    delta = functions.calculate_delta(memory, values, gamma, done)
                    gae = functions.calculate_gae(delta, gamma * lamb)
                    oracle_values = functions.calculate_oracle_values(values, gae)
                    if len(gae) != len(memory.states):
                        replay.append_states(deepcopy(memory.states[:-1]))
                        replay.append_actions(deepcopy(memory.actions[:-1]))
                        replay.append_gae(deepcopy(gae))
                        replay.append_oracle(deepcopy(oracle_values))
                    else:
                        replay.append_states(deepcopy(memory.states))
                        replay.append_actions(deepcopy(memory.actions))
                        replay.append_gae(deepcopy(gae))
                        replay.append_oracle(deepcopy(oracle_values))
                    memory.reset()
                if cooltime <= cooltime_cnt:
                    if len(replay.states) > TRAIN_NUM:
                        cooltime_cnt = 0
                        print('In episode ', episode)
                        for k in range(epochs):
                            print('epoch: ', k)
                            al = model.optimize(replay, k)
                            if al > 1:
                                print('update fail')
                                model.saver.restore(model.sess, model.name)
                                break
                        model.update_weights()


            # done
            if bug:
                memory.reset()
                continue
            quad_pos = env.client.getMultirotorState().kinematics_estimated.position
            print('Ep %d: Step %d Score %.2f random %s' % (episode, t, quad_pos.y_val, rand_flag))
            print('cooltimecnt: ', cooltime_cnt)
            score_bank.append(quad_pos.y_val)
            if not rand_flag and len(replay.states) > TRAIN_NUM:
                with open('output_true.csv', 'a') as fd:
                    fd.write(str(quad_pos.y_val)+','+str(level)+','+str(reward_sum)+','+str(t)+'\n')
            episode += 1
            if is_done():
                model.save_weights()
                print('finally done. Congratulations')
                break
            if episode % 10 == 1:
                rand_prob *= 0.95
                model.save_weights()
                model.sess.close()
                model.set_session(tf.Session(), resume=True)
            elif episode % 50 == 3 and episode > 300:
                save_best_weight(model)

    except KeyboardInterrupt:
        env.disconnect()


