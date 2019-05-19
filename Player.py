import tensorflow as tf
import numpy as np
from PPO_Agent import PPO_Network
from PPO_Agent import PPO_Functions
from airsim_env import Env
from Simulator import transform_input

TRAIN_NUM = 512
RANDCONST = 100000
SEQUENCE_SIZE = 5
W = 128
H = 72
ACTION_SIZE = 3
targetY = 58
gamma = 0.99
lamb = 0.90
max_step = 600

score_bank = []
episode = 0

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Make RL agent
    model = PPO_Network(SEQUENCE_SIZE, W, H, ACTION_SIZE)
    functions = PPO_Functions()
    env = Env()
    #set session, and initialize
    model.set_session(tf.Session(config=config))
    try:
        for e in range(5000000):
            done = False
            bug = False
            level = -1
            reward_sum = 0
            t = 0
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
                state = [[history, quad_vel]]
                action_mean = model.action_forward(state)[0]
                action = functions.play_sample(action_mean)
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
                print('Step %d Action %s Reward %.2f Info %s:' % (t, action, reward, info))
                print('Action mean: ', action_mean)
                history = next_history
            # done
            if bug:
                continue
            quad_pos = env.client.getMultirotorState().kinematics_estimated.position
            print('Ep %d: Step %d Score %.2f' % (episode, t, quad_pos.y_val))
            score_bank.append(quad_pos.y_val)
            episode += 1

    except KeyboardInterrupt:
        env.disconnect()