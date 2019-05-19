import time
import numpy as np
import airsim
import config
from time import sleep

clockspeed = 1
timeslice = 0.5 / clockspeed
goalY = 57
outY = -1.5
floorZ = 1.18
COLISION_THRESHOLD = 2
# level = [0]
goals = [7, 17, 27.5, 45, 57]
ACTION = ['00', '+x', '+y', '+z', '-x', '-y', '-z']

class Env:
    def __init__(self):
        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.action_size = 3
        self.level = 0

    def reset(self):
        self.level = 0
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # my takeoff
        self.client.simPause(False)
        self.client.moveByVelocityAsync(0, 0, -1, 2 * timeslice).join()
        self.client.moveByVelocityAsync(0, 0, 0, 0.1 * timeslice).join()
        self.client.hoverAsync().join()
        self.client.simPause(True)
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])
        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        observation = [responses, quad_vel]
        return observation

    def step(self, quad_offset, t):
        # move with given velocity
        self.client.simPause(False)

        has_collided = False
        landed = False
        self.client.moveByVelocityAsync(quad_offset[0], quad_offset[1], quad_offset[2], timeslice)
        start_time = time.time()
        i=0
        while time.time() - start_time < timeslice:
            sleep(0.05)
            # get quadrotor states
            quad_pos = self.client.getMultirotorState().kinematics_estimated.position

            # decide whether collision occured
            collided = self.client.simGetCollisionInfo().has_collided
            landed = quad_pos.y_val > 10 and self.client.getMultirotorState().landed_state == airsim.LandedState.Landed
            landed = landed or quad_pos.z_val > floorZ
            collision = collided or landed
            if collision:
                i+=1
            if i > COLISION_THRESHOLD:
                has_collided = True
        if i > 0:
            print('colision cnt: ',i)
        self.client.simPause(True)
        # observe with depth camera
        responses = self.client.simGetImages([airsim.ImageRequest(1, airsim.ImageType.DepthVis, True)])

        # get quadrotor states
        quad_pos = self.client.getMultirotorState().kinematics_estimated.position
        quad_vel = self.client.getMultirotorState().kinematics_estimated.linear_velocity
        # decide whether done
        dead = has_collided or quad_pos.y_val <= outY
        done = dead or quad_pos.y_val >= goalY

        # compute reward
        if done and t<2:
            done=False
        reward = self.compute_reward(quad_pos, quad_vel, dead)
        if reward == config.reward['slow']:
            done = True
        # log info
        info = {}
        info['Y'] = quad_pos.y_val
        info['level'] = self.level
        if landed:
            info['status'] = 'landed'
        elif has_collided:
            info['status'] = 'collision'
        elif quad_pos.y_val <= outY:
            info['status'] = 'out'
        elif quad_pos.y_val >= goalY:
            info['status'] = 'goal'
        elif reward == config.reward['slow']:
            info['status'] = 'slow'
        else:
            info['status'] = 'going'
        quad_vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val])
        observation = (responses, quad_vel)
        return observation, reward, done, info

    def compute_reward(self, quad_pos, quad_vel, dead):
        speed_limit = 0.3
        vel = np.array([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val], dtype=np.float)
        speed = np.linalg.norm(vel)
        if dead:
            reward = config.reward['dead']
        elif quad_pos.y_val >= goals[self.level]:
            self.level += 1
            # reward = config.reward['forward'] * (1 + self.level / len(goals))
            reward = config.reward['goal'] * (1 + self.level / len(goals))
        elif speed < speed_limit:
            reward = config.reward['slow']
        # elif vel[1] > 0:
        #     reward = float(vel[1]) * 0.1
        else:
            reward = float(vel[1]) * 0.1
        return reward
    
    def disconnect(self):
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
        print('Disconnected.')