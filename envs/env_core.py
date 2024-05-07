import numpy as np
import pygame
import cv2
# import matplotlib.pyplot as plt
from math import *
from scipy.linalg import LinAlgError

class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self, fix=False, render=False, num=0):
        self.fixed = fix
        self.render = render
        self.render_num = num
        self.done = False

        self.worst_target = None
        self.worst_value = 1e8

        self.total_num_steps = 0
        self.target_num = 3  # 可调
        self.agent_num = 6  # 可调
        self.obs_dim = 3 * (self.agent_num - 1) + 3 * self.target_num
        self.action_dim = 3
        self.max_dxy = 0.2  # UAV moving ability, km
        self.step_counter = 0
        self.pos_delta = 1e-2
        self.v_delta = 1e-4
        self.active_gamma = 4.
        self.passive_gamma = 2.
        self.personal_weight = 1.
        self.max_step_counter = 300

        self.render_interval = 10

        self.uu_collision_d = 0.0025  # 50m
        self.ut_collision_d = 0.04  # 200m
        self.ut_final_d = 0.04
        self.uu_collision_penalty = 4000.
        self.ut_collision_penalty = 400.
        self.C_R = 1e13
        self.C_theta = 1e13
        self.C_passive = 1e14
        self.large = 1e30
        self.UAV_exploration_distance = 0.25  # 500m
        self.exploration_reward_base = 3.

        # self.target_rmin = 18.  # 15.
        # self.target_rmax = 22.  # 25.
        # self.target_tmin = 0. + pi / 5  # + pi / 6  # - pi / 4
        # self.target_tmax = pi / 2 - pi / 5  # - pi / 6  # 3 * pi / 4
        # self.targetv_rmin = 0.09  # 0.07
        # self.targetv_rmax = 0.11  # 0.13
        # self.targetv_tmin = pi + pi / 5  # + pi / 6  # + pi / 6  # 3 * pi / 4
        # self.targetv_tmax = 3 * pi / 2 - pi / 5  # - pi / 6  # - pi / 6  # 7 * pi / 4
        # self.target_tv_bias = pi / 3
        # # self.agent_l = 10

        self.target_rmin = 15.  # 15.
        self.target_rmax = 25.  # 25.
        self.target_tmin = 0.  # + pi / 6  # - pi / 4
        self.target_tmax = pi / 2  # - pi / 6  # 3 * pi / 4
        self.targetv_rmin = 0.07  # 0.07
        self.targetv_rmax = 0.13  # 0.13
        self.targetv_tmin = pi  # + pi / 6  # + pi / 6  # 3 * pi / 4
        self.targetv_tmax = 3 * pi / 2  # - pi / 6  # - pi / 6  # 7 * pi / 4
        self.target_tv_bias = pi / 3 * 0.1

        self.agent_state = np.array([0.])  # [x, y, mode], x & y km, mode {-1, 1}
        self.target_state = np.array([0.])  # [x, y, mode], x & y km, mode {-1, 1}
        self.target_v = np.array([0.])
        self.target_next_predict = np.array([0.])
        self.reset()
        # self.render_init()
        pass


    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        self.step_counter = 0
        self.done = False
        # if self.total_num_steps > 2_000_000:  # Curriculum Learning
        #     rate = (self.total_num_steps - 2_000_000) / 4_000_000  # / total_steps_left - 2_000_000
        #     self.target_rmin = 18. - 3. * rate  # 15.
        #     self.target_rmax = 22. + 3. * rate  # 25.
        #     self.target_tmin = 0. + pi / 5 * (1. - rate)  # + pi / 6  # - pi / 4
        #     self.target_tmax = pi / 2 - pi / 5 * (1. - rate)  # - pi / 6  # 3 * pi / 4
        #     self.targetv_rmin = 0.09 - 0.02 * rate  # 0.07
        #     self.targetv_rmax = 0.11 + 0.02 * rate  # 0.13
        #     self.targetv_tmin = pi + pi / 5 * (1. - rate)  # + pi / 6  # + pi / 6  # 3 * pi / 4
        #     self.targetv_tmax = 3 * pi / 2 - pi / 5 * (1. - rate)  # - pi / 6  # - pi / 6  # 7 * pi / 4
        #     self.target_tv_bias = pi / 3 * (1.1 - rate)

        # self.agent_state = np.array([[21, 9, -1], [15, 14, -1], [9, 17, -1], [1.5, 17.5, -1], [0, 14.5, -1],
        #                              [3, 20.5, -1]],
        #                             dtype=np.float32)
        self.agent_state = np.array([[0, 0, -1], [-2, 2, -1], [-4, 0, -1], [2, -2, -1], [2, 2, -1],
                                     [-2, -2, -1]],
                                    dtype=np.float32)  # [x, y, mode], x & y km, mode {-1, 1}  [0, 4, -1]
        # self.target_state = np.array([[20, 10, 1], [5, 20, -1], [-2, 15, -1], [10, 18, 1]],
        #                              dtype=np.float32)  # [x, y, mode], x & y km, mode {-1, 1}
        self.target_state = np.array([[20, 10, 10], [5, 20, -10], [10, 15, -10]],
                                     dtype=np.float32)  # [x, y, mode], x & y km, mode {-1, 1}
        self.target_v = np.array([[-0.1, -0.005], [-0.02, -0.01], [-0.02, -0.05]])
        # self.target_v = np.array([[-0.2, -0.05], [-0.02, -0.1], [-0.01, -0.2], [-0.15, -0.1]])
        if not self.fixed:
            target_random = np.random.rand(self.target_num, 5)
            random_bias = (np.random.rand() - 0.5) * 2 * self.target_tv_bias
            random_direction = np.random.rand() * 2 * np.pi
            # agent_random = np.random.rand(self.agent_num, 2)
            target_random[:, 0] = target_random[:, 0]*(self.target_rmax-self.target_rmin)+self.target_rmin
            # changed: use random_direction for data augmenting
            target_random[:, 1] = target_random[:, 1]*(self.target_tmax-self.target_tmin)+self.target_tmin + random_direction
            target_random[:, 2] = target_random[:, 2] * (self.targetv_rmax - self.targetv_rmin) + self.targetv_rmin
            target_random[:, 3] = target_random[:, 3] * (self.targetv_tmax - self.targetv_tmin) + self.targetv_tmin + random_direction + random_bias
            a = target_random[:, 0]
            b = target_random[:, 0]
            target_x = target_random[:, 0] * np.cos(target_random[:, 1])
            target_y = target_random[:, 0] * np.sin(target_random[:, 1])
            target_vx = target_random[:, 2] * np.cos(target_random[:, 3])
            target_vy = target_random[:, 2] * np.sin(target_random[:, 3])
            # self.agent_state[:, 0:2] = agent_random * self.agent_l - self.agent_l / 2.
            self.target_state[:, 0] = target_x
            self.target_state[:, 1] = target_y
            self.target_v[:, 0] = target_vx
            self.target_v[:, 1] = target_vy
            self.target_state[:, 2] = 10 * np.sign(target_random[:, 4] - 0.5)
        # self.target_v = np.array([[-0.2, -0.05], [-0.02, -0.1], [0.2, 0.02], [0.04, -0.1]])
        # self.target_v = np.array([[-0.2, -0.05], [-0.02, -0.1], [-0.01, -0.2], [-0.15, -0.1]])
        self.target_next_predict = np.zeros_like(self.target_state)
        for i in range(self.target_num):
            self.target_next_predict[i] = self.target_state[i] + np.concatenate((self.target_v[i], np.zeros(1)))
        sub_agent_obs = self.generate_obs()
        if self.render:
            self.render_init()
        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        self.tar_step_forward()
        sub_agent_done, sub_agent_reward, sub_agent_info = self.agent_step_forward(self.action_transform(actions))

        self.step_counter += 1
        if self.render:
            if self.step_counter % self.render_interval == 0:
                self.render_step()
        if self.step_counter >= self.max_step_counter:
            sub_agent_done = [True for _ in range(self.agent_num)]
        if sub_agent_done[0]:
            if self.render:
                self.render_step()
                self.gif_writer.release()
            self.total_num_steps += 1
            sub_agent_obs = self.reset()
        else:
            sub_agent_obs = self.generate_obs()

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    def generate_obs(self):
        sub_agent_obs = []
        for i in range(self.agent_num):
            agent_pos = np.concatenate((self.agent_state[i][0:-1], np.array([0.])))
            sub_obs = np.array([])
            for j in range(self.agent_num):
                if i == j:
                    continue
                sub_obs = np.concatenate((sub_obs, self.agent_state[j] - agent_pos))
            for j in range(self.target_num):
                sub_obs = np.concatenate((sub_obs, self.target_next_predict[j] - agent_pos))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def tar_step_forward(self):
        self.target_state = self.target_next_predict + np.concatenate(
            (self.pos_delta * np.random.randn(self.target_num, 2), np.zeros((self.target_num, 1))), axis=1
        )
        self.target_v += self.v_delta * np.random.randn(self.target_num, 2)
        for i in range(self.target_num):
            self.target_next_predict[i] = self.target_state[i] + np.concatenate((self.target_v[i], np.zeros(1)))

    def agent_step_forward(self, actions):  # do action, return reward & done
        sub_agent_info = [{} for _ in range(self.agent_num)]
        for i, x in enumerate(actions):
            self.agent_state[i] = self.agent_state[i] + np.array(x)
            self.agent_state[i][-1] = x[-1]
        no_jam_target = []
        jam_target = []
        active_agent = []
        passive_agent = []
        agent_pair = []
        for i in range(self.target_num):
            if self.target_state[i][2] >= 0:
                jam_target.append(self.target_state[i][0:2])
            else:
                no_jam_target.append(self.target_state[i][0:2])
        for i in range(self.agent_num):
            if self.agent_state[i][2] >= 0:
                agent_pair.append([True, len(active_agent)])
                active_agent.append(self.agent_state[i][0:2])
            else:
                agent_pair.append([False, len(passive_agent)])
                passive_agent.append(self.agent_state[i][0:2])
        shared_reward = self.tracking_effect(no_jam_target, jam_target, active_agent, passive_agent, target_output=True)
        total_reward = []
        for i in range(self.agent_num):
            temp_a_agent = active_agent.copy()
            temp_p_agent = passive_agent.copy()
            if agent_pair[i][0]:
                temp_a_agent.pop(agent_pair[i][1])
            else:
                temp_p_agent.pop(agent_pair[i][1])
            personal_reward = self.tracking_effect(no_jam_target, jam_target, temp_a_agent, temp_p_agent)
            total_reward.append(shared_reward + self.personal_weight * (shared_reward - personal_reward))
        # TODO: 加入引导性奖励，如UAV模式接近最近的目标模式，UAV分散，（UAV靠近最离群的目标？这次先不加）
        # if not self.render:  # training set
        #     exploration_reward = self.exploration()
        #     for i, r in enumerate(exploration_reward):
        #         total_reward[i] += r
        penalty_pair, done = self.collsion()
        for (type_bool, index, rate) in penalty_pair:
            if type_bool:
                total_reward[index] -= self.uu_collision_penalty * rate
                sub_agent_info[index]['collision'] = 'UAV'
            else:
                total_reward[index] -= self.ut_collision_penalty * rate
                sub_agent_info[index]['collision'] = 'target'
        dones = [done for _ in range(self.agent_num)]
        rewards = []
        for reward in total_reward:
            rewards.append([reward])
        return dones, rewards, sub_agent_info


    def action_transform(self, actions):
        t_actions = []
        for action in actions:
            new_action = action.copy()
            new_action[0:2] = new_action[0:2] * self.max_dxy  # [-1, 1] maps to [-dxy, dxy]
            new_action[2] = np.sign(new_action[2])
            for x in range(2):
                if new_action[x] > self.max_dxy:
                    new_action[x] = self.max_dxy
                if new_action[x] < -self.max_dxy:
                    new_action[x] = -self.max_dxy
            t_actions.append(new_action)
        return t_actions

    def euc(self, pos1: np.ndarray, pos2: np.ndarray):
        return np.sum((pos1 - pos2) ** 2)

    def vec_norm(self, vec: np.ndarray):
        return np.sum(vec ** 2) ** 0.5

    def theta(self, vec1, vec2):
        return abs(np.dot(vec1, vec2)) / self.vec_norm(vec1) / self.vec_norm(vec2)

    def tracking_effect(self, no_jam_target, jam_target, active_agent, passive_agent, target_output=False):
        # compute active tracking
        var_list = []
        if target_output:
            self.worst_target = None
            self.worst_value = 0
        if len(active_agent) == 0:
            for _ in no_jam_target:
                var_list.append(self.large)
            if target_output:
                if len(no_jam_target) > 0:
                    self.worst_target = [no_jam_target[0], -1]
                    self.worst_value = self.large
        else:
            for target in no_jam_target:
                fisher_mat = np.zeros((2, 2))
                for agent in active_agent:
                    R = self.euc(target, agent)
                    G = np.array([(target-agent)/R, (agent-target)[::-1]/R/R])
                    sigma_inv = np.diag([self.C_R/(R**self.active_gamma), self.C_theta/(R**self.active_gamma)])
                    fisher_mat += np.matmul(np.matmul(G.T, sigma_inv), G)
                try:
                    new_var = np.trace(np.linalg.inv(fisher_mat))
                    var_list.append(new_var)
                    if target_output:
                        if new_var > self.worst_value:
                            self.worst_target = [target, -1]
                            self.worst_value = new_var
                except LinAlgError:
                    var_list.append(self.large)
                    if target_output:
                        self.worst_target = [target, -1]
                        self.worst_value = self.large

        # compute passive tracking
        p_var_list = []
        if len(passive_agent) <= 1:
            for _ in jam_target:
                p_var_list.append(self.large)
            if target_output:
                if len(jam_target) > 0:
                    self.worst_target = [jam_target[0], 1]
                    self.worst_value = self.large
        else:
            for target in jam_target:
                H = np.zeros((len(passive_agent), 2))
                c_vec = []
                for i, agent in enumerate(passive_agent):
                    R = self.euc(agent, target)
                    H[i][0] = (agent[1] - target[1]) / R / R
                    H[i][1] = (target[0] - agent[0]) / R / R
                    c_vec.append(self.C_passive / R / R)
                C_inv = np.diag(c_vec)
                try:
                    new_var = np.trace(np.linalg.inv(np.matmul(np.matmul(H.T, C_inv), H)))
                    p_var_list.append(new_var)
                    if target_output:
                        if new_var > self.worst_value:
                            self.worst_target = [target, 1]
                            self.worst_value = new_var
                except LinAlgError:
                    p_var_list.append(self.large)
                    if target_output:
                        self.worst_target = [target, 1]
                        self.worst_value = self.large
        reward = self.target_num / (sum(var_list) + sum(p_var_list))
        try:
            if reward <= 0.:
                reward = 1. / self.large
            x = 3 * log(reward, 10.) + 2
        except ValueError:
            print('Wrong reward: '+str(reward))
            raise ValueError
        return x
        # return reward ** 0.5
        # # 草率拟合版本
        # # compute active tracking
        # fisher_list = []
        # for target in no_jam_target:
        #     fisher_info = 1e-10
        #     for agent in active_agent:
        #         distance = self.euc(target, agent)
        #         fisher_info += 1. / (distance ** self.active_gamma)
        #     fisher_list.append(fisher_info)
        # temp = 0.
        # for info in fisher_list:
        #     temp += 1. / info
        #
        # # compute passive tracking
        # p_fisher_list = []
        # for target in jam_target:
        #     fisher_info = 1e-10
        #     for i, agent_i in enumerate(passive_agent):
        #         for j, agent_j in enumerate(passive_agent):
        #             if i >= j:
        #                 continue
        #             baseline = self.euc(agent_i, agent_j)
        #             agent_aver = (agent_i + agent_j) / 2
        #             distance = (self.euc(agent_i, target) + self.euc(agent_j, target)) / 2
        #             baseline_v = agent_j - agent_i
        #             baseline_t = baseline_v.copy()
        #             baseline_t[0] = baseline_v[1]
        #             baseline_t[1] = -baseline_v[0]
        #             target_v = target - agent_aver
        #             inner = self.theta(baseline_t, target_v)
        #             fisher_info += 10. * (inner ** 2) * baseline / (distance ** self.passive_gamma)
        #     p_fisher_list.append(fisher_info)
        # for info in p_fisher_list:
        #     temp += 1. / info
        # reward = self.target_num / temp
        # return reward

    def collsion(self):
        penalty_pair = []
        done = False
        for i in range(self.agent_num):
            for j in range(self.agent_num):
                if i >= j:
                    continue
                if self.euc(self.agent_state[i][0:2], self.agent_state[j][0:2]) <= self.uu_collision_d:
                    penalty_pair.append((True, i, 1.))
                    penalty_pair.append((True, j, 1.))
                    done = True
            for j in range(self.target_num):
                dis = self.euc(self.agent_state[i][0:2], self.target_state[j][0:2])
                if dis <= self.ut_collision_d:
                    if dis > self.ut_final_d:
                        d_rate = 1. - (dis - self.ut_final_d) / (self.ut_collision_d - self.ut_final_d)
                        p_rate = 3 * (d_rate ** 2) + 2 * (d_rate ** 3)
                        penalty_pair.append((False, i, p_rate ** 4))
                    else:
                        penalty_pair.append((False, i, 1.))
                        done = True
        return penalty_pair, done

    def exploration(self):
        exploration_reward = []
        for i in range(self.agent_num):
            agent = self.agent_state[i]
            reward = 0.
            nearest_t = None
            nearest_d = 1e6
            # work mode
            for j in range(self.target_num):
                target = self.target_state[j]
                d = self.euc(agent[0:2], target[0:2])
                if d < nearest_d:
                    nearest_t = target
                    nearest_d = d
            if agent[2] * nearest_t[2] < 0:
                reward += self.exploration_reward_base
            else:
                reward -= self.exploration_reward_base
            # space entropy
            for j in range(self.agent_num):
                if i == j:
                    continue
                if self.euc(agent[0:2], self.agent_state[j][0:2]) <= self.UAV_exploration_distance:
                    reward -= self.exploration_reward_base / 2
            exploration_reward.append(reward)
        return exploration_reward

    def render_init(self):
        name = str(self.total_num_steps) + '_' + str(self.render_num) + '.avi'
        self.H = 600
        self.W = 600
        self.map = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        self.map[:] = (255, 255, 255)
        self.red_d = np.array([127, 127, 0], dtype=np.uint8)
        self.green_d = np.array([127, 0, 127], dtype=np.uint8)
        self.blue_d = np.array([0, 127, 127], dtype=np.uint8)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.gif_writer = cv2.VideoWriter('../results/videos/' + name, fourcc, 4.0, (self.H, self.W), True)
        self.render_step()
        # self.gif_writer.release()

    def render_step(self):
        self.map[:] = (255, 255, 255)
        for i in range(self.agent_num):
            x = int(10 * self.agent_state[i][0] + self.W / 2)
            y = int(10 * self.agent_state[i][1] + self.H / 2)
            if x < 0 or y < 0 or x >= self.W or y >= self.H:
                continue
            if self.agent_state[i][2] >= 0:
                self.draw_circle(x, y, 5, self.blue_d)  # active radar
            else:
                self.draw_circle(x, y, 5, self.red_d)  # passive radar
        for i in range(self.target_num):
            x = int(10 * self.target_state[i][0] + self.W / 2)
            y = int(10 * self.target_state[i][1] + self.H / 2)
            if x < 0 or y < 0 or x >= self.W or y >= self.H:
                continue
            if self.target_state[i][2] >= 0:
                self.draw_x(x, y, 5, self.red_d)  # jamming target
            else:
                self.draw_x(x, y, 5, self.blue_d)  # no jam target
        # self.draw_circle(300+2*f, 300+2*f, 5, self.red_d)
        # self.draw_circle(320+2*f, 320+2*f, 5, self.blue_d)
        # self.draw_x(500-2*f, 500-2*f, 5, self.red_d)
        # self.draw_x(400-2*f, 450-2*f, 5, self.blue_d)
        self.gif_writer.write(self.map)



    def draw_circle(self, x, y, r, color_d):
        center = np.array([x, y])
        xmin = max(0, floor(x - r))
        xmax = min(self.map.shape[0], ceil(x + r))
        ymin = max(0, floor(y - r))
        ymax = min(self.map.shape[1], ceil(y + r))
        for i in range(xmin, xmax):
            for j in range(ymin, ymax):
                if self.euc(np.array([i, j]), center) <= r ** 2 + 0.0001:
                    self.map[i][j][:] -= color_d


    def draw_x(self, x, y, lx, color_d):
        for i in range(lx):
            if x - i >= 0:
                if y - i >= 0:
                    self.map[x - i][y - i][:] -= color_d
                if y + i < self.map.shape[1]:
                    self.map[x - i][y + i][:] -= color_d
            if x + i < self.map.shape[0]:
                if y - i >= 0:
                    self.map[x + i][y - i][:] -= color_d
                if y + i < self.map.shape[1]:
                    self.map[x + i][y + i][:] -= color_d





###################################################################

# import numpy as np
#
#
# class EnvCore(object):
#     """
#     # 环境中的智能体
#     """
#
#     def __init__(self):
#         self.agent_num = 2  # 设置智能体(小飞机)的个数，这里设置为两个 # set the number of agents(aircrafts), here set to two
#         self.obs_dim = 14  # 设置智能体的观测维度 # set the observation dimension of agents
#         self.action_dim = 5  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional
#
#     def reset(self):
#         """
#         # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
#         # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
#         """
#         sub_agent_obs = []
#         for i in range(self.agent_num):
#             sub_obs = np.random.random(size=(14,))
#             sub_agent_obs.append(sub_obs)
#         return sub_agent_obs
#
#     def step(self, actions):
#         """
#         # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
#         # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
#         # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
#         # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
#         """
#         sub_agent_obs = []
#         sub_agent_reward = []
#         sub_agent_done = []
#         sub_agent_info = []
#         for i in range(self.agent_num):
#             sub_agent_obs.append(np.random.random(size=(14,)))
#             sub_agent_reward.append([np.random.rand()])
#             sub_agent_done.append(False)
#             sub_agent_info.append({})
#
#         return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

if __name__ == '__main__':
    e = EnvCore(fix=True, render=False)
    e.step([np.array([0., 0., -1.]), np.array([0., 0., -1.]), np.array([0., 0., -1.]), np.array([0., 0., 1.]),
            np.array([0., 0., 1.]), np.array([0., 0., 1.])])
    # e.step([np.array([0., 0., 1.]), np.array([0., 0., -1.]), np.array([0., 0., -1.]), np.array([0., 0., 1.]), np.array([0., 0., -1.]),
    #         np.array([0., 0., 1.])])
