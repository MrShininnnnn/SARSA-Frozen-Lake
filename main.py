#! /usr/bin/env python
__author__ = 'Ning Shi'
__email__ = 'mrshininnnnn@gmail.com'


# import dependency
import gym # OpenAI Game Environment
import gym.envs.toy_text # Customized Map
import numpy as np
from tqdm import trange # Processing Bar
import matplotlib.pyplot as plt


class FrozenLake(object):
      """docstring for FrozenLake"""
      def __init__(self, amap='SFFFHFFFG'):
            super(FrozenLake, self).__init__()
            print('Initialize environment...')
            self.env = self.initialize_env(amap)
            self.n_states, self.n_actions = self.env.observation_space.n, self.env.action_space.n
            self.RESULT_IMG_PATH = 'img/result_img_{}.png'

      def initialize_env(self, amap):
            grid_shape = np.int(np.sqrt(len(amap)))
            custom_map = np.array(list(amap)).reshape(grid_shape, grid_shape)
            env = gym.envs.toy_text.frozen_lake.FrozenLakeEnv(desc=custom_map).unwrapped
            env.render()
            return env

      def random_agent(self, n_episodes):
            # performance of an agent taking random actions
            t = trange(n_episodes)
            # to record reward for each episode
            reward_array = np.zeros(n_episodes)
            # for each episode
            for i in t: 
                  # reset environment 
                  self.env.reset()
                  # done flag
                  done = False
                  while not done:
                        # randomly pick an action
                        action = np.random.randint(self.n_actions)
                        # get feedback from the environment
                        _, reward, done, _ = self.env.step(action)
                        if done: 
                              # update processing bar
                              t.set_description('Episode {} Reward {}'.format(i + 1, reward))
                              t.refresh()
                              reward_array[i] = reward
                              break
            self.env.close()
            # show average reward
            avg_reward = round(np.mean(reward_array), 4)
            print('Averaged reward per episode {}'.format(avg_reward))
            # generate output image
            title = 'Random Strategy\nReward Per Episode for {} Episodes - Average: {:.2f}'.format(n_episodes, avg_reward)
            self.gen_img(reward_array, title, 0)

      # initialize the agent’s Q-table to zeros
      def init_q(self, s, a): 
            """
            s: number of states
            a: number of actions
            """
            return np.zeros((s, a))

      # epsilon-greedy exploration strategy
      def epsilon_greedy(self, Q, epsilon, s):
            """
            Q: Q Table
            epsilon: exploration parameter
            s: state
            """
            # selects a random action with probability epsilon
            if np.random.random() <= epsilon:
                  return np.random.randint(self.n_actions)
            else:
                  return np.argmax(Q[s, :])

      # SARSA Process
      def sarsa_agent(self, alpha, gamma, epsilon, n_episodes): 
            """
            alpha: learning rate
            gamma: exploration parameter
            n_episodes: number of episodes
            """
            # initialize Q table
            Q = self.init_q(self.n_states, self.n_actions)
            # initialize processing bar
            t = trange(n_episodes)
            # to record reward for each episode
            reward_array = np.zeros(n_episodes)
            for i in t:
                  # initial state
                  s = self.env.reset()
                  # initial action
                  a = self.epsilon_greedy(Q, epsilon, s)
                  done = False
                  while not done:
                        s_, reward, done, _ = self.env.step(a)
                        a_ = self.epsilon_greedy(Q, epsilon, s_)
                        # update Q table
                        Q[s, a] += alpha * (reward + (gamma * Q[s_, a_]) - Q[s, a])
                        # update processing bar
                        if done:
                              t.set_description('Episode {} Reward {}'.format(i + 1, reward))
                              t.refresh()
                              reward_array[i] = reward
                              break
                        s, a = s_, a_
            self.env.close()
            # show Q table
            print('Trained Q Table:')
            print(Q)
            # show average reward
            avg_reward = round(np.mean(reward_array), 4)
            print('Training Averaged reward per episode {}'.format(avg_reward))
            return Q

      def eva(self, Q, n_episodes):
            """
            Q: trained Q table
            n_episodes: number of episodes
            """
            t = trange(n_episodes)
            # to record reward for each episode
            reward_array = np.zeros(n_episodes)
            # for each episode
            for i in t:
                  # initial state
                  s = self.env.reset()
                  # initial action
                  a = np.argmax(Q[s])
                  done = False
                  while not done:
                        s_, reward, done, _ = self.env.step(a)
                        # pick an action according the state and trained Q table
                        a_ = np.argmax(Q[s_])
                        if done:
                              t.set_description('Episode {} Reward {}'.format(i + 1, reward))
                              t.refresh()
                              reward_array[i] = reward
                              break
                        s, a = s_, a_
            self.env.close()
            # show average reward
            avg_reward = round(np.mean(reward_array), 4)
            print('Training Averaged reward per episode {}'.format(avg_reward))
            # generate output image
            title = 'SARSA Agent\nReward Per Episode for {} Episodes - Average: {:.2f}'.format(n_episodes, avg_reward)
            self.gen_img(reward_array, title, 1)

      def gen_img(self, reward_array, title, idx):
            # show reward per episode
            plt.subplots(figsize = (6, 6), dpi=100)
            plt.plot(reward_array, color='black', linewidth=0.5)
            plt.ylabel('Reward', fontsize=12)
            plt.xlabel('Episode', fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title(title, fontsize=12)
            plt.savefig(self.RESULT_IMG_PATH.format(idx), dpi=100, bbox_inches='tight')
            print('Saving output to ' + self.RESULT_IMG_PATH.format(idx))

def main():

      fl = FrozenLake()
      print('\nAn agent taking random actions:')
      fl.random_agent(100)
      print('\nSARSA agent:')
      Q = fl.sarsa_agent(0.1, 0.9, 0.5, 1000)
      reward_array = fl.eva(Q, 100)
      print('Done.')

if __name__ == '__main__':
      main()