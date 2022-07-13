import numpy as np
import gym
from gym import spaces

import math

class RetailerEnv(gym.Env):
    def __init__(self, num_states, start_state):
        self.done = False
        self.num_states = num_states

        self.alpha = np.zeros((num_states, 8))
        self.beta = np.zeros((num_states, 2))

        #alpha and beta values are randomly generated for testing
        self.alpha[:, 0] = np.random.rand(num_states)
        self.alpha[:, 1] = -np.random.rand(num_states)
        self.alpha[:, 2] = np.random.rand(num_states)
        self.alpha[:, 3] = np.random.rand(num_states)
        self.alpha[:, 4] = 10 * np.random.rand(num_states)
        self.alpha[:, 5] = -10 * np.random.rand(num_states)
        self.alpha[:, 6] = 10 * np.random.rand(num_states)
        self.alpha[:, 7] = np.random.rand(num_states)

        self.beta[:, 0] = np.random.randint(100, 150, num_states)
        self.beta[:, 1] = -10 * np.random.rand(num_states)

        self.states = np.random.rand(num_states, 5)

        self.history = [[], [], []] #store quantity, reward, and discount
        self.customer_feature = np.zeros(5) #customer features for the current state
        self.start_state = start_state
        self.prev_state = start_state #total is k
        self.e_t = 0
        self.p_t = 7

        self.action_space = spaces.Box(low=-1.0, high=2.0, shape=(4, 1), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=2.0, shape=(1, ), dtype=np.float32)

    def reset(self):
        self.prev_state = self.start_state
        observation = 0
        return observation

    def bernoulli_buy(self, s_t, action, e_t):
        l = self.alpha[s_t, 0] + self.alpha[s_t, 1] * self.states[s_t, 0] + self.alpha[s_t, 2] * self.states[s_t, 1] \
            + self.alpha[s_t, 3] * self.states[s_t, 2] + self.alpha[s_t, 4] * (self.states[s_t, 3]/self.states[s_t, 0]) ** 2 \
            + self.alpha[s_t, 5] * self.states[s_t, 4] + self.alpha[s_t, 6] * action[0] + self.alpha[s_t, 7] * e_t
        denom = 1 + np.exp(-l)
        buy_prob = 1/denom
        buy = np.random.choice(2, p=[1-buy_prob, buy_prob])
        return buy

    def demand(self, s_t, action, p_t):
        #TODO: review
        mu = self.beta[s_t, 0] + self.beta[s_t, 1] * (1 - action[0]) * p_t
        minimum = 1
        prob = np.zeros(1000)
        i = 0
        while np.sum(prob) < 1:
            prob[i] = self.truncated_poisson(mu, minimum, i)
            i += 1
        prob = prob[prob != 0]
        prob = prob[~np.isnan(prob)]
        prob = prob/np.sum(prob) #normalize
        quantity = np.random.choice(len(prob), p=prob)
        return quantity

    def truncated_poisson(self, mu, k, x):
        #TODO: can be vectorized
        sum_over_k = np.sum([mu**j/math.factorial(j) for j in range(0, k+1)])
        return mu ** x / (math.factorial(x) * (np.exp(mu) - sum_over_k))

    def step(self, action):
        #TODO: minimum quantity needed to use discount
        buy_probability = self.bernoulli_buy(self.prev_state, action, self.e_t)
        if buy_probability != 0:
            quantity = self.demand(self.prev_state, action, self.p_t)
        else:
            quantity = 0

        reward = self._reward(action, self.p_t, quantity)
        self.history[0] += [quantity]
        self.history[1] += [reward]
        self.history[2] += [action[0]]


        self.customer_feature = self.update_features()
        next_state = self.feature_to_state()
        self.prev_state = next_state

        return next_state, reward, self.done, self._get_info()

    def _reward(self, action, price, quantity):
        """
        :return: adjusted revenue
        """
        return (1 - action[0]) * price * quantity

    def update_features(self):
        #history to feature
        average_frequency = np.count_nonzero(np.array(self.history[0]))/len(self.history[0])
        average_quantity = np.mean(self.history[0])
        average_dollar = np.mean(self.history[1])
        not_zero = np.nonzero(self.history[0])[0]
        if len(not_zero) > 0:
            days_til_last_purchase = len(self.history[0]) - np.max(not_zero)
        else:
            days_til_last_purchase = len(self.history[0])
        average_discount = np.mean(self.history[2])
        return np.array([average_frequency, average_quantity, average_dollar, days_til_last_purchase, average_discount])

    def feature_to_state(self):
        dist = []
        for i in range(self.num_states):
            dist += [np.linalg.norm(self.states[i, :] - self.customer_feature)]
        return np.argmin(dist)

    def _get_info(self):
        return {}

    def render(self, mode='human'):
        pass

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


env = RetailerEnv(9, 0)
timestep = env.reset()
for i in range(10):
    action = [0.6, 1, 0, 0]
    timestep = env.step(action)
    print(timestep, env.customer_feature, env.prev_state)