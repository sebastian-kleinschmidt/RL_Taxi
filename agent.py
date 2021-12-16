import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = .01
        self.gamma =  1.0

    def update_Q(self, Qsa, Qsa_next, reward, alpha, gamma):
        """ updates the action-value function estimate using the most recent time step """
        return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

    def epsilon_greedy_probs(self, nA, Q_s, i_episode, eps=None):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        epsilon = 1.0 / i_episode
        if eps is not None:
            epsilon = eps
        policy_s = np.ones(nA) * epsilon / nA
        policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / nA)
        return policy_s
        
    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        policy_s = self.epsilon_greedy_probs(self.nA, self.Q[state], i_episode)
        action = np.random.choice(np.arange(self.nA), p=policy_s)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if not done:
            next_action = self.select_action(self, next_state)
            self.Q[state][action] = self.update_Q(self.Q[state][action], self.Q[next_state][next_action], reward, self.alpha, self.gamma)
        if done:
            self.Q[state][action] = self.update_Q(self.Q[state][action], 0, reward, self.alpha, self.gamma)