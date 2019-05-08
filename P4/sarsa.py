# Imports.
import numpy as np
import numpy.random as npr
from collections import Counter

from SwingyMonkey import SwingyMonkey


discount = 0.5
epsilon = 0.2
eta = 0.1


class SARSALearner(object):


    def __init__(self):
        self.q_values = Counter()
        self.discount = discount
        self.eta = eta
        self.epsilon = {1: epsilon, 4: epsilon}
        self.gravity = 0

        self.new = True
        self.prev_state  = None
        self.last_action = None
        self.last_reward = None
        self.epoch = 0

    def reset(self):
        self.prev_state  = None
        self.last_action = None
        self.last_reward = None
        self.new = True
        self.gravity = 0
        self.epoch +=1


    #vertical_discretization
    def discretize(self, state, vel_size=10, pos_size=100):
        new_states = {}
        new_states['gravity'] = self.gravity
        new_states['monkey_to_bot'] = (state['monkey']['bot'] - state['tree']['bot']) / pos_size
        new_states['dist_to_next_tree'] = state['tree']['dist'] / pos_size
        new_states['monkey_to_top'] = (state['tree']['top'] - state['monkey']['top']) / pos_size
        return new_states

    # return Q value for state and action
    def getQValue(self, state, action):
        return self.q_values[(tuple(state.values()), action)]

    # Q update
    def update_Q(self, prev_state, last_action, actual_state, actual_action, reward):
        last_q = self.getQValue(prev_state, last_action)
        #actual_action = self.getMaxAction(actual_state)
        current_q = self.getQValue(actual_state, actual_action)
        gradient = last_q - (reward + self.discount * current_q)

        
        self.q_values[(tuple(prev_state.values()), last_action)] = (last_q - 
            self.eta * gradient)

    def getMaxAction(self, state):
        return np.argmax([self.getQValue(state,i) for i in [0,1]])

    def fetchPolicy(self, state):

        if npr.rand() < self.epsilon[self.gravity]:
            action = npr.choice([0, 1])
        else:
            action = self.getMaxAction(state)

        return action


    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # get gravity
        if self.new and not (self.prev_state is None ):
            self.gravity = self.prev_state['monkey']['vel'] - state['monkey']['vel']
            self.new = False
        
        
        if self.prev_state is None:
            actual_action = 0
        else:
            prev_state = self.discretize(self.prev_state)
            actual_state = self.discretize(state)
            # get current action according to policy
            actual_action = self.fetchPolicy(actual_state)

            self.update_Q(prev_state, self.last_action, 
                actual_state, actual_action, self.last_reward)
            

        self.last_action = actual_action
        self.prev_state = state

        return self.last_action

    def reward_callback(self, reward):
        self.last_reward = reward


def evaluate(SARSALearner, hist, iters = 100, t_len = 100):
    
    for m in range(iters):
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (m),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=SARSALearner.action_callback,
                             reward_callback=SARSALearner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
    
        hist.append(swing.score)

        # Reset the state of the SARSALearner.
        SARSALearner.reset()
    return


if __name__ == '__main__':

	# Select agent.
	agent = SARSALearner()

	# Empty list to save history.
	hist = []; 

	evaluate(agent, hist, 500, 1)
