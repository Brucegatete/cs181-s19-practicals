#this implements a Q-learning agent

import numpy as np
import numpy.random as npr
import sys
import math
import random
import matplotlib.pyplot as plt

from SwingyMonkey import SwingyMonkey


class QLearner:

    def __init__(self):

        #bin discretization
        self.tree_distance_range = (0, 600)
        self.tree_dist_bins = 10
        self.monkey_vel_range = (-50,50)
        self.monkey_vel_bins = 10
        self.top_diff_range = (-450, 400)
        self.top_diff_bins = 20
        
        # epoch number
        self.epoch = 0

        #parameters 
        self.alpha = 0.1
        self.gamma = 0.1
        self.epsilon = 0.5 * (math.exp(- self.epoch/ 500))

         # state parameters
        self.current_action = None
        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None

        # dimensionality
        dimensions = self.state_rep_dimensions() + (2,)
        self.Q = np.zeros(dimensions)

        self.k = np.ones(dimensions)


    def reset(self):
        self.current_action = None
        self.current_state  = None
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epoch += 1
        
    def state_rep_dimensions(self):
        '''Returns a tuple containing the dimensions of the state space;
        should match the dimensions of an object returned by self.state_rep'''
        return (
            self.tree_dist_bins, 
            self.monkey_vel_bins, 
            self.top_diff_bins)

    def action_callback(self, state):
        "Learns and takes actions, returns 0 or 1"
        # epsilon-greedy policy
        if (random.random() < self.epsilon):
            new_action = random.choice((0,1))
        else:
            new_action = np.argmax(self.Q[self.state_rep(state)])
        new_state  = state

        # learning
        self.last_action = new_action
        self.last_state  = self.current_state
        self.current_state = new_state

        st  = self.state_rep(state)
        act  = (self.last_action,)

        self.k[st + act] += 1

        return self.last_action
    def reward_callback(self, reward):
        if (self.current_state != None) and (self.last_state != None) and (self.last_action != None):
            st  = self.state_rep(self.last_state)
            cur_st = self.state_rep(self.current_state)
            act  = (self.last_action,)

            if self.epoch < 100:
                alpha = self.alpha
            else:
                alpha = self.alpha*0.1

            # Q learning
            self.Q[st + act] = self.Q[st + act] + alpha * (reward + self.gamma * np.max(self.Q[cur_st]) - self.Q[st + act] )

        self.last_reward = reward
    
    def discretize(self, value, range, bins):
        #divides the state into bins
        bin_size = (range[1] - range[0]) / bins
        return math.floor((value - range[0]) / bin_size)
    
    def state_rep(self, state):
         #takes state dict and returns a tuple representing state
         return (
                self.discretize(state["tree"]["dist"],self.tree_distance_range,self.tree_dist_bins), 
                self.discretize(state["monkey"]["vel"],self.monkey_vel_range,self.monkey_vel_bins), 
                self.discretize(state["tree"]["top"]-state["monkey"]["top"],self.top_diff_range,self.top_diff_bins))


av_score = []
cur_score = []

def run_game(gamma=0.4, iters=100, chatter=True):
    learner = QLearner()
    learner.gamma = gamma

    highscore = 0
    avgscore = 0.0

    if chatter:
        print ("epoch", "\t", "score", "\t", "high", "\t", "avg")

    for m in range(iters):

        learner.epsilon = 1/(m+1)

        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,
                             text="Epoch %d" % (m), # Display the epoch on screen.
                             tick_length=1,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        score = swing.get_state()['score']
        highscore = max([highscore, score])
        avgscore = (m*avgscore+score)/(m+1)

        if chatter:
            print (m, "\t", score, "\t", highscore, "\t", avgscore)

        # Reset the state of the learner.
        learner.reset()
        av_score.append(avgscore)
        cur_score.append(score)

    return -avgscore


def hyper_paramters_opt():

    # find the best value for hyperparameters
    best_parameters = (0,0)
    best_value = 0
    for gamma in np.arange(0.1,1,0.1):
        parameters = {"gamma": gamma}
        value = run_game(**parameters)
        if value < best_value:
            best_parameters = parameters
            print("Best: ",parameters, " : ", value)


    print(best_parameters)
    return best_parameters

run_game(iters=5,gamma=0.4)

x_axis = [i for i in range(10)]
av_score = [int(i) for i in av_score]
print(x_axis)
print(av_score)
print(len(cur_score))
#plot info
plt.plot(x_axis, av_score, 'r')
plt.xlabel("epoch")
plt.ylabel("av_score")
plt.show()






