#! /usr/bin/env python

import random
from gridWorld import init
import math

START_LOCATION = [0,9]

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ALPHA = 0.01
BETA = 0.9

EPSILON = 1 #Numerator over 10

CONVERGE_LIMIT = 1000 #Number of unchanged iterations

INIT_TEMP = 1000

changed_tries = 0

class Agent:
    def initialize(self):
        self.current_location = START_LOCATION
        self.is_converged = False

    def __init__(self):
        self.current_location = START_LOCATION
        self.previous_location = None


def arg_max(l):
    m = max(l)
    while(True):
        i = random.randrange(4)
        if l[i] == m:
            return i


def find_probs(q_values, num_iterations):
    T = max([INIT_TEMP - num_iterations, 1])
    
    exp_probs = [math.exp(q/T) for q in q_values]

    exp_sum = sum(exp_probs)

    action_probs = [prob/exp_sum for prob in exp_probs]
    return action_probs


def choose_action(current_location, gridWorld, num_iterations):
    #Given a state, chooses finds the optimal action for exploitation, and depending on the epsilon value, either returns a random number between 0 and 4
    state = gridWorld[current_location[0]][current_location[1]]
    q_values = state[0]

    action_probs = find_probs(q_values, num_iterations)

    rand_var = random.uniform(0, 1.0)

    for i in range(len(action_probs)):
        if rand_var < sum(action_probs[:i+1]):
            return i

    return len(action_probs) - 1


def optimal_action(current_location, gridWorld):
    state = gridWorld[current_location[0]][current_location[1]]
    q_values = state[0]
    return arg_max(q_values)


def update_agent_location(agent, action, gridWorld):
    #Find the agent's current location, use the action to check what the next location is.
    wall = False
    goal_state = False
    old_location = agent.current_location
    #If agent is at goal state, then update location to start state no matter what the action
    if (gridWorld[old_location[0]][old_location[1]][1] == 1):
        agent.previous_location = agent.current_location
        agent.current_location = START_LOCATION
        print ("GOAL STATE");
        return True
    new_location = old_location[:]
    if action == UP:
        new_location[1] += 1
    if action == DOWN:
        new_location[1] -= 1
    if action == LEFT:
        new_location[0] -= 1
    if action == RIGHT:
        new_location[0] += 1
    if (new_location[0] < 0 or new_location[0] > 9 or new_location[1] < 0 or new_location[1] > 9):
        wall = True
    if (wall == False and gridWorld[new_location[0]][new_location[1]] == 0):
        wall = True

    if wall == True:
        #print "WALL"
        agent.current_location = old_location
        agent.previous_location = old_location
        return
    
    agent.current_location = new_location
    agent.previous_location = old_location
    return False
    


def update_q_value(agent, action, gridWorld):
    global changed_tries
    current_location = agent.current_location
    previous_location = agent.previous_location
   # print current_location, previous_location
    cur_x, cur_y = current_location[0], current_location[1]
    pre_x, pre_y = previous_location[0], previous_location[1]
    new_q = gridWorld[cur_x][cur_y][0]
    q_values = gridWorld[pre_x][pre_y][0][:]
    reward = gridWorld[pre_x][pre_y][1]
    q_values[action] += ALPHA * (reward + (BETA * max(new_q)) - q_values[action])
    if gridWorld[pre_x][pre_y][0] != q_values[:]:
        changed_tries += 1
    gridWorld[pre_x][pre_y][0] = q_values[:]


def print_optimal_policy(gridWorld):
    agent = Agent()
    goal_state = False
    while (not goal_state):
        action = optimal_action(agent.current_location, gridWorld)
        if (action == UP):
            print ('UP');
        if (action == DOWN):
            print ("DOWN");
        if (action == LEFT):
            print ("LEFT");
        if (action == RIGHT):
            print ("RIGHT");
        goal_state = update_agent_location(agent, action, gridWorld)
        update_result = update_q_value(agent, action, gridWorld)
    

def main():
    global changed_tries
    #Repeat until convergence:
    #pick an action
    #Execute the action and get the resulting location from the world
    #Based on the new location, update the Q values of the previous location
    #If new location is goal state, reset back to START_LOCATION
    agent = Agent()
    gridWorld = init()
    is_converged = False
    num_iterations = 0
    num_unchanged_iterations = 0
    while (num_unchanged_iterations < CONVERGE_LIMIT):
        changed_tries = 0
        goal_state = False
        print(num_iterations);
        while (not goal_state):
            action = choose_action(agent.current_location, gridWorld, num_iterations)
            #print action
            goal_state = update_agent_location(agent, action, gridWorld)
            update_result = update_q_value(agent, action, gridWorld)
        if changed_tries == 0:
            num_unchanged_iterations += 1
        else: num_unchanged_iterations = 0
        num_iterations += 1

    print(gridWorld)
    print("Iterations required for convergence: ", num_iterations - CONVERGE_LIMIT)
    print_optimal_policy(gridWorld)

if __name__ == "__main__":
    main()
