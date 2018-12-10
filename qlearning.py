
import numpy as np
import operator, random, sys, math
from copy import copy, deepcopy

def max_action(actions):
    #just return value if the state is good or bad state
    if(type(actions) != dict):
        return None
    maxvalue = max(actions.items(), key=operator.itemgetter(1))[1]
    max_actions = [key for key in actions.keys() if actions[key]==maxvalue]
    return max_actions

def max_action_value(actions):
    #just return value if the state is good or bad state
    if(type(actions) != dict):
        return actions
    return max(actions.items(), key=operator.itemgetter(1))[1]

def print_qtable(qtable):
    for i in range(0, 10):
        print(i,': [', end='')
        for j in range(0, 10):
            print(j,':',{None if max_action(qtable[i,j]) == None else max_action(qtable[i,j])[0]: max_action_value(qtable[i,j])},', ', end='')
        print(']\n')

def print_table(table):
    for i in range(0, 10):
        print(i,': [', end='')
        for j in range(0, 10):
            print(j,':',table[i, j],', ', end='')
        print(']\n')

def explore_exploit_policy(q_table, curr_state, policy, value):
    if policy == 'egreedy':
        if np.random.random() <= value:
            #action = ('up','down','right','left')[np.random.randint(0,3)]
            action = random.choice(list(q_table[curr_state[0]][curr_state[1]].keys()))
        else:
            action = random.choice(max_action(q_table[curr_state[0]][curr_state[1]]))
        return action
    elif policy == 'boltzmann':
        key_length = len(q_table[curr_state[0]][curr_state[1]].keys())
        pr = {}
        denom = 0.0
        maxpr = 0.0
        for key in q_table[curr_state[0]][curr_state[1]].keys():
            denom += np.exp(q_table[curr_state[0]][curr_state[1]][key] / value)

        for key in q_table[curr_state[0]][curr_state[1]].keys():
            pr[key] = np.exp(q_table[curr_state[0]][curr_state[1]][key] / value) / denom
            if pr[key] > maxpr:
                maxpr = pr[key]

        actions = []
        for key in q_table[curr_state[0]][curr_state[1]].keys():
            if pr[key] == maxpr:
                actions.append(key)
        return random.choice(actions)

def compare_values(otable, ctable):
    for i in range(0,10):
        for j in range(0, 10):
            if type(ctable[i, j]) == dict:
                for key in ctable[i, j].keys():
                    if not math.isclose(otable[i, j][key], ctable[i, j][key]):
                        return False
            elif ctable[i, j] != None:
                if not math.isclose(otable[i, j], ctable[i, j]):
                        return False
    return True

def q_learning(q_table, state_space, ee_policy):

    learning_rate_alpha = 0.01
    discount_factor_beta = 0.9

    curr_state = [0,0]
    next_state = [0,0]

    step_cost = 0.0

    #temperature = sys.maxsize
    temperature = np.iinfo(np.int).max
    iterations = 0;

    old_q_table = deepcopy(q_table)

    comparisons = []

    while True:
        #temperature -= 1000
        if state_space[curr_state[0]][curr_state[1]] == 1:
            #print("reached goal state")
            iterations += 1
            temperature -= 1000
            old_value = q_table[curr_state[0]][curr_state[1]]
            q_table[curr_state[0]][curr_state[1]] += learning_rate_alpha * (state_space[curr_state[0]][curr_state[1]] - q_table[curr_state[0]][curr_state[1]])
            #q_table[curr_state[0]][curr_state[1]] += 0.5 * (state_space[curr_state[0]][curr_state[1]] - q_table[curr_state[0]][curr_state[1]])
            #print("value "+str(old_value))
            #if abs( old_value - q_table[curr_state[0]][curr_state[1]] ) <= 0.00001:
            #if math.isclose(abs(old_value - q_table[curr_state[0]][curr_state[1]]), 0.0):
            comparisons.append(compare_values(old_q_table, q_table))
            if list(set(comparisons[len(comparisons)-100:len(comparisons)]))[0] == True and len(set(comparisons[len(comparisons)-100:len(comparisons)])) == 1:
                print("iterations "+str(iterations))
                break
            curr_state = [0, 0]
            old_q_table = deepcopy(q_table)
            #print_qtable(q_table)
            continue
        if state_space[curr_state[0]][curr_state[1]] == -1:
            iterations += 1
            temperature -= 1000
            #print("bad reward state "+str(iterations)+"\nstart over")
            #print("bad reward state")
            q_table[curr_state[0]][curr_state[1]] += learning_rate_alpha * (state_space[curr_state[0]][curr_state[1]] - q_table[curr_state[0]][curr_state[1]])
            #q_table[curr_state[0]][curr_state[1]] += 0.5 * (state_space[curr_state[0]][curr_state[1]] - q_table[curr_state[0]][curr_state[1]])
            curr_state = [0, 0]
            continue

        #np.random.seed(123)
##        if np.random.random() <= explore_exploit:
##            action = ('up','down','right','left')[np.random.randint(0,3)]
##        else:
##            action = max_action(q_table[curr_state[0]][curr_state[1]])

        if ee_policy == 'egreedy':
            explore_exploit = float(sys.argv[2]) # 0.1, 0.2, 0.3
            action = explore_exploit_policy(q_table, curr_state, ee_policy, explore_exploit)
        elif ee_policy == 'boltzmann':
            action = explore_exploit_policy(q_table, curr_state, ee_policy, temperature)

        if action=='up' and curr_state[0] > 0 and state_space[curr_state[0]-1][curr_state[1]] != 2:
            next_state[0] = curr_state[0] - 1
            next_state[1] = curr_state[1]
        elif action=='down' and curr_state[0] < 9 and state_space[curr_state[0]+1][curr_state[1]] != 2:
            next_state[0] = curr_state[0] + 1
            next_state[1] = curr_state[1]
        elif action=='right' and curr_state[1] < 9 and state_space[curr_state[0]][curr_state[1]+1] != 2:
            next_state[0] = curr_state[0]
            next_state[1] = curr_state[1] + 1
        elif action=='left' and curr_state[1] > 0 and state_space[curr_state[0]][curr_state[1]-1] != 2:
            next_state[0] = curr_state[0]
            next_state[1] = curr_state[1] - 1

        #print("current state: ",curr_state)
        #print("action: "+action)
        #print("q-value", q_table[curr_state[0]][curr_state[1]][action])
        #print("temperature", temperature)
        #print("next state: ",next_state)
        #print(q_table)
        #print(type(q_table[curr_state[0]][curr_state[1]]))
        #print(max_action_value(q_table[next_state[0]][next_state[1]]))

        #TD update #decrease cost for every step
        q_table[curr_state[0]][curr_state[1]][action] += learning_rate_alpha * (state_space[curr_state[0]][curr_state[1]] + (discount_factor_beta * max_action_value(q_table[next_state[0]][next_state[1]])) - q_table[curr_state[0]][curr_state[1]][action]) - step_cost
        print_qtable(q_table)
        curr_state = next_state[:]

##        if action == 'up':
##            q_table[i][j]['up'] += learning_rate_alpha(state_space[i][j] + (discount_factor_beta * max_value_action(q_table[i-1][j])[1]) - q_table[i][j]['up'])
##        elif action == 'down':
##            q_table[i][j]['down'] += learning_rate_alpha(state_space[i][j] + (discount_factor_beta * max_value_action(q_table[i+1][j])[1]) - q_table[i][j]['down'])
##        elif action == 'right':
##            q_table[i][j]['right'] += learning_rate_alpha(state_space[i][j] + (discount_factor_beta * max_value_action(q_table[i][j+1])[1]) - q_table[i][j]['right'])
##        elif action == 'left':
##            q_table[i][j]['left'] += learning_rate_alpha(state_space[i][j] + (discount_factor_beta * max_value_action(q_table[i][j-1])[1]) - q_table[i][j]['left'])
    pass

def main():
    #initializing state space
    state_space = []
    for i in range(0,10):
        state_space.append(np.zeros(10, dtype=np.int))
    state_space = np.array(state_space)
    #adding walls
    for i in range(1,5):
        state_space[2][i] = 2
    for i in range(6,9):
        state_space[2][i] = 2
    for i in range(2,8):
        state_space[i][4] = 2
    #adding positive reward
    state_space[5][5] = 1
    #adding negative rewards
    state_space[4][5] = -1
    state_space[4][6] = -1
    state_space[5][6] = -1
    state_space[5][8] = -1
    state_space[6][8] = -1
    state_space[7][5] = -1
    state_space[7][6] = -1
    state_space[7][3] = -1

    #Q table of values
    q_values = []
    for i in range(0,10):
        q_values.append([])
        for j in range(0,10):
            #q_values[i].append(np.zeros(4, dtype=np.int))
            q_values[i].append([])
            if state_space[i][j] == 2:
                q_values[i][j] = None
            elif state_space[i][j] == 1 or state_space[i][j] == -1:
                q_values[i][j] = 0.0
            else:
                if i==0 and j==0:
                    q_values[i][j] = {'down':0.0, 'right':0.0}
                elif i==0 and j==9:
                    q_values[i][j] = {'down':0.0, 'left':0.0}
                elif i==9 and j==0:
                    q_values[i][j] = {'up':0.0, 'right':0.0}
                elif i==9 and j==9:
                    q_values[i][j] = {'up':0.0, 'left':0.0}
                elif i==0:
                    q_values[i][j] = {'down':0.0, 'right':0.0, 'left':0.0}
                elif i==9:
                    q_values[i][j] = {'up':0.0, 'right':0.0, 'left':0.0}
                elif j==0:
                    q_values[i][j] = {'up':0.0, 'down':0.0, 'right':0.0}
                elif j==9:
                    q_values[i][j] = {'up':0.0, 'down':0.0, 'left':0.0}
                else:
                    q_values[i][j] = {'up':0.0, 'down':0.0, 'right':0.0, 'left':0.0}
        q_values[i] = np.array(q_values[i])
    q_values = np.array(q_values)

    print_table(state_space)
    print("Q values",q_values)
 #   print(q_values[5][5])
  #  print(q_values[1][1])
  #  print(q_values[8][6]['up'])

    ee_policy = sys.argv[1]

    q_learning(q_values, state_space, ee_policy)
    print("Q value ",q_values)

    pass

if __name__ == '__main__':
    main()
