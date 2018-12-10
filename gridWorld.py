#! /usr/bin/env python

gridWorld = list()

q_value_init = [0,0,0,0]


block_cells = [[1,7], [2,7], [3,7], [4,7], [6,7], [7,7], [8,7], [4,6], [4,5], [4,4], [4,3], [4,2]]

pos_reward_cells = [[5,4]]

neg_reward_cells = [[3,2], [5,2], [6,2], [3,6], [8,3], [8,4], [6,4], [6,5], [5,5]]

def init():
    for i in range(10):
        row = []
        for j in range(10):
            if [i,j] not in block_cells:
                row.append([q_value_init[:], 0])
            else:
                row.append(0)
        gridWorld.append(row[:])

    for cell in pos_reward_cells:
        gridWorld[cell[0]][cell[1]] = [[1,1,1,1],1]

    for cell in neg_reward_cells:
        gridWorld[cell[0]][cell[1]][1] = -1

    return gridWorld[:]


init()

print (gridWorld);
