import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import deque

class Floorplan:
    def __init__(self, n:int, m:int, doorcoord:tuple[int, int] = (0,0)) -> None:
        self.n = n
        self.m = m
        self.heatdata = np.ones(shape=(n,m), dtype=np.int32)
        self.doorcoord = doorcoord
    def mask_rect(self, left, right, bottom, top) :
        self.heatdata[left:right, bottom:top]  = 0
    def color_door(self) :
        self.heatdata[self.doorcoord] = 0

    #fill in each cell with its manhattan distance from the door
    def manhattan_fill(self) :
        for x, row in enumerate(self.heatdata) :
            for y, cell in enumerate(row) :
                doorx, doory = self.doorcoord
                self.heatdata[x, y] = (abs(x-doorx) + abs(y-doory))

    #do BFS from door to get manhattan distance but wrapping around furniture
    def bfs_fill(self) :
        seenSet = set()
        queue = deque()
        queue.append((self.doorcoord, 0))
        seenSet.add(self.doorcoord)
        while (queue) :
           
            ((i, j), dist) = queue.popleft()
            # print("expand:", (i,j))

            if (self.heatdata[i, j] == 0) : continue #skip furniture cells
            self.heatdata[i, j] = dist #color the cell its dist from door when expanding

            neighbors = [(i, j+1), (i, j-1), (i+1, j), (i-1, j)] #right left down up
            for i, j in neighbors : #add unvisited neighbors that are in bounds to queue
                if (i,j) in seenSet : continue
                if i<self.n and i>=0 and j<self.m and j>=0 :
                    queue.append(((i,j), dist+1))
                    seenSet.add((i,j))




def main():
    roomWidth = 500
    roomHeight = 400
    fp = Floorplan(n=roomHeight, m=roomWidth, doorcoord= (0, 80))
    # fp.color_door()

    # fp.manhattan_fill()
    # fp.mask_rect(5,12,5,12)
    fp.mask_rect(30,40,10,100)
    fp.mask_rect(70,80,20,40)
    fp.bfs_fill()

    # fp = Floorplan(n=50, m=50, doorcoord= (0, 0))
    # fp.mask_rect(30,40,10,100)
    # fp.mask_rect(70,80,20,40)
    # fp.bfs_fill()



        


    print(fp.heatdata)

    colormap = "rocket_r"
    hm = sns.heatmap(data = fp.heatdata, cmap=colormap,square=True, 
                    linewidths=0, linecolor=None, vmin=None, vmax=None, annot=False, cbar=True)
    # hm = sns.heatmap(data = fp.heatdata, cmap=colormap)


    #set runtime config to make the figure a square to get square cells
    #doesn't work (USE SQUARE PARAMETER OF HEATMAP)
    # sns.set (rc = {'figure.figsize':(8, 8)}) 
    # plt.gcf().set_size_inches(5, 5)
    # fig, ax = plt.subplots (figsize=(4, 4))

    plt.show()
    

main()