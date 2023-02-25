import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Floorplan:
    def __init__(self, n:int, m:int, doorcoord:tuple[int, int] = (0,0)) -> None:
        self.heatdata = np.ones(shape=(n,m), dtype=np.int32)
        self.doorcoord = doorcoord
    def mask_rect(self, left, right, bottom, top) :
        self.heatdata[left:right, bottom:top]  = 0
        

colormap = "rocket"





fp = Floorplan(n=100, m=100, doorcoord= (30, 99))
fp.heatdata[fp.doorcoord] = 0 #color the door


#fill in each cell with its manhattan distance from 0,0
for x, row in enumerate(fp.heatdata) :
    for y, cell in enumerate(row) :
        doorx, doory = fp.doorcoord
        fp.heatdata[x, y] = -(abs(x-doorx) + abs(y-doory))

fp.mask_rect(70,90,70,90)
fp.mask_rect(70,80,20,40)

print(fp.heatdata)

hm = sns.heatmap(data = fp.heatdata, cmap=colormap, linewidths=0, linecolor=None, vmin=None, vmax=None, annot=False, cbar=True)
# hm = sns.heatmap(data = fp.heatdata, cmap=colormap)


#set runtime config to make the figure a square to get square cells
#not sure this really works
sns.set (rc = {'figure.figsize':(8, 8)}) 
# plt.gcf().set_size_inches(5, 5)
# fig, ax = plt.subplots (figsize=(4, 4))

plt.show()