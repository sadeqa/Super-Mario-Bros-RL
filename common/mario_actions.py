import numpy as np

#THE FIRST ACTION MUST BE NOOP\n",
ACTIONS = [[0, 0, 0, 0, 0, 0], #0 - no button,\n",
           #AFTER THAT, THE ORDER DOES NOT MATTER\n",
           [1, 0, 0, 0, 0, 0], #1 - up only (to climb vine)\n",
           [0, 0, 1, 0, 0, 0], #2 - left only\n",
           [0, 1, 0, 0, 0, 0], #3 - down only (duck, down pipe)\n",
           [0, 0, 0, 1, 0, 0], #4 - right only\n",
           [0, 0, 0, 0, 0, 1], #5 - run only\n",
           [0, 0, 0, 0, 1, 0], #6 - jump only\n",
           [0, 0, 1, 0, 0, 1], #7 - left run\n",
           [0, 0, 1, 0, 1, 0], #8 - left jump\n",
           [0, 0, 0, 1, 0, 1], #9 - right run\n",
           [0, 0, 0, 1, 1, 0], #10 - right jump\n",
           [0, 0, 1, 0, 1, 1], #11 - left run jump\n",
           [0, 0, 0, 1, 1, 1], #12 - right run jump\n",
           [0, 1, 0, 0, 1, 0]] #13 - down jump\n",

ACTIONS = np.array(ACTIONS)
