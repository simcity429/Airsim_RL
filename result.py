import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
SMOOTH_NUM = 20
def smooth(l):
    if len(l) < SMOOTH_NUM:
        return l
    tmp = []
    current_sum = 0
    for i in range(len(l)):
        current = l[i]
        current_sum += current
        tmp.append(current_sum/(i+1))
        if i == SMOOTH_NUM-2:
            break
    for i in range(SMOOTH_NUM-1, len(l)):
        tmp.append(sum(l[i-(SMOOTH_NUM-1):i+1])/SMOOTH_NUM)
    l = tmp
    return l

result = pd.read_csv('output_true.csv')
y_val = smooth(np.array(result['y_val']))
index = np.arange(2000)
plt.plot(index, y_val)
plt.savefig("stat.png")