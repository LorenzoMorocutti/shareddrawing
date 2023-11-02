import numpy as np

with open('/usr/local/src/robot/cognitiveinteraction/container/12_categories/12_categories_NPZ/ambulance/train.npy', 'rb') as f:
      previous_strokes = np.load(f, allow_pickle=True)
f.close()
print("previous_strokes\n", previous_strokes, "\n", len(previous_strokes))
list_strokes = []

copy_strokes=np.copy(previous_strokes).astype('float')
#print(copy_strokes)

strokes = [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
trasp_strokes = np.array(strokes).reshape(len(strokes), 2)
dimensions = np.shape(trasp_strokes)

list_strokes = np.zeros((dimensions[0], dimensions[1]+1))
list_strokes[:, :-1] = trasp_strokes
list_strokes[dimensions[0]-1][dimensions[1]] = 1.0
#print(list_strokes)


