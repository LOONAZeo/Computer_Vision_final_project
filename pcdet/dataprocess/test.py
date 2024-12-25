import numpy as np

point = np.array([433, 302, 646])
cube_size = 64

cube_index = tuple((point // cube_size).astype("int"))
local_point =433%64

print(f'{local_point}')