import math
import numpy as np
import matplotlib.pyplot as plt

def generate_icosphere():
    """ 
    Generate icosphere """
    # golden ratio
    r = (1.0 + math.sqrt(5.0)) / 2.0

    scale = 1.0/np.sqrt(r**2.0 + 1)
    
    vertices = scale*np.array([
        [-1.0,   r, 0.0],
        [ 1.0,   r, 0.0],
        [-1.0,  -r, 0.0],
        [ 1.0,  -r, 0.0],
        [0.0, -1.0,   r],
        [0.0,  1.0,   r],
        [0.0, -1.0,  -r],
        [0.0,  1.0,  -r],
        [  r, 0.0, -1.0],
        [  r, 0.0,  1.0],
        [ -r, 0.0, -1.0],
        [ -r, 0.0,  1.0],
        ], dtype=float)

    faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [5, 4, 9],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
        ])

    return(vertices,faces)

vertices,faces=generate_icosphere()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(vertices[:,0],vertices[:,1],vertices[:,2],"o")


for i in range(vertices.shape[0]):
    print(np.sqrt(np.sum(vertices[i,:]**2.0)))

for i in range(faces.shape[0]):
    idx=faces[i,:]
    print(idx)
    idxp = np.concatenate((idx,[idx[0]]))
    ax.plot(vertices[idxp,0],vertices[idxp,1],vertices[idxp,2],color="blue")
print(faces.shape)
plt.show()
