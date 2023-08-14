import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

# 定义立方体的8个顶点
big_cube_pos = np.array([[2, 2, 2],  # 修改顶点坐标，使得一个顶点在原点
                     [2, 2, 0],
                     [2, 0, 0],
                     [2, 0, 2],
                     [0, 2, 2],
                     [0, 2, 0],
                     [0, 0, 0],
                     [0, 0, 2]])

kid_cube_pos = big_cube_pos/3

def get_face(cube_pos):
    return [[cube_pos[0], cube_pos[1], cube_pos[5], cube_pos[4]],
         [cube_pos[7], cube_pos[6], cube_pos[2], cube_pos[3]],
         [cube_pos[0], cube_pos[1], cube_pos[2], cube_pos[3]],
         [cube_pos[7], cube_pos[6], cube_pos[5], cube_pos[4]],
         [cube_pos[0], cube_pos[3], cube_pos[7], cube_pos[4]],
         [cube_pos[1], cube_pos[2], cube_pos[6], cube_pos[5]]]


big_face = get_face(big_cube_pos)
kid_face = get_face(kid_cube_pos)


# 创建一个多边形集合
big_face_collection = Poly3DCollection(big_face, alpha=0.3,edgecolor='k', facecolor=None)  # alpha设置透明度

kid_face_collection = Poly3DCollection(kid_face, alpha=0.3,edgecolor='g', facecolor=None)  # alpha设置透明度


ax.add_collection3d(big_face_collection)
ax.add_collection3d(kid_face_collection)


ax.set_xlim([0, 2])
ax.set_ylim([0, 2])
ax.set_zlim([0, 2])
ax.axis('off')
plt.show()