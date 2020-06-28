import face_alignment
from skimage import io
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cpu', flip_input=False)
input_ = io.imread('./abc.png')
preds = fa.get_landmarks(input_)
img = cv2.imread('./abc.png')
point_size = 1
point_color = (0, 0, 255) # BGR
thickness = 4 # 可以为 0 、4、8
print(len(preds))
xx = []
yy = []
zz = []
for point in preds[0]:
    print(point)
    xx.append(point[0])
    yy.append(point[1])
    zz.append(point[2])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xx, yy, zz)
plt.show()
