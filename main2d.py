import face_alignment
from skimage import io
import cv2

# 2D
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu', flip_input=False)
input_ = io.imread('./abc.png')
preds = fa.get_landmarks(input_)
img = cv2.imread('./abc.png')
point_size = 1
point_color = (0, 0, 255) # BGR
thickness = 4 # 可以为 0 、4、8
print(len(preds))
for point in preds[0]:
    print(point)
    po1 = int(point[0])
    po2 = int(point[1])
    point_ = (po1, po2)
    cv2.circle(img, point_, point_size, point_color, thickness)
cv2.imwrite('abc2.png', img)

cv2.imshow('img', img)
cv2.waitKey(0)
