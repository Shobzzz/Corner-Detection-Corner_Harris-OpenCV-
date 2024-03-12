import cv2
import numpy as np
import matplotlib.pyplot as plt




flat_chess = cv2.imread('/home/nlab/OpenCv_Tutorials_Shobhit/Object Detection/chess.jpg')
flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2RGB)
plt.imshow(flat_chess)
plt.show()

gray_flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_flat_chess,cmap='gray')

gray = np.float32(gray_flat_chess)

dst = cv2.cornerHarris(src=gray,blockSize=2,ksize=3,k=0.04)
dst = cv2.dilate(dst,None)
flat_chess[dst>0.01*dst.max()] = [255,0,0]

plt.imshow(flat_chess)
plt.show()


real_chess = cv2.imread('/home/nlab/OpenCv_Tutorials_Shobhit/Object Detection/real_chess.jpg')
real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2RGB)

plt.imshow(real_chess)
plt.show()
gray_real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_flat_chess,cmap='gray')

gray = np.float32(gray_real_chess)
dst1 =cv2.cornerHarris(src=gray,blockSize=2,ksize=3,k=0.04)

dst1= cv2.dilate(dst1,None)

real_chess[dst1>0.01*dst1.max()] = [0,0,255]
plt.imshow(real_chess)
plt.show()


me= cv2.imread('/home/nlab/OpenCv_Tutorials_Shobhit/Object Detection/group.jpg')
me = cv2.cvtColor(me,cv2.COLOR_BGR2RGB)
plt.imshow(me)
plt.show()

gray_me = cv2.cvtColor(me,cv2.COLOR_BGR2GRAY)
plt.imshow(gray_me,cmap='gray')

gray = np.float32(gray_me)
dst2 = cv2.cornerHarris(src=gray,blockSize=2,ksize=3,k=0.04)
dst2 = cv2.dilate(dst2,None)
me[dst2 > 0.01*dst2.max()] = [0,0,255]

plt.imshow(me)
plt.show()




