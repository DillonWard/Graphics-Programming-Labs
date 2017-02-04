import cv2
import numpy as np
from matplotlib import pyplot as plt

nrows = 3
ncols = 4
kernelWidth = 3
kernelHeight = 3
cannyThreshold = 100
sobelThreshold = 20

img = cv2.imread('GMIT.jpg',)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.png', gray_image)
blurImg = cv2.GaussianBlur(gray_image,(kernelWidth, kernelHeight),0)
blurImg = cv2.GaussianBlur(gray_image,(kernelWidth + 10, kernelHeight + 10),0)
sobelH = cv2.Sobel(gray_image,cv2.CV_64F,1,0,ksize=5) # x dir
sobelV = cv2.Sobel(gray_image,cv2.CV_64F,0,1,ksize=5) # y dir
sobelComp = sobelH + sobelV
canny = cv2.Canny(gray_image,cannyThreshold,200)

#cv2.imshow('color_image', img)
#cv2.imshow('gray_image', gray_image)

plt.subplot(nrows, ncols,1),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(gray_image, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,3),plt.imshow(blurImg, cmap = 'gray')
plt.title('Blur 3 x 3'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,4),plt.imshow(blurImg, cmap = 'gray')
plt.title('Blur 13 x 13'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,5),plt.imshow(sobelH, cmap = 'gray')
plt.title('Sobel Horizontal'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,6),plt.imshow(sobelV, cmap = 'gray')
plt.title('Sobel Vertical'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,7),plt.imshow(sobelComp, cmap = 'gray')
plt.title('Sobel Combination'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,8),plt.imshow(canny, cmap = 'gray')
plt.title('Canny'), plt.xticks([]), plt.yticks([])

cv2.waitKey(0)
cv2.destroyAllWindows() 
plt.show()