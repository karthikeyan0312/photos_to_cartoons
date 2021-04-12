import cv2 as cv
import numpy as np

def read_file(fname):
    img=cv.imread(fname)
    ims=cv.resize(img,(500,600))
    #cv.imshow("hi",ims)
    #cv.waitKey(0)
    return img 

def edge_mask(img,size,blur_val):
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gray_blur=cv.medianBlur(gray,blur_val)
    edges=cv.adaptiveThreshold(gray_blur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,size,blur_val)
    
    return edges

def color_quantization(img, k):
# Transform the image
  data = np.float32(img).reshape((-1, 3))
  
# Determine criteria
  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.001)

# Implementing K-Means
  ret, label, center = cv.kmeans(data, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return result
#calling fun read_image()
file_name= # add your file name here
img=read_file(file_name)

line_blue=7
blur_val=7

edges=edge_mask(img,line_blue,blur_val)
#edg=cv.resize(edges,(500,600))
#cv.imshow("new",edg)
#cv.waitKey(0)

total_color=9
img=color_quantization(img,total_color)
edg=cv.resize(img,(500,600))
cv.imshow("new",edg)
#cv.waitKey(0)
blurred = cv.bilateralFilter(img, d=7, sigmaColor=200,sigmaSpace=200)
edg=cv.resize(blurred,(500,600))
cv.imshow("new",edg)
#cv.waitKey(0)
cartoon = cv.bitwise_and(blurred, blurred, mask=edges)
edg=cv.resize(cartoon,(500,600))
cv.imshow("new",edg)
cv.waitKey(0)
