import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import (LinearRegression,RANSACRegressor)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#############################Sub Functions############################

# set ROI of gray scale
def set_gray(img,region):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, region, 255)
    img_ROI=cv2.bitwise_and(img, mask)
    return img_ROI

# set ROI of red scale
def set_red(img,region):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, region, (0,0,255))
    img_red=cv2.bitwise_and(img,mask)
    return img_red

# RANSAC
def RANSAC(x_points,y_points,y_min,y_max):
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    
    y_points = y_points.reshape(len(y_points),1)
    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())

    try :
        model_ransac.fit(y_points, x_points)
    except ValueError : pass
    else :
        line_Y = np.arange(y_min, y_max)
        line_X_ransac = model_ransac.predict(line_Y[:, np.newaxis])
    
        return line_X_ransac
# RANSAC
def S_RANSAC(x_points,y_points,y_min,y_max):
    x_points = np.array(x_points)
    y_points = np.array(y_points)
    
    y_points = y_points.reshape(len(y_points),1)
    model_Sransac = make_pipeline(PolynomialFeatures(2),RANSACRegressor(random_state=42))

    try :
        model_Sransac.fit(y_points, x_points)
    except ValueError : pass
    else :
        line_Y = np.arange(y_min, y_max)
        line_X_ransac = model_Sransac.predict(line_Y[:, np.newaxis])
    
        return line_X_ransac
#################################Main Function##########################

# 값 초기화
L_num = 0
R_num = 0
L_line = [[0,0],[0,0],[0,0]]
R_line = [[0,0],[0,0],[0,0]]

height = 540
width = 960
bird_height = 960
bird_width = 540
height_ROI = 250
L_error = 0
R_error = 0
num_y = bird_height - height_ROI
num = 0
template = cv2.imread('road sample5.png',0)

# read video
cam = cv2.VideoCapture('KATARI.mp4')
if (not cam.isOpened()):
    print ("cam open failed")

while (cam.isOpened()):
#img = cv2.imread('12 2image.jpg',cv2.IMREAD_COLOR)
    s, img = cam.read()
    
    # set cross point
    y1 = 303
    y2 = 520

    # 변형 전 사각점
    L_x1 = 484 #cross_point[0]-(cross_point[1]-y1)/(cross_point[2])
    L_x2 = 80  #cross_point[0]-(cross_point[1]-y2)/(cross_point[2])

    R_x1 = 540 #cross_point[0]-(cross_point[1]-y1)/(cross_point[3])
    R_x2 = 880 #cross_point[0]-(cross_point[1]-y2)/(cross_point[3])

    road_width = R_x2 - L_x2

    # 변형 후 사각점
    Ax1 = height/2-200 #170
    Ax2 = height/2+200 #370
    Ay1 = 0
    Ay2 = 860

    # Homograpy transform
    pts1 = np.float32([[L_x1,y1],[R_x1,y1],[L_x2,y2],[R_x2,y2]])
    pts2 = np.float32([[Ax1,Ay1],[Ax2,Ay1],[Ax1,Ay2],[Ax2,Ay2]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(height,width))

    # red ROI
    temp = np.zeros((bird_height, bird_width, 3), dtype=np.uint8)
    rect = np.array([[(Ax1-50,height_ROI-10),(Ax1-50,bird_height),
                      (Ax2+25,bird_height),(Ax2+25,height_ROI-10)]])

    # cv2.polylines(temp,[rect],True,(0,255,0),2)
    img_red = set_red(dst,rect)

    
    # opening
    kernel = np.ones((2,4), np.uint8)
    img_dilation = cv2.dilate(img_red, kernel, iterations=1)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    
    
    img_canny = cv2.Canny(img_erosion,30,80)

    L_rect = np.array([[(Ax1,height_ROI+10),(Ax1,bird_height-130),
                        (height/2-30,bird_height-60),(height/2-30,height_ROI+10)]])
    R_rect = np.array([[(Ax2,height_ROI+10),(Ax2,bird_height-130),
                        (height/2+10,bird_height-60),(height/2+10,height_ROI+10)]])
     
    cv2.polylines(dst,L_rect,1,(255,255,0),5)
    cv2.polylines(dst,R_rect,1,(0,255,0),5)
    
        
    # canny edge
    L_edge = set_gray(img_canny,L_rect)
    R_edge = set_gray(img_canny,R_rect)

    edge_lx,edge_ly = np.where(L_edge >= 255)
    edge_rx,edge_ry = np.where(R_edge >= 255)

    
    # edge
    for i in range(len(edge_lx)):
        try:
            cv2.circle(dst,(int(edge_ly[i]),int(edge_lx[i])),1,(0,255,255),2)
        except TypeError :
            pass
    for i in range(len(edge_rx)):
        try:
            cv2.circle(dst,(int(edge_ry[i]),int(edge_rx[i])),1,(0,255,255),2)
        except TypeError :
            pass
    '''
    for i in range(len(template_lx)):
        try:
            cv2.circle(dst,(int(template_ly[i]),int(template_lx[i])),1,(255,0,255),2)
        except TypeError :
            pass
    for i in range(len(template_rx)):
        try:
            cv2.circle(dst,(int(template_ry[i]),int(template_rx[i])),1,(255,0,255),2)
        except TypeError :
            pass        
    '''
    # 곡선 RANSAC
    L_Sransac = S_RANSAC(edge_ly, edge_lx, height_ROI,bird_height)
    R_Sransac = S_RANSAC(edge_ry, edge_rx, height_ROI,bird_height)
    
    if num == 0 :
        L_check = L_Sransac
        R_check = R_Sransac
        
    # Error 걸러내기
    try :
        if abs(L_check[num_y-1]-L_Sransac[num_y-1]) > 40 or abs(L_check[0]-L_Sransac[0]) > 40 :
            if L_error % 7 != 0 :
                L_Sransac = L_check
                L_error += 1
            else :
                L_error += 1
        else :
            L_error = 0
    except TypeError : pass
    try :
        if abs(R_check[num_y-1]-R_Sransac[num_y-1]) > 40 or abs(R_check[0]-R_Sransac[0]) > 40:
            if R_error % 7 != 0 :
                R_Sransac = R_check
                R_error += 1
            else :
                R_error += 1
        else :
            R_error = 0
    except TypeError : pass
     
    L_check = L_Sransac
    R_check = R_Sransac
    temp1 = np.zeros((bird_height, bird_width, 3), dtype=np.uint8)
    for i in range(num_y-100):
        try :
            cv2.circle(temp1,(int(L_Sransac[i]),height_ROI+i),1,(0,0,255),2)
        except TypeError :
            pass
        try :
            cv2.circle(temp1,(int(R_Sransac[i]),height_ROI+i),1,(255,0,0),2)
        except TypeError :
            pass
   
    cv2.imshow('bird view.png',dst)
    #cv2.imwrite('bird view.jpg',dst)

    i_M = cv2.getPerspectiveTransform(pts2,pts1)
    i_dst = cv2.warpPerspective(temp1,i_M,(bird_height,bird_width))

    result = cv2.addWeighted(img,0.5,i_dst,6,0.)
    cv2.imshow('result',result)
    #cv2.imwrite('inverse bird view.jpg',i_dst)
    num += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
plt.matshow(i_dst,cmap='gray')
plt.show()
cv2.destroyAllWindows()
cv2.waitKey(0)
    
