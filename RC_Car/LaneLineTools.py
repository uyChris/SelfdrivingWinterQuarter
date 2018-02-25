import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
import glob
#from scipy.misc import imread, imresize
from skimage.transform import resize
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

file_name = 'Camera_Calibration/fisheye_360x640_calibration.p'

with open(file_name, 'rb') as f:
    mtx, dist = pickle.load(f) # load camera matrix and list of distortion coefficients

#new_mtx = cv2.getOptimalNewCameraMatrix(mtx, dist, (360,640), 0.8)[0]
new_mtx = mtx.copy()
new_mtx[0,0]=mtx[0,0]/0.85
new_mtx[1,1]=mtx[1,1]/0.85
# Just by scaling the matrix coefficients!
# print('mtx: ',mtx)
# print('new_mtx: ',new_mtx)
# print('done')

# Draws viewing window onto copy of image
def draw_viewing_window(image,viewing_window):
    image_copy = np.copy(image)
    for line in viewing_window:
        for x1,y1,x2,y2 in line:
            cv2.line(image_copy,(x1,y1),(x2,y2),(0,255,0),5)
    return image_copy

# Makes the viewwing window which is used by draw_viewing_window()
def make_viewing_window(bottom_left,top_left,top_right,bottom_right):
    left_line = np.array([[bottom_left[0],bottom_left[1],top_left[0],top_left[1]]])
    top_line = np.array([[top_left[0],top_left[1],top_right[0],top_right[1]]])
    right_line = np.array([[top_right[0],top_right[1],bottom_right[0],bottom_right[1]]])
    bottom_line = np.array([[bottom_right[0],bottom_right[1],bottom_left[0],bottom_left[1]]])
    viewing_window = [left_line,top_line,right_line,bottom_line]

    return viewing_window

# Applies sobel algorithm with respect to x or y axis with thresholded gradient
def abs_sobel_thresh(img, orient='x', sobel_kernel=7, thresh_min=0, thresh_max=255):


    # Convert to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take derivative
    if (orient == 'x'):
        sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0 ,ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_32F, 0, 1,ksize=sobel_kernel)

    # Take absolute value of derivative
    abs_sobel = np.absolute(sobel)

    # Convert to 8-bit image (0 - 255)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Make copy of scaled_sobel with all zeros
    sobel_binary = np.zeros_like(scaled_sobel)
    # Make all pixels within threshold range a value of 1
    # Keep all other pixels as 0
    sobel_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    binary_output = sobel_binary
    # binary_output = np.copy(img) # Remove this line
    return binary_output

# Function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def sobel_mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):

    thresh_min = mag_thresh[0]
    thresh_max = mag_thresh[1]

    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take derivatives in both x and y direction
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel)
    # Find magnitude of the gradient
    sum_of_squares = np.square(sobelx) + np.square(sobely)
    sobel_mag = np.power(sum_of_squares,0.5)
    # Convert to 8-bit image (0 - 255)
    scaled_sobel = np.uint8(255*sobel_mag/np.max(sobel_mag))
    # Make a copy of sobel_mag with all zeros
    sobel_binary = np.zeros_like(scaled_sobel)
    # Make all pixels within threshold range a value of 1
    # Keep all other pixels as 0
    sobel_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    binary_output = sobel_binary

    return binary_output

# Function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):

    # Min and Max Threshold Angles
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the derivatives with respect to x and y
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=sobel_kernel)
    # Take absolute value of derivatives
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # Calculate angle for direction of gradient in radians
    sobel_angle = np.arctan2(abs_sobely,abs_sobelx)
    # Make a copy of sobel_angle with all zeros
    sobel_binary = np.zeros_like(sobel_angle)
    # Apply thresholding
    sobel_binary[(sobel_angle >= thresh_min) & (sobel_angle <= thresh_max)] = 1
    binary_output = sobel_binary

    return binary_output

def region_of_interest(gray,limit_look_ahead):
    height = gray.shape[0]
    m = np.copy(gray) + 1
    #m[:, :50] = 0
    #m[:, 590:] = 0
    #m[:,0:50] = 0
    m[:int(limit_look_ahead*height),:] = 0 # cutoff top part of image to limit look-ahead
    #m[440:480,200:400] = 0 # cutoff small rectangle at bottom of image to cover up the car

    return m


# Takes in RGB image and applies color and gradient thresholding
def combined_threshold(image, kernel_size=3, gradx_low_thresh=40, gradx_high_thresh=100,
                       grady_low_thresh=40, grady_high_thresh=100, mag_low_thresh=40,
                       mag_high_thresh=100, dir_low_thresh=0.7, dir_high_thresh=1.3,
                       white_L_low_thresh = 130, white_L_high_thresh = 244, white_S_low_thresh =13,
                       white_S_high_thresh=50, yellow_S_low_thresh=140, yellow_S_high_thresh = 255,
                       yellow_H_low_thresh=20, yellow_H_high_thresh=30, L_agr=205):

    # Find White Lane Line Pixels

    # Convert image to HLS space
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # Extract Just H channel
    H = image_hls[:,:,0]
    H_thresholded = np.zeros_like(H)
    thresh_H = (0,165)
    H_thresholded[(H >= thresh_H[0]) & (H <= thresh_H[1])] = 1

    # Extract just S channel
    S = image_hls[:,:,2]
    S_thresholded = np.zeros_like(S)
    # thresh_S = (140, 255)
    thresh_S = (white_S_low_thresh, white_S_high_thresh)
    S_thresholded[(S >= white_S_low_thresh) & (S <= white_S_high_thresh)] = 1

    # Extract just L channel
    L = image_hls[:,:,1]
    L_thresholded = np.zeros_like(L)
    thresh_L = (white_L_low_thresh, white_L_high_thresh)
    L_thresholded[(L >= white_L_low_thresh) & (L <= white_L_high_thresh)] = 1

    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    A = image_lab[:,:,1]
    A_thresholded = np.zeros_like(A)
    thresh_A = (125,131)
    A_thresholded[(A >= thresh_A[0]) & (A <= thresh_A[1])] = 1

    B = image_lab[:,:,2]
    B_thresholded = np.zeros_like(B)
    thresh_B = (120,129)
    B_thresholded[( B >= thresh_B[0]) & (B <= thresh_B[1])] = 1

    LAB_L = image_lab[:,:,0]
    LAB_L_thresholded = np.zeros_like(LAB_L)
    LAB_L_thresholded[(LAB_L >=121) & (LAB_L <= 237)] = 1

    image_YCrCb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    Cr = image_YCrCb[:,:,1]
    Cr_thresholded = np.zeros_like(Cr)
    #thresh_Cr = (143,154)
    thresh_Cr = (124,127)
    Cr_thresholded[( Cr >= thresh_Cr[0]) & (Cr <= thresh_Cr[1])] =1

    Cb = image_YCrCb[:,:,2]
    Cb_thresholded = np.zeros_like(Cb)
    thresh_Cb = (128,136)
    #thresh_Cb = (0,255)
    Cb_thresholded[( Cb >= thresh_Cb[0] & (Cb <= thresh_Cb[1]))] = 1

    image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    Y = image_yuv[:,:,0]
    Y_thresholded = np.zeros_like(Y)
    thresh_Y = (132,255)
    #thresh_U = (0,255)
    Y_thresholded[((Y >= thresh_Y[0]) & (Y <= thresh_Y[1]))] = 1

    U = image_yuv[:,:,1]
    U_thresholded = np.zeros_like(U)
    thresh_U = (127,135)
    #thresh_U = (0,255)
    U_thresholded[((U >= thresh_U[0]) & (U <= thresh_U[1]))] = 1

    V = image_yuv[:,:,2]
    V_thresholded = np.zeros_like(V)
    thresh_V = (124,129)
    #thresh_V = (0,255)
    V_thresholded[((V >= thresh_V[0]) & (V <= thresh_V[1]))] = 1

    image_luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)

    LUV_U = image_luv[:,:,1]
    LUV_U_thresholded = np.zeros_like(LUV_U)
    LUV_U_thresholded[((LUV_U >= 94) & (LUV_U <= 97))] = 1

    LUV_V = image_luv[:,:,2]
    LUV_V_thresholded = np.zeros_like(LUV_V)
    LUV_V_thresholded[((LUV_V >= 124) & (LUV_V <= 138))] = 1

    image_hsv = cv2.cvtColor(image , cv2.COLOR_RGB2HSV)

    HSV_H = image_hsv[:,:,0]
    HSV_H_thresholded = np.zeros_like(HSV_H)
    HSV_H_thresholded[(HSV_H >= 75) & (HSV_H <= 120)] = 1

    HSV_S = image_hsv[:,:,1]
    HSV_S_thresholded = np.zeros_like(HSV_S)
    HSV_S_thresholded[(HSV_S >= 1) & (HSV_S <= 31)] = 1

    HSV_V = image_hsv[:,:,2]
    HSV_V_thresholded = np.zeros_like(HSV_V)
    HSV_V_thresholded[(HSV_V >= 124) & (HSV_V <= 255)] = 1

    white_thresholded = np.zeros_like(B)
    #white_thresholded[((Cr_thresholded == 1) & (L_thresholded == 1) & (H_thresholded == 1) & (S_thresholded == 1) & (V_thresholded == 1) \
    #                   & (U_thresholded == 1) & (LUV_V_thresholded == 1) & (LUV_U_thresholded == 1) & (Cb_thresholded == 1)
    #                  \ & (A_thresholded == 1) & (B_thresholded == 1) & (HSV_H_thresholded == 1) & (HSV_S_thresholded == 1) & \
    #                  (LAB_L_thresholded == 1))] = 1
    white_thresholded[((S_thresholded == 1) & (HSV_V_thresholded == 1) & (Y_thresholded == 1) \
                        & (V_thresholded == 1) & (B_thresholded == 1) & (LUV_U_thresholded == 1) \
                        & (LUV_V_thresholded == 1))] = 1

    # Find Yellow Lane Line Pixels

    # Extract just S channel
    S = image_hls[:,:,2]
    S_thresholded = np.zeros_like(S)
    # thresh_S = (140, 255)
    thresh_S = (yellow_S_low_thresh, yellow_S_high_thresh)
    S_thresholded[(S >= yellow_S_low_thresh) & (S <= yellow_S_high_thresh)] = 1

    # Extract just H channel

    H = image_hls[:,:,0]
    H_thresholded = np.zeros_like(H)
    thresh_H = (yellow_H_low_thresh, yellow_H_high_thresh)
    H_thresholded[(H >= yellow_H_low_thresh) & (H <= yellow_H_high_thresh)] = 1

    image_luv = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)

    LUV_U = image_luv[:,:,1]
    LUV_U_thresholded = np.zeros_like(LUV_U)
    LUV_U_thresholded[((LUV_U >= 91) & (LUV_U <= 96))] = 1

    LUV_V = image_luv[:,:,2]
    LUV_V_thresholded = np.zeros_like(LUV_V)
    LUV_V_thresholded[((LUV_V >= 160) & (LUV_V <= 160))] = 1

    image_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

    YUV_V = image_yuv[:,:,2]
    YUV_V_thresholded = np.zeros_like(YUV_V)
    YUV_V_thresholded[((YUV_V >= 125) & (YUV_V) <= 129)] = 1

    yellow_thresholded = np.zeros_like(H)
    yellow_thresholded[(H_thresholded == 1) & (S_thresholded == 1)] = 1

    # Keep very brigh pixels by looking at just L channel

    # Extract just L channel
    L = image_hls[:,:,1]
    #L_thresholded = np.zeros_like(L)
    #thresh_L = (L_low_thresh, L_high_thresh)
    #L_thresholded[(L >=thresh_L[0]) & (L <= thresh_L[1])] = 1
    thresh_L_agr = L_agr
    L_thresholded2 = np.zeros_like(L)
    L_thresholded2[L>=thresh_L_agr] = 1

    # Calculate gradients

    # gradx original: (40,255)
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=kernel_size, thresh_min=gradx_low_thresh,
                             thresh_max=gradx_high_thresh)
    # I did not use grady in the final output
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=kernel_size, thresh_min=grady_low_thresh,
                             thresh_max=grady_high_thresh)
    # mag_binary original: (40,255)
    mag_binary = sobel_mag_thresh(image, sobel_kernel=kernel_size, mag_thresh=(mag_low_thresh,mag_high_thresh))
    dir_binary = dir_threshold(image,sobel_kernel=kernel_size,thresh=(dir_low_thresh,dir_high_thresh))


    # Combine all the thresholds

    combined = np.zeros_like(dir_binary)



    # Mithi's gradient and color thresholds
    #combined[((mag_binary == 1) & (dir_binary == 1) & (gradx == 1)) | \
    #          (((S_thresholded == 1) & (L_thresholded == 1)) | (L_thresholded2 == 1))] = 1

    # Adam's gradient and color thresholds
    combined[((mag_binary == 1) &  (gradx == 1) & (dir_binary == 1)) | \
              ( (white_thresholded == 1) | (yellow_thresholded == 1) | (L_thresholded2 == 1))] = 1

    #combined[((mag_binary == 1) & (dir_binary == 1) & (S_thresholded == 1) & (L_thresholded == 1))] = 1

    #combined = np.logical_and(combined,region_of_interest(combined)).astype(np.uint8)

    return combined

# image is an undistorted image
def apply_birdseye(image,source_points,dest_points):

    source_points = np.asarray(source_points)
    dest_points = np.asarray(dest_points)
    M = cv2.getPerspectiveTransform(source_points,dest_points)
    Minv = cv2.getPerspectiveTransform(dest_points,source_points)
    img_size = (image.shape[1],image.shape[0])

    birds_eye_image = cv2.warpPerspective(image, M, img_size,
                                          flags=cv2.INTER_LINEAR)

    # Applying Combined Color and Gradient Thresholding to Birds Eye View Image
    #combined = combined_threshold(birds_eye_image,kernel_size=11)


    #return combined
    return birds_eye_image
"""
video_number = 4
image_paths = glob.glob('RC_Car_Images_And_Videos/Hershels_Garage/Fisheye/video%d/test_images/*.jpg' % video_number)
num_pictures = len(image_paths)
f, axs = plt.subplots(4,4, figsize=(10,15))
axs = axs.ravel()
for i,img in enumerate(image_paths):
    #img = 'hershels_test_images/frame' + str(i) + '.jpg'
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print('image.shape: ',image.shape)
    #image_undistorted = cv2.undistort(image,mtx,dist,None,mtx)
    image_undistorted = cv2.undistort(image,mtx,dist,None,mtx)
    axs[i*2].imshow(image)
    axs[i*2].set_title('original')
    axs[i*2+1].imshow(image_undistorted)
    axs[i*2+1].set_title('undistorted')

plt.show()
"""
