from LaneLineTools import *
from multiprocessing import Process
from multiprocessing import Pool

from multiprocessing.pool import ThreadPool
import sys
import threading
import multiprocessing as mp
from threading import Thread





video_number = 4

image_paths = glob.glob('RC_Car_Images_And_Videos/Hershels_Garage/Fisheye/video%d/test_images/*.jpg' % video_number)
images = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

"""
Applies undistortion and birdseye view to input image
returns resulting image
"""
import time
def stage1(image):
    print('start stage1')
    t1 = time.time()
    # source points for 360x640 using fisheye in hershel's garage
    src_bottom_left = [-50,360]
    src_top_left = [150,200]
    src_top_right = [490,200]
    src_bottom_right = [690,360]
    source_points = np.float32([src_bottom_left,src_top_left,src_top_right,
                     src_bottom_right])

    bottom = 360
    top = 0
    left = 0
    right = 640
    dest_bottom_left = [left,bottom]
    dest_top_left = [left,top]
    dest_top_right = [right,top]
    dest_bottom_right = [right,bottom]
    dest_points = np.float32([dest_bottom_left,dest_top_left,dest_top_right,
                              dest_bottom_right])


    M = cv2.getPerspectiveTransform(source_points,dest_points)
    Minv = cv2.getPerspectiveTransform(dest_points,source_points)
    # undistort image
    # image = cv2.undistort(image, mtx, dist, None, mtx)
    #image_undistorted = image
    #birdseye_view = apply_birdseye(image,source_points,dest_points)
    img_size = (image.shape[1],image.shape[0])

    birds_eye_image = cv2.warpPerspective(image, M, img_size,
                                          flags=cv2.INTER_LINEAR)
    print(image.shape)

    print('finish stage1')
    t2 = time.time()
    print('stage1 time: ', t2-t1)
    return image

"""
Applies color and gradient thresholds to image
returns resulting image
"""
def stage2(birdseye_view):
    t1 = time.time()
    print('start stage2')
    # Applying Combined Color and Gradient Thresholding to Birds Eye View Image
    combined = combined_threshold(birdseye_view, kernel_size=3, gradx_low_thresh=30, gradx_high_thresh=255,
                                  grady_low_thresh=40, grady_high_thresh=255, mag_low_thresh=30,
                                  mag_high_thresh=254, dir_low_thresh=0.7, dir_high_thresh=1.3,
                                  white_L_low_thresh=117, white_L_high_thresh=190, white_S_low_thresh=12,
                                  white_S_high_thresh=255,yellow_S_low_thresh=30, yellow_S_high_thresh=140,
                                  yellow_H_low_thresh=28, yellow_H_high_thresh=53, L_agr=225)

    combined = np.logical_and(combined,region_of_interest(combined,limit_look_ahead=limit_look_ahead)).astype(np.uint8)
    print('finish stage2')
    t2 = time.time()
    print('stage2 time: ', t2-t1)
    return combined

limit_look_ahead = 0.4 # increase number to limit look ahead (1.0 is maximum; 0.0 is minimum)
"""
Performs right lane line window search
returns coefficients of best-fit line for right lane line as a list
"""
def stage3(combined):
    t1 = time.time()
    print('start stage3')
    binary_warped = combined
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/1.35):,:], axis=0) # 1.25
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 3
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    righty_current = binary_warped.shape[0] - window_height//2
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    #margin = 80
    margin = 120 # best is 120
    window_height = binary_warped.shape[0]//nwindows




    # Set minimum number of pixels found to recenter window
    minpix = 500 # best is 50 (500)
    # Create empty lists to receive left and right lane pixel indices
    right_lane_inds = []

    # Re-center window based on both x and y position


    window = 0
    # Initialize top, bottom, left, and right boundaries of right search windows
    win_yright_low = righty_current + window_height//2
    win_yright_high = righty_current - window_height//2
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    # Initialize the direction the right window searches move in
    right_dx = 0
    right_dy = -1

    # margin of wiggle room before stopping window search when it exits the side of the image
    side_margin = 1.5 #1.25
    # margin of wiggle room before stopping window search when it crosses into other half of image
    middle_margin = 2.0

    n_right_windows = 0 # Initialize the number of right windows used
    min_n_windows = 100 # min number of windows before terminating window search
    # While
    # ((right window search is within right side of image) OR (right window count is less than min_n_windows)))
    while ((((win_xright_low >= (binary_warped.shape[1]//2 - ((margin//2)*middle_margin))) & (win_xright_high <= (binary_warped.shape[1] + (margin//2)*side_margin)) & (win_yright_high > (limit_look_ahead)*binary_warped.shape[0])) | (n_right_windows < min_n_windows))):

        # Do right lane line
        # Find left and right boundaries of window
        win_yright_low = righty_current + window_height//2
        win_yright_high = righty_current - window_height//2
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        area = (win_yright_low-win_yright_high)*(win_xright_high-win_xright_low)
        #print('window area: ',area)
        # Stop performing right window search if right lane line exits right side of image
        if (((win_xright_high <= binary_warped.shape[1] + (margin//2)*side_margin) & (win_xright_low >=(binary_warped.shape[1]//2 - (margin//2)*middle_margin)) | (n_right_windows < min_n_windows))): # 1.5
            n_right_windows += 1
            # Draw Window
            cv2.rectangle(out_img,(win_xright_low,win_yright_low),(win_xright_high,win_yright_high),
            (0,255,0), 2)
            # Get indicies of nonzero pixels within window
            good_right_inds = ((nonzeroy < win_yright_low) & (nonzeroy >= win_yright_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indicies to list of right lane line indicies
            right_lane_inds.append(good_right_inds)
            # if you found > minpix pixels, recenter next window on mean x-position
            #print('len(good_right_inds): ',len(good_right_inds))
            if len(good_right_inds) > minpix:
                rightx_previous = rightx_current
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
                right_dx = rightx_current - rightx_previous
                if (np.int(np.mean(nonzeroy[good_right_inds])) < righty_current):
                    righty_previous = righty_current
                    righty_current = np.int(np.mean(nonzeroy[good_right_inds]))
                    right_dy = righty_current - righty_previous
                else:
                    #righty_current -= 1
                    righty_current += right_dy
            else:
                #righty_current -= 1
                rightx_current += right_dx
                righty_current += right_dy

    # Concatenate the arrays of indices
    if (len(right_lane_inds) > 0):
        right_lane_inds = np.concatenate(right_lane_inds)

    # Extractright line pixel positions
    right_lane_inds = np.unique(right_lane_inds) # get rid of repeats

    # Temporary fix
    if (len(right_lane_inds) > 0):
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
    else:
        rightx = []
        righty = []

    ploty = np.linspace(limit_look_ahead*binary_warped.shape[0], (binary_warped.shape[0]-1)*1.25, binary_warped.shape[0] )

    if ((len(righty) > 0) & (len(rightx) > 0)):
        #right_fit = np.polyfit(righty, rightx, 2)

        right_model = make_pipeline(PolynomialFeatures(2), RANSACRegressor(random_state=42))
        right_model.fit(righty.reshape(-1,1), rightx)

        # getting coefficients of best-fit curve
        # for some reason the one line of code below fails to return the correct coefficients
        # it only returns 2/3 of the coefficients
        # one coefficient is always zero...
        # so I have to re-calculate the coefficients using np.polyfit()
        # we need the coefficients of the best-fit line to calculate radius of curvature
        #right_fit = right_model.named_steps['ransacregressor'].estimator_.coef_

        right_fitx = right_model.predict(ploty.reshape(-1,1))
        right_fit = np.polyfit(ploty,right_fitx,2)


    else:
        right_fit = [0,0,640]
        #right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    print('finish stage3')
    t2 = time.time()
    print('stage3 time: ', t2-t1)
    return right_fit

"""
Performs left lane line window search
returns coefficients of best-fit line for left lane line as a list
"""
def stage4(combined):
    t1 = time.time()
    print('start stage4')
    binary_warped = combined
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/1.35):,:], axis=0) # 1.25
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])

    # Choose the number of sliding windows
    nwindows = 3
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    lefty_current = binary_warped.shape[0] - window_height//2


    # Set the width of the windows +/- margin
    #margin = 80
    margin = 120 # best is 120
    window_height = binary_warped.shape[0]//nwindows




    # Set minimum number of pixels found to recenter window
    minpix = 500 # best is 50 (500)
    # Create empty lists to receive left lane line pixel indices
    left_lane_inds = []

    # Re-center window based on both x and y position

    window = 0
    # Initialize top, bottom, left, and right boundaries of left search window
    win_yleft_low = lefty_current + window_height//2
    win_yleft_high = lefty_current - window_height//2
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin

    # Initialize the direction the left window searches move in
    left_dx = 0
    left_dy = -1

    # margin of wiggle room before stopping window search when it exits the side of the image
    side_margin = 1.5 #1.25
    # margin of wiggle room before stopping window search when it crosses into other half of image
    middle_margin = 2.0

    n_left_windows = 0 # Initialize the number of left windows used
    min_n_windows = 100 # min number of windows before terminating window search
    # While
    # (((left window search is within left side of image) OR (left window count is less than min_n_windows))
    while (((win_xleft_low >= -1*(margin//2)*side_margin) & (win_xleft_high <= (binary_warped.shape[1]//2 + ((margin//2)*middle_margin))) & (win_yleft_high > (limit_look_ahead)*binary_warped.shape[0])) | (n_left_windows < min_n_windows)):
        #window += 1
        # Do left lane line
        # Find left, right, top, bottom, boundaries of window
        win_yleft_low = lefty_current + window_height//2
        win_yleft_high = lefty_current - window_height//2
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        #print('window number: ',window)
        # Stop performing left window search if left lane line exits left side of image
        if (((win_xleft_low >= -1*(margin//2)*side_margin) & (win_xleft_high <= binary_warped.shape[1]//2 + (margin//2)*middle_margin)) | (n_left_windows < min_n_windows)): # 1.5
            n_left_windows += 1
            # Draw window
            cv2.rectangle(out_img,(win_xleft_low,win_yleft_low),(win_xleft_high,win_yleft_high),
            (0,255,0), 2)
            # Get indicies of nonzero pixels within window
            good_left_inds = ((nonzeroy < win_yleft_low) & (nonzeroy >= win_yleft_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            # Append these indicies to list of left lane line indicies
            left_lane_inds.append(good_left_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                #print('found minpix')
                # Always re-center x position; let new x position go to the left or right
                leftx_previous = leftx_current
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                left_dx = leftx_current - leftx_previous
                # Only re-center y position if the new center is higher up on the image than the previous center
                # higher up on the image means a smaller y value
                # 0 y value is at the top of the image
                if (np.int(np.mean(nonzeroy[good_left_inds])) < lefty_current):
                    lefty_previous = lefty_current
                    lefty_current = np.int(np.mean(nonzeroy[good_left_inds]))
                    left_dy = lefty_current - lefty_previous
                # If the re-centering of the y position causes the window to stay in the same spot or go back down
                # then force the window to move one pixel up the image (y value goes down)
                # This way the window search does not get stuck
                # But it still moves up slowly enough that it does not miss
                # lane lines that are far apart horizontally
                else:
                    lefty_current += left_dy
            else:
                #print('didnt find minpix')
                leftx_current += left_dx
                lefty_current += left_dy


    # Concatenate the arrays of indices
    if (len(left_lane_inds) > 0):
        left_lane_inds = np.concatenate(left_lane_inds)

    # Extract left line pixel positions
    left_lane_inds = np.unique(left_lane_inds) # get rid of repeats

    # Temporary fix
    if (len(left_lane_inds) > 0):
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
    else:
        leftx = []
        lefty = []

    ploty = np.linspace(limit_look_ahead*binary_warped.shape[0], (binary_warped.shape[0]-1)*1.25, binary_warped.shape[0] )

    if ((len(lefty) > 0) & (len(leftx) > 0)):
        #left_fit = np.polyfit(lefty, leftx, 2)

        left_model = make_pipeline(PolynomialFeatures(2), RANSACRegressor(random_state=42))
        left_model.fit(lefty.reshape(-1,1), leftx)

        # getting coefficients of best-fit curve
        # for some reason the one line of code below fails to return the correct coefficients
        # it only returns 2/3 of the coefficients
        # one coefficient is always zero...
        # so I have to re-calculate the coefficients using np.polyfit()
        # we need the coefficients of the best-fit line to calculate radius of curvature
        #left_fit = left_model.named_steps['ransacregressor'].estimator_.coef_

        left_fitx = left_model.predict(ploty.reshape(-1,1))
        left_fit = np.polyfit(ploty,left_fitx,2)



    else:
        left_fit = [0,0,0]
        #left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]

    print('finish stage4')
    t2 =time.time()
    print('stage4 time: ',t2-t1)
    return left_fit

print('done')


#rocket = 0
def func1(rocket):
    t1 = time.time()
    rocket = 0
    a = 0
    print('start func1')
    while rocket < 10000000:
        rocket += 1
        a += 4
    t2 = time.time()
    print('func1 time: ',t2-t1)
    print ('end func1')
    return a

def func2(rocket):
    t1 = time.time()
    rocket = 0
    print ('start func2')
    while rocket < 10000000:
        rocket += 1
    print ('end func2')
    t2 = time.time()
    print('func2 time: ',t2-t1)
    return rocket

if __name__ == '__main__':
    """
    #successfully parallelized; there is a speedup
    print('parallel approach')
    pool = Pool(2)
    t1 = time.time()
    result1 = pool.apply_async(func=func1, args=(rocket,))
    result2 = pool.apply_async(func=func2, args=(rocket,))
    #print(result1.get())
    #print(result2.get())
    result1.wait()
    result2.wait()
    t2 = time.time()
    print('total parallel time: ',t2-t1)

    print()
    print('sequential approach')
    t1 = time.time()
    result1 = func1(rocket)
    result2 = func2(rocket)
    t2 = time.time()
    print('total sequential time: ',t2-t1)
    """

    m = mp.Manager()  # use a manager, Queue objects cannot be shared
    q = m.Queue()
    work_queue = m.Queue()


    print("Found %d CPUs." % mp.cpu_count())
    print("Operation queue has %d items." % work_queue.qsize())
    #Sequential Approach
    #print('sequential approach')
    #t1 = time.time()
    #result1 = stage1(images[0])
    #result2 = stage2(images[0])
    #t2 = time.time()
    #print('total sequential time: ',t2-t1)

    #Parallel Approach




    #Uses Two Workers
    pool = Pool(2)

    #Takes in the time and records it
    t1 = time.time()

    #Takes the first part of the lane line detection and the second part
    #and parallizes the process
    #func is the argument name taking in the  function name of the lane line detection
    #args is the argument for the function.
    #images in this case takes the first image in the images array

    #Original code
    result1 = pool.apply_async(func=stage1, args=(images[0],))
    result2 = pool.apply_async(func=stage2, args=(images[0],))

    #result1 = pool.apply(func=stage1, args=(images[0],))
    #result2 = pool.apply(func=stage2, args=(images[1],))

    #pool.close()
    #pool.join()
    print(result1.get())
    print(result2.get())
    #cv2.imshow(result1.get())
    #result2.wait()
    t2 = time.time()
    print('total parallel time: ',t2-t1)


    """
    multithreading does not speed up anything

    pool = ThreadPool(processes=2)
    t1 = time.time()
    result1 = pool.apply_async(func1, args=(rocket,)) # tuple of args for foo
    result2 = pool.apply_async(func2, args=(rocket,))
    result1.wait()
    result2.wait()
    t2 = time.time()
    print('parallel total time: ', t2-t1)

    print()
    print('sequential approach')
    t1 = time.time()
    result1 = func1(rocket)
    result2 = func2(rocket)
    t2 = time.time()
    print('total sequential time: ',t2-t1)
    """
    """
    cap = cv2.VideoCapture(0)

    frame_height = 360
    frame_width = 640

    cap.set(3,frame_height)
    cap.set(4,frame_width)

    count = 0


    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        else:
            print('not true')
            break
    """
    """
    pool = Pool(4)
    t1 = time.time()
    #result1 = pool.apply_async(func=func1, args=(rocket,))
    #result2 = pool.apply_async(func=func2, args=(rocket,))
    #print(result1.get())
    #print(result2.get())
    for i, image in enumerate(images):
        if (i == 0):
            result1 = pool.apply_async(func=stage1, args=(image,))

            birdseye_image = result1.get()
        elif(i == 1):
            result1 = pool.apply_async(func=stage1, args=(image,))
            result2 = pool.apply_async(func=stage2, args=(birdseye_image,))

            birdseye_image = result1.get()
            combined = result2.get()
        elif(i >= 2):
            result1 = pool.apply_async(func=stage1, args=(image,))
            result2 = pool.apply_async(func=stage2, args=(birdseye_image,))
            result3 = pool.apply_async(func=stage3, args=(combined,))
            result4 = pool.apply_async(func=stage4, args=(combined,))

            birdseye_image = result1.get()
            combined = result2.get()
            right_fit = result3.get()
            left_fit = result4.get()
    """
