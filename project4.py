import numpy as np
import cv2
import matplotlib.pyplot as plt

def warper(img, src, dst):

    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped

def binaryImage(image, sobel_thresh=[0, 255], l_thresh=[0, 255], s_thresh=[0,255]):
    # Create a copy to edit
    image_copy = np.copy(image)

    # We then use the HLS color space.
    hls_image = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HLS)
    l_channel = hls_image[:, :, 1]
    s_channel = hls_image[:, :, 2]

    # Next we use an X direction Sobel
    # 1, 0 below for x direction
    sobel_x =  cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    # Then we scale
    abs_sobel_x = np.absolute(sobel_x)
    scaled_sobel_x = np.uint8( 255 * abs_sobel_x / np.max(abs_sobel_x))

    # Now create a binary image from the our threshold values
    sobel_x_binary = binaryImageHelper(scaled_sobel_x, sobel_thresh[0], sobel_thresh[1])
    l_channel_binary = binaryImageHelper(l_channel, l_thresh[0], l_thresh[1])
    s_channel_binary = binaryImageHelper(s_channel, s_thresh[0], s_thresh[1])

    # Then combine the three thresholds together
    combined = np.zeros_like(l_channel)
    combined[((l_channel_binary == 1) & (s_channel_binary == 1) | (sobel_x_binary == 1))] = 1
    combined = prepImgForOut(combined)
    sobel_x_binary = prepImgForOut(sobel_x_binary)
    l_channel_binary = prepImgForOut(l_channel_binary)
    s_channel_binary = prepImgForOut(s_channel_binary)

    return combined, sobel_x_binary, l_channel_binary, s_channel_binary

def binaryImageHelper(image, min_threshold, max_threshold):
    zeros_image = np.zeros_like(image)
    zeros_image[(image >= min_threshold) & (image <= max_threshold)] = 1
    return zeros_image

def prepImgForOut(image):
    image = cv2.GaussianBlur(image,(5,5),0)
    return 255 * np.dstack((image, image, image)).astype('uint8')

def warp(img):

    img_size = (img.shape[1], img.shape[0])

    src = np.float32([[230, 690],[590, 450],[685, 450],[1075, 690]])
    dst = np.float32([[300, 690],[300, 0],[970, 0],[1000, 690]])

    M = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

def unwarp(img):
    img_size = (img.shape[1], img.shape[0])

    src = np.float32([[230, 690],[590, 450],[685, 450],[1075, 690]])
    dst = np.float32([[300, 690],[300, 0],[970, 0],[1000, 690]])

    M = cv2.getPerspectiveTransform(dst, src)

    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, mtx, dist):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        self.mtx = mtx
        self.dist = dist
        self.left_fit = []
        self.right_fit = []

    def analyze(self, input_image):
        img = np.copy(input_image)

        # 1. Undistort Image
        undistorted_img = self.undistort(img)
        # 2. Warp Image
        warped_img = warp(undistorted_img)
        # 3. Create Binary Representation
        binary_imgs = binaryImage(warped_img, sobel_thresh=[20, 100], l_thresh=[90, 255], s_thresh=[170,255])
        binary_img = binary_imgs[0]


        if self.detected:
            fit_left, fit_right = self.repeat_lane_finder(binary_img)
        else:
            fit_left, fit_right = self.first_lane_finder(binary_img)

        fill_img = self.fill_lanes(binary_img, fit_left, fit_right)

        merge_imgs = self.merge_imgs(fill_img, img)

        return merge_imgs

    def undistort(self, img):
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def first_lane_finder(self, img):
        histogram = np.sum(img[img.shape[0]//2:,:,0], axis=0)

        mid = np.int(histogram.shape[0] / 2)

        leftx_base = np.argmax(histogram[0:mid])
        rightx_base = np.argmax(histogram[mid:]) + mid # Add mid as an offset for splitting the image in half
        print(leftx_base)
        print(rightx_base)

        binary_warped = img

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 75
        # Set minimum number of pixels found to recenter window
        minpix = 35
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

        self.detected = True

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        return left_fitx, right_fitx

    def repeat_lane_finder(self, img):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] - margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + self.left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] - margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + self.right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        return left_fitx, right_fitx

    def fill_lanes(self, binary_img, left_fitx, right_fitx):
        ploty = np.linspace(0, binary_img.shape[0]-1, binary_img.shape[0] )
        margin = 25

        window_img = np.zeros_like(binary_img)

        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255,0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255,0))

        return window_img

    def merge_imgs(self, binary_img, original_img):
        cp_bin = np.copy(binary_img)
        cp_original = np.copy(original_img)

        cp_bin = unwarp(cp_bin)
        return cv2.addWeighted(cp_original, 1, cp_bin, 0.3, 0)
