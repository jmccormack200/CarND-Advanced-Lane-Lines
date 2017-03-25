import numpy as np
import cv2

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
