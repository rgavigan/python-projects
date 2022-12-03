# Import all necessary packages
import numpy as np
import cv2

# Function to order points
def order_points(pts):
    ''' Initialize coordinate list that will be ordered
    by: top-left, top-right, bottom-right, bottom-left '''
    rect = np.zeros((4, 2), dtype='float32')

    # Top-left: Smallest Sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    # Bottom-right: Largest sum
    rect[2] = pts[np.argmax(s)]

    ''' Compute difference between points: Top-right
    will have smallest difference, bottom-left will
    have largest difference '''
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return ordered points
    return rect

# Function to transform points
def four_point_transform(image, pts):
    # Obtain ordered points and unpack individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    ''' Compute width of new image - max distance
    between bottom-right and bottom-left x-coords
    or between top-right and top-left x-coords'''
    botWidth = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    topWidth = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(botWidth), int(topWidth))

    # Similar process for height - topR + botR or topL + botL
    rightHeight = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    leftHeight = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(rightHeight), int(leftHeight))

    ''' Construct destination points for birds-eye view
    of the image, specifying poitns in top-left, top-right
    bot-right, bot-left'''
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype='float32')

    # Compute perspective transform matrix and apply
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped