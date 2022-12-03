# Import all necessary packages
from imagetransform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# Create argument parser and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required = True,
    help = 'Path to the image to be scanned')
args = vars(ap.parse_args())


# Load images + compute ratio of old height to new height, clone, resize
image = cv2.imread(args['image'])
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height = 500)

# Convert to grayscale, blur, find edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# Show original image + edge-detected image
print('Step One: Edge Detection')
cv2.imshow('Image', image)
cv2.imshow('Edged', edged)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Find contours in edged image, keep largest ones
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

# Loop over contours
for c in cnts:
    # Approximate contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # If 4 points, can assume found
    if len(approx) == 4:
        screenCnt = approx
        break
# Show contour on paper
print('Step Two: Contours on Paper Image')
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow('Outline', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Perspective Transform + Threshold
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# Convert grayscale, threshold to give B&W
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = 'gaussian')
warped = (warped > T).astype('uint8') * 255

# Show original and scanned images
print('Step Three: Perspectie Transform')
cv2.imshow('Original', imutils.resize(orig, height = 650))
cv2.imshow('Scanned Image', imutils.resize(warped, height = 650))
cv2.waitKey(0)