#%% Script to find the sudoku board and extract digits from the board.

# Imports
import numpy as np
import cv2
import imutils
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border

# Logs
import logging as log
log.basicConfig(level=log.INFO,
                format='[%(filename)s:%(lineno)s] %(message)s',
                datefmt="%I:%M:%S %p")

#%% Sudoku board finder
def find_puzzle(image, debug=False):

    if not debug:
        log.basicConfig(level=log.INFO)
    
    # Convert to grayscale and blur the image to remove noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # Adaptive thresholding to get the puzzle and then invert
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # Check if debug mode is on and if so, show the image
    if debug:
        cv2.imshow("Puzzle Threshold", thresh)
        cv2.waitKey(0)

    # Find the contours in the thresholded image
    # we will sort them by size in descending order. In theory, sudoku board will be the bigger one.
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Initialize a contur that corresponds to the puzzle outline
    puzzleCnt = None

    # Loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True) # perimeter of the contour
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we can
        # assume that we have found the puzzle outline
        if len(approx) == 4:
            puzzleCnt = approx
            log.debug('[DEBUG] Puzzle contour found')
            break

    # If there's no countour, raise an error
    if puzzleCnt is None:
        log.error('[ERROR] No puzzle contour found')
        raise Exception('[ERROR] No puzzle contour found')

    # If debug mode is on, show the contour
    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Contour", output)
        cv2.waitKey(0)


    # Apply a perspective transform to the puzzle contour to obtain a 
    # top-down view of the puzzle. This will be applyed in both grayscale and original image.
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

    # If debug mode is on, show the warped image
    if debug:
        cv2.imshow("Puzzle Transformed", puzzle)
        cv2.waitKey(0)

    # Return the puzzle and the warped image
    return puzzle, warped


#%% Debugging
if __name__ == '__main__':

    log.info('[INFO] Testing the puzzle finder')
    
    # Load the image
    image = cv2.imread('D:\Programacion\ComputerVision\sudoku-solver\sudoku_test.jpg.png')

    # Find the puzzle
    find_puzzle(image, debug=True)

