#%% Importing packages
import numpy as np
import cv2
import argparse
import logging as log
import glob
import cv2
import imutils

# Logging config
log.basicConfig(level=log.INFO,
                format='[%(filename)s:%(lineno)s] %(message)s',
                datefmt="%I:%M:%S %p")

log.debug('[DEBUG] Testing logs.')

#%% Construction of the Argparse

ap = argparse.ArgumentParser()

i_help = "Path to the image (jpg format). REQUIRED"
t_help = "Path to the template. REQUIRED"
v_help = "Visualization of images. Default False"

ap.add_argument("-i", "--image", required=True, type=str, help=i_help)
ap.add_argument("-t", "--template", required=True, type=str, help=t_help)
ap.add_argument("-v", "--visualize", required=False, type=int, default=0, help=v_help)

args = vars(ap.parse_args())

template_path = args["template"]
visualize = bool(args["visualize"])

log.debug(f"[DEBUG] Template path: {template_path}")
log.debug(f"[DEBUG] Visualize: {visualize}")

#%% Loading the template and image preprocessing
# Loading
template = cv2.imread(template_path)

# Preprocessing
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)

# Get the template shape
(tH, tW) = template.shape[:2]

# Show
if visualize:
    cv2.imshow("Template", template)

#%% Multi-Scale Matching Trick.
images_folder = args['image']
log.debug(f"[DEBUG] Images folder: {images_folder}")
for image_match in glob.glob(args['image'] + '/*.jpg'):
    log.debug(f"[DEBUG] Image path: {image_match}")

    # Loading the image and preprocessing.
    image = cv2.imread(image_match)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Creating a placeholder
    find = None

    # Iteration over different scales of the input image
    # We're going to resize from 100% to 20% in 20% steps.
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # Resize the image to the given scale.
        resized = imutils.resize(image, width=int(image.shape[1] * scale))
        # We get the resizing ratio.
        r = resized.shape[1] / float(image.shape[1])

        # If the resized image is smaller than the template, then break
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # Now detect edges in the input image and apply templateMatching.
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)

        # Get the minValue, maxValue, minLoc and maxLoc.
        # We're interested in maxValue and his locations.
        (_, maxValue, _, maxLoc) = cv2.minMaxLoc(result)

        # Check if visualize is True to show how the program check for
        # matching
        if visualize:
            clone = np.dstack([edged, edged, edged])
                         # img    (x,y) loc  (x+tW, y+tH), (colour), (thickness))     
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]), (0, 0, 255), 2)
            cv2.imshow("Visualize", clone)

        # If we have a new maximun value (maxValue) then update the placeholder
        if find is None or maxValue > find[0]:
            find = (maxValue, maxLoc, r)

    try: # If no matching was found, then we throw an exception    
        # Now unpack the placeholder and compute the (x, y) coordinates of the bounding box.
        log.debug(f"[DEBUG] Find: {find}")
        (_, maxLoc, r) = find
        
        # Restore the size with the ratio
        (startX, startY) = int(maxLoc[0] * r), int(maxLoc[1] * r)
        (endX, endY) = int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r)

        # Show the result
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 255), 2)
        cv2.imshow("Final result", image)
        cv2.waitKey(0)

    except TypeError:
        log.info(f'[INFO] No matching found for the image {image_match}')

log.info('[INFO] Done.')

