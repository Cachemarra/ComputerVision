#%% Libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import logging as log

log.basicConfig(level=log.INFO,
                format='[%(filename)s:%(lineno)s] %(message)s',
                datefmt="%I:%M:%S %p")

#%% Construction of the Argparse
# Construcción del Argparse

ap = argparse.ArgumentParser()
# Create a group for the arguments
# Creamos el grupo de argumentos
'''
    -i, --image:        Path to the image. REQUIRED 
                        Ruta de la imagen. REQUERIDO
    -t, --template:     Path to the template. REQUIRED
                        Ruta de la plantilla. REQUERIDO
    -b, --threshold:    Threshold for the template matching. Default 0.6
                        Umbral para la coincidencia de la plantilla. Por defecto 0.6
    -v, --visualize:    Flag to show the results. Default False
                        Bandera para mostrar los resultados. Por defecto False
'''
ap.add_argument("-i", "--image", required=True, type=str, help="Path to the image to find the template")
ap.add_argument("-t", "--template", required=True, type=str, help="Path to the template image")
ap.add_argument("-b", "--threshold", required=False, type=float, default=0.8, help="Threshold for the template matching")
ap.add_argument("-v", "--visualize", required=False, type=bool, default=False, help="Visualize the results")
args = vars(ap.parse_args())

#%% Load the images
# Cargamos las imágenes
log.debug("[INFO] Loading the images...")

image = cv2.imread(args["image"])
template = cv2.imread(args["template"])

# get template shapes
# obtenemos la dimensión de la plantilla
(tH, tW) = template.shape[:2]

# Show the images
# Mostramos las imágenes
if args["visualize"]:
    cv2.imshow("Image", image)
    cv2.imshow("Template", template)
    cv2.waitKey(0)