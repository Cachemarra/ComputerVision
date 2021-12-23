#%% Libraries
import numpy as np
import cv2
import argparse
import logging as log
import nms

log.basicConfig(level=log.DEBUG,
                format='[%(filename)s:%(lineno)s] %(message)s',
                datefmt="%I:%M:%S %p")

log.debug(f'[DEBUG] OpenCV Version: {cv2.__version__}')
# print(f'[DEBUG] OpenCV Version: {cv2.__version__}')


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
ap.add_argument("-b", "--threshold", required=False, type=float, default=0.6, help="Threshold for the template matching")
ap.add_argument("-v", "--visualize", required=False, type=int, default=1, help="Visualize the results")
args = vars(ap.parse_args())

image_path = args["image"]
template_path = args["template"]
threshold = args["threshold"]

#%% Load the images
# Cargamos los datos
log.debug("[INFO] Loading the images...")

image = cv2.imread(image_path)
template = cv2.imread(template_path)
template = cv2.resize(template, (50, 50))

log.info(f"[INFO] image shape: {image.shape}")
log.info(f"[INFO] template shape: {template.shape}")
log.info(f'[INFO] Visualize: {bool(args["visualize"])}')
# get template shapes
# obtenemos la dimensión de la plantilla
(tH, tW) = template.shape[:2]

# Show the images
# Mostramos las imágenes
if args["visualize"]:
    cv2.imshow("Image", image.copy())
    cv2.imshow("Template", template.copy())
#    cv2.waitKey(0)

#%% Image Normalization
# Normalizamos la imagen
log.debug("[INFO] Performing template matching...")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#%% Perform template matching
# Realizamos la coincidencia de la plantilla
log.info("[INFO] Performing template matching.")

result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
log.debug(f"[DEBUG] result shape: {result.shape}")

#%% Multiple object detection
# Detectamos varios objetos
(y_coord, x_coord) = np.where(result >= threshold)
clone = image.copy()

# Se detectaron *len(y_coord)* objetos antes de aplicar Non-Maxima Suppression
log.info(f"[INFO] {len(y_coord)} objects detected.")

# Draw a bounding box around the detected object
# Dibujamos un cuadro alrededor del objeto detectado
for (x, y) in zip(x_coord, y_coord):
    cv2.rectangle(clone, (x, y), (x + tW, y + tH), (0, 0, 255), 2)

if args["visualize"]:
    cv2.imshow("Visualization without NMS", clone)
#    cv2.waitKey(0)

#%% Applying Non-Maxima Suppression
# Aplicamos Non-Maxima Suppression

# Initializing our list of bounding boxes and creating a new clone
# Inicializamos nuestra lista de cuadros y creamos una copia
rects = []
del clone; clone = image.copy()

# Looping over the coordinates
# Recorremos las coordenadas
for (x, y) in zip(x_coord, y_coord):
    # Appending the bounding box to the list
    # Añadimos el cuadro al listado
    rects.append((x, y, tW, tH))

# Applying NMS
pick = nms.non_max_suppression(np.array(rects), 0.6)
log.info(f"[INFO] {len(pick)} objects detected after applying NMS.")

# Loop over the final bounding boxes
# Recorremos los cuadros finales
for (xA, yA, xB, yB) in pick:
    # Drawing the bounding box
    # Dibujamos el cuadro
    cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

# Show the output
# Mostramos el resultado
# if args["visualize"]:
cv2.imshow("Visualization with NMS", clone)
cv2.waitKey(0)

