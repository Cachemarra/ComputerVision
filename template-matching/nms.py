import numpy as np

def non_max_suppression(boxes: np.ndarray, overlapThresh=0.6) -> np.ndarray:
    # If not boxes were passed return an empty list
    if len(boxes) == 0:
        return []
    
    # Convert boxes to float
    # This is important as we'll do a lot of divisions
    # Se convierte la variable boxes a flotante
    # Esto se hace debido a que vamos a hacer muchas divisiones
    if boxes.dtype.kind == 'i':
        boxes = boxes.astype('float')

    # Initialize the list of picked indexes
    # Inicializamos la lista de índices seleccionados
    pick = []

    # Grab the coordinates of the bounding boxes
    # Obtenemos las coordenadas de las cajas delimitadoras
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute the area of the bounding boxes and sort the 
    # bounding boxes by the bottm-right y-coordinate of the bounding box
    # Calculamos el área de las cajas delimitadoras y las ordenamos 
    # por la coordenada inferior derecha y-coordenada de la caja
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Keep looping while some indexes still remain in the indexes list
    # Mientras que algunos índices aún queden en la lista de índices
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index 
        # value to the list of picked indexes
        # Obtenemos el último índice en la lista de índices y lo agregamos
        # a la lista de índices seleccionados
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding box
        # Encontramos las coordenadas más grandes (x, y) para el inicio de la caja
        # delimitadora y las coordenadas más pequeñas (x, y) para el final de la caja
        # Busca los puntos (x, y) mas lejanos ya que contendrán TODA la caja.
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.maximum(y2[i], y2[idxs[:last]])

        # Compute the width and height of the bounding box
        # Calculamos el ancho y el alto de la caja delimitadora
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        # Calculamos la proporción de superposición
        overlap = (w * h) / area[idxs[:last]]

        # Delete all indexes from the index list that have
        # overlap greater than the provided overlap threshold
        # Eliminamos todos los índices de la lista de índices que tienen
        # una superposición mayor que el umbral de superposición proporcionado
        idxs = np.delete(idxs, 
                        np.concatenate(([last], 
                                        np.where(overlap > overlapThresh)[0])
                                        )
                        )
    # Return only the bounding boxes that were picked using the integer data type
    return boxes[pick].astype('int')
