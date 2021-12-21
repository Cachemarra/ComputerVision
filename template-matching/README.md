# Multi template matching with OpenCV

-----------------------------

## English

In order to work we're going to do:

- Apply `cv2.matchTemplate`
- Find all (x, y) coordinates where the template matching result matrix is greater than a preset threshold score
- Extract all of these regions
- Apply non-maxima suppression to them

In short, we will filter the result matrix obtained from `cv2.matchTemplate` function and then apply non-maxima suppression

-----------------

## Español

Para lograr hacer una detección de una plantilla haremos lo siguiente:

- Utilizar la función `cv2.matchTemplate`
- Encontrar todas las coordenadas (x, y) donde la matriz de similitud será mayor que el umbral predeterminado
- Extraer todas las regiones coincidentes
- Se aplica una supresión no maxima

En pocas palabras, se filtrará la matríz resultante de la función `cv2.matchTemplate` y luego aplicarémos una supresión non-maxima.