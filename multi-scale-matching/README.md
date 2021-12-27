# Multi-Scale-Matching

-----------

## English

Normally you can't use matchTemplate when your template has a different size than your image but you can use a trick:

1. Loop over the input image at multiple scales (i.e. make the input image progressively smaller and smaller)
2. Apply template matching using `cv2.matchTemplate` and keep track of the match with the largest correlation coefficient along with the _(x, y)_ coordinates.
3. After looping over all scales take the region with the largest correlation coefficient and use that as your matched region.

The trick is pretty simple but can save a lot of extra code and dealing with fancy techniques to match objects in images.

**NOTE** If you already try/checked the `template-matching` folder you'll know that `cv2.matchTemplate` is translation invariant but no scaling invariant. That means that can detect the template without care about his position but the size is
pretty important. With this project we're trying to make it a little more invariant to scale, but have care, it won't be rotation invariant! If you want this you must change to another match technique like keypoint matching.

--------

## Español

Normalmente no podemos usar la función _matchTemplate_ cuando la plantilla tiene un tamaño distinto que la imagen a analizar, sin embargo puedes aplicar este truco:

1. Itera sobre la imagen de entrada con multiples escalas de la plantilla (i.e. hacer que la imagen de entrada se haga progresivamente mas pequeña).
2. Aplicar la plantilla con `cv2.matchTemplate` y dar seguimiento a las coincidencias que tengan los mayores coeficientes de correlación sobre las coordenadas _(x, y)_.
3. Luego de iterar sobre todas las posibles escalas, se toma la región con el mayor coeficiente de correlación y se utilizará como la región coincidente.

El truco es bastante simple y puede salvar mucho código extra y el lidiar con tecnicas mas especializadas para encontrar coincidencias de objetos en imagenes.

**NOTA** Si ya te has paseado por la carpeta `template-matching` sabrás que `cv2.matchTemplate` es invariante a la traslación, mas no invariante a la escala. Es decir, puede detectar la plantilla sin importar la ubicación de esta pero el tamaño debe coincidir. En este proyecto intentaremos solventar este ultimo punto, haciendolo un poco invariante a la escala. Sin embargo, hay que tener en cuenta que este método no es invariante a la rotación, por lo que si deseas afrontarte a
un problema que requiera esta característica, deberás utilizar algún otra técnica de coincidencia cómo _keypoint matching_.




