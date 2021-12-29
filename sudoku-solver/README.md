# Sudoku Solver

## Tabla de contenidos

1. [English](#English)
2. [Español](#Español)

-------
## English

The pipeline to solve a sudoku with OpenCV is the following:

1. Load the input image.
2. Localize sudoku puzzle in it.
3. Given the Sudoku location, localize each of the cells.
4. Determine if digit exists in cell, if so, OCR it.
5. Given cell locations and digits, solve the sudoku puzzle.
6. Display solved sudoku puzzle.

For this project we will need the following libraries:

- OpenCV
- Scikit-image
- Scikit-Learn
- TensorFlow
- Py-Sudoku

The project structure is the following:

- root:
  - solve_sudoku.py
  - sudoku.jpg
  - train_ocr.py
  - Configurations/
    - models
      - SudokuNet.py
    - Sudoku
      - puzzle.py

SudokuNet.py will contains the CNN architecture with TensorFlow.

Puzzle.py will have helper utilities for finding the Sudoku board and the digits.

train_ocr.py will have the code to train the SudokuNet classifier.


-------------

## Español

