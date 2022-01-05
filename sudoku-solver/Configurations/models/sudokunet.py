#%% SudokuNet
# A net to read the numbers in the sudoku grid.

# Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation
from tensorflow.keras.layers import Dense, Dropout, Flatten

#%% SudokuNet Class

class SudokuNet:
    # Static methods
    @staticmethod
    def build(width: int=28, height: int=28, depth: int=1, classes: int=10) -> Sequential:
        '''
        :param width: The width of an MNIST digit.
        :param height: The height of an MNIST digit.
        :param depth: The number of channels in the MNIST digit (grayscale).
        :param classes: The number of classes in the MNIST digit (0-9, hence 10).
        '''
        # Initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # Model Building
        # First Block Conv2D -> Activation -> MaxPooling2D
        model.add(Conv2D(32, (5, 5), padding='same', input_shape=inputShape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second Block Conv2D -> Activation -> MaxPooling2D
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Third Block Flatten -> Dense -> Activation -> Dropout
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))

        # Fourth Block Dense -> Activation -> Dropout
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))

        # Fifth Block Dense -> Activation -> Classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        # Return the model
        return model



if '__main__' == __name__:
    # Test the SudokuNet class
    model = SudokuNet.build()
    print(model.summary())


