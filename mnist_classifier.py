from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier  # For Feed-Forward NN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score

class MnistClassifierInterface(ABC):
    """
    Interface for MNIST classifier models.
    All models should implement train and predict methods.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the classifier model.

        Args:
            X_train: Training data features.
            y_train: Training data labels.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predicts class labels for input data.

        Args:
            X: Input data features.

        Returns:
            Predictions (e.g., array of class labels).
        """
        pass

class RandomForestMnistClassifier(MnistClassifierInterface):
    """
    MNIST classifier using Random Forest.
    """
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42) # You can adjust parameters

    def train(self, X_train, y_train):
        # Random Forest works best with 1D input features
        n_samples, img_height, img_width = X_train.shape
        X_train_reshaped = X_train.reshape((n_samples, img_height * img_width))
        self.model.fit(X_train_reshaped, y_train)

    def predict(self, X):
        n_samples, img_height, img_width = X.shape
        X_reshaped = X.reshape((n_samples, img_height * img_width))
        return self.model.predict(X_reshaped)


class NeuralNetworkMnistClassifier(MnistClassifierInterface):
    """
    MNIST classifier using a Feed-Forward Neural Network.
    """
    def __init__(self):
        self.model = Sequential([
            Flatten(input_shape=(28, 28)), # Flatten 28x28 images to 1D array
            Dense(128, activation='relu'),
            Dense(10, activation='softmax') # 10 classes for digits 0-9
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy', # for integer labels
                           metrics=['accuracy'])

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=10, verbose=0) # epochs can be adjusted

    def predict(self, X):
        probabilities = self.model.predict(X)
        return np.argmax(probabilities, axis=1) # Convert probabilities to class labels


class CNNMnistClassifier(MnistClassifierInterface):
    """
    MNIST classifier using a Convolutional Neural Network.
    """
    def __init__(self):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), # Input shape with channel dimension
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(10, activation='softmax') # 10 classes
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy', # for integer labels
                           metrics=['accuracy'])

    def train(self, X_train, y_train):
        # CNN expects input with channel dimension
        X_train_expanded = np.expand_dims(X_train, axis=-1) # Add channel dimension (grayscale)
        self.model.fit(X_train_expanded, y_train, epochs=10, verbose=0) # epochs can be adjusted

    def predict(self, X):
        X_expanded = np.expand_dims(X, axis=-1) # Add channel dimension for prediction
        probabilities = self.model.predict(X_expanded)
        return np.argmax(probabilities, axis=1) # Convert probabilities to class labels


class MnistClassifier:
    """
    Wrapper class for MNIST classifiers.
    Chooses the underlying classifier based on the 'algorithm' parameter.
    """
    def __init__(self, algorithm):
        self.algorithm = algorithm.lower() # To handle 'CNN' and 'cnn' consistently
        self._model = self._create_model() # Initialize the concrete model

    def _create_model(self):
        """
        Creates and returns the concrete classifier model based on the algorithm.
        """
        if self.algorithm == 'rf':
            return RandomForestMnistClassifier()
        elif self.algorithm == 'nn':
            return NeuralNetworkMnistClassifier()
        elif self.algorithm == 'cnn':
            return CNNMnistClassifier()
        else:
            raise ValueError(f"Invalid algorithm: {self.algorithm}. "
                             "Choose from 'rf', 'nn', or 'cnn'.")

    def train(self, X_train, y_train):
        """
        Trains the selected classifier model.
        Delegates the training to the underlying concrete model.
        """
        self._model.train(X_train, y_train)

    def predict(self, X):
        """
        Predicts with the selected classifier model.
        Delegates the prediction to the underlying concrete model.
        """
        return self._model.predict(X)


if __name__ == "__main__":
    # 1. Load and Prepare MNIST Data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize pixel values to be between 0 and 1
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0


    # 2. Choose and Instantiate a Classifier via MnistClassifier
    algorithms = ['rf', 'nn', 'cnn']
    for algorithm in algorithms:
        print(f"\n--- Training and Evaluating {algorithm.upper()} Model ---")
        mnist_classifier = MnistClassifier(algorithm=algorithm)

        # 3. Train the Model
        mnist_classifier.train(X_train, y_train)

        # 4. Make Predictions
        predictions = mnist_classifier.predict(X_test)

        # 5. Evaluate Accuracy
        accuracy = accuracy_score(y_test, predictions)
        print(f"{algorithm.upper()} Model Accuracy: {accuracy:.4f}")