# Task 1: MNIST Image Classification with OOP

## Solution Explanation

This project implements an MNIST digit classifier using three different machine learning models: Random Forest, Feed-Forward Neural Network, and Convolutional Neural Network. The solution is designed following Object-Oriented Programming (OOP) principles to ensure modularity, flexibility, and maintainability.

**Key Components:**

*   **`MnistClassifierInterface` (Interface):** Defines a common interface for all MNIST classifier models. It specifies the `train` and `predict` methods that each concrete model class must implement. This ensures that all models have a consistent way of being trained and used for prediction.

*   **Concrete Model Classes:**
    *   `RandomForestMnistClassifier`: Implements the `MnistClassifierInterface` using a Random Forest model from scikit-learn.
    *   `NeuralNetworkMnistClassifier`: Implements the interface with a Feed-Forward Neural Network built using TensorFlow/Keras.
    *   `CNNMnistClassifier`: Implements the interface with a Convolutional Neural Network built with TensorFlow/Keras, designed to leverage spatial features in images.

*   **`MnistClassifier` (Wrapper Class/Factory):** This class acts as a client-facing entry point. It takes an `algorithm` parameter (`'rf'`, `'nn'`, or `'cnn'`) and internally instantiates and uses the corresponding concrete model. This provides a unified interface for users, hiding the complexity of choosing and managing different model types.

**How it Addresses the Task Requirements:**

*   **Image Classification:** The project solves the MNIST image classification problem by training models to recognize handwritten digits.
*   **OOP:** The code is structured using interfaces and classes, demonstrating key OOP principles like abstraction, encapsulation, and polymorphism.
*   **Three Models:**  Three distinct models (Random Forest, Feed-Forward NN, CNN) are implemented.
*   **`MnistClassifierInterface`:** An interface is defined and implemented by the models.
*   **`MnistClassifier` Wrapper:** A wrapper class encapsulates the model selection logic.
*   **Consistent Input/Output:** The `MnistClassifier` ensures that regardless of the chosen algorithm, the input and output structure for training and prediction remains consistent.

## Setup Instructions

To run this project, you need to have Python 3 installed, along with the required libraries. Follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd MNIST_test
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    ```
    ```bash
    python3 -m venv venv
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Jupyter Notebook (Demo):**
    ```bash
    jupyter notebook demo.ipynb
    ```
    Or run the main Python script directly (if you have a separate script for execution, otherwise, the notebook is the main demo):
    ```bash
    python mnist_classifier.py
    ```

## Requirements

*   Python 3.11
*   Libraries listed in `requirements.txt` (install using `pip install -r requirements.txt`)

## Running the Demo

The `demo.ipynb` Jupyter Notebook provides a demonstration of how to use the `MnistClassifier` class to train and evaluate the different MNIST classification models. Open the notebook and follow the instructions within to see examples of:

*   Loading the MNIST dataset.
*   Instantiating `MnistClassifier` with different algorithms (`'rf'`, `'nn'`, `'cnn'`).
*   Training the models.
*   Making predictions on the test dataset.
*   Evaluating model accuracy.

## Edge Cases and Considerations

*(Optional: Briefly mention things like data normalization being important, the models are trained with default parameters and hyperparameter tuning could improve performance, potential for overfitting in more complex models if not regularized etc. Keep it concise, focus is on demonstrating the core functionality)*

For example:

*   **Data Normalization:** The MNIST pixel values are normalized to the range [0, 1]. This is crucial for the neural network models to train effectively.
*   **Model Hyperparameters:** The models are initialized with default parameters. For optimal performance in a real-world scenario, hyperparameter tuning would be necessary.
*   **Computational Resources:** Training Convolutional Neural Networks, especially deeper ones, can be computationally intensive and may require a GPU for faster training.

---
**Author:** [Your Name or GitHub Username]
**Date:** [Current Date]