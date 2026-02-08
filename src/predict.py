#prediction script for cnn model
#loads a trained model and makes predictions on test images
#displays a random test image with its predicted label

import numpy as np
import matplotlib.pyplot as plt

from datasets.mnist import load_mnist
from models.cnn import CNN
from utils.io import load_model

#predict the digit from the image
def main():
    #load saved cnn parameters
    params = load_model("cnn_mnist.npz")

    #create cnn model and load parameters
    model = CNN()
    model.set_parameters(params)
    model.eval()

    #load test data (flatten=False for cnn format: (N, 1, 28, 28))
    _, _, X_test, y_test = load_mnist(flatten=False)

    #randomly select a test image
    rng = np.random.default_rng()
    idx = rng.integers(0, X_test.shape[0])

    #extract the image for display (28x28) and the true label
    img_for_display = X_test[idx, 0]  #shape (28, 28)
    true_label = y_test[idx]

    #forward pass through the model
    X = X_test[idx:idx+1]  #shape (1, 1, 28, 28)
    probs = model.forward(X)
    pred = np.argmax(probs, axis=1)[0]

    #print the true label, the predicted digit, and the probabilities
    print(f"True label: {true_label}")
    print(f"Predicted digit: {pred}")
    print(f"Confidence: {probs[0, pred]:.4f}")
    print(f"Probabilities: {probs}")

    #plot the image and the predicted digit
    plt.imshow(img_for_display, cmap="gray")
    plt.title(f"Predicted: {pred} (True: {true_label})")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
