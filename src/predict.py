import numpy as np
import matplotlib.pyplot as plt

from datasets.mnist import load_mnist
#from models.softmax_regression import SoftmaxRegression
from models.mlp import MLP
from utils.io import load_model

#preprocess the image to be between 0 and 1
def preprocess_image(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    if img.max() > 1.0 :
        img /= 255.0
    return img.reshape(1, -1)

#predict the digit from the image
def main():
    params = load_model("mlp_mnist.npz")
    n_features = params["W1"].shape[1]
    n_hidden = params["W1"].shape[0]
    n_classes = params["W2"].shape[1]
    model = MLP(n_features=n_features, n_hidden=n_hidden, n_classes=n_classes)
    model.W1 = params["W1"]
    model.b1 = params["b1"]
    model.W2 = params["W2"]
    model.b2 = params["b2"]

    # W, b = load_model("softmax_mnist.npz")
    # model = SoftmaxRegression(n_classes=10, n_features=W.shape[1])
    # model.W = W
    # model.b = b

    #load the test data
    _, _, X_test, y_test = load_mnist()

    #randomly select an image from the test data
    rng = np.random.default_rng()
    idx = rng.integers(0, X_test.shape[0])

    #reshape the image to 28x28
    img = X_test[idx].reshape(28, 28)
    true_label = y_test[idx]

    #preprocess the image to be between 0 and 1
    X = preprocess_image(img)
    #forward pass through the model
    probs = model.forward(X)
    pred = np.argmax(probs, axis=1)[0]
    
    #print the true label, the predicted digit, and the probabilities
    print("True label:", true_label)
    print("Predicted digit:", pred)
    print("Probabilities:", probs)

    #plot the image and the predicted digit
    plt.imshow(img, cmap="gray")
    plt.title("Predicted digit: " + str(pred))
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()

