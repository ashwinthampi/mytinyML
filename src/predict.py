#prediction script for mlp model
#loads a trained model and makes predictions on test images
#displays a random test image with its predicted label

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
    
    #infer layer sizes from parameter keys
    #handle both old format (W1, b1, W2, b2) and new format (W0, b0, W1, b1, ...)
    weight_keys = sorted([k for k in params.keys() if k.startswith("W")], 
                         key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    
    if not weight_keys:
        raise ValueError("No weight parameters found in saved model")
    
    #build layer_sizes from weights
    layer_sizes = []
    for i, key in enumerate(weight_keys):
        W = params[key]
        if i == 0:
            #first layer: input size is first dimension
            layer_sizes.append(W.shape[0])
        #output size is second dimension
        layer_sizes.append(W.shape[1])
    
    #create model with inferred architecture
    model = MLP(layer_sizes=layer_sizes)
    model.set_parameters(params)

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

