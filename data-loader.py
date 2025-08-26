# data_loader.py
import numpy as np
import urllib.request
import gzip
import os

def load_fashion_mnist():
    """
    Loads the Fashion-MNIST dataset from the official source using pure NumPy.
    Returns:
        (x_train, y_train), (x_test, y_test)
    """
    # URLs for the Fashion-MNIST dataset files
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    # Create a data directory if it doesn't exist
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Download and save files
    for file in files:
        url = base_url + file
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(url, file_path)
        else:
            print(f"{file} already exists.")
    
    # Helper function to parse image files
    def parse_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
        return data.reshape(-1, 28, 28).astype(np.float32)
    
    # Helper function to parse label files
    def parse_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return data.astype(np.int32)
    
    # Load and parse all files
    print("Loading and parsing data files...")
    x_train = parse_images(os.path.join(data_dir, files[0])) / 255.0  # Normalize to [0,1]
    y_train = parse_labels(os.path.join(data_dir, files[1]))
    x_test = parse_images(os.path.join(data_dir, files[2])) / 255.0   # Normalize to [0,1]
    y_test = parse_labels(os.path.join(data_dir, files[3]))
    
    # Flatten the images from 28x28 to 784-dimensional vectors
    x_train_flat = x_train.reshape((-1, 784))
    x_test_flat = x_test.reshape((-1, 784))
    
    # Convert labels to one-hot encoding using only NumPy
    def to_onehot(labels, num_classes=10):
        onehot = np.zeros((len(labels), num_classes))
        onehot[np.arange(len(labels)), labels] = 1
        return onehot
        
    y_train_onehot = to_onehot(y_train)
    y_test_onehot = to_onehot(y_test)
    
    print(f"Training data shape: {x_train_flat.shape}")   # Should be (60000, 784)
    print(f"Training labels shape: {y_train_onehot.shape}") # Should be (60000, 10)
    print(f"Test data shape: {x_test_flat.shape}")        # Should be (10000, 784)
    print(f"Test labels shape: {y_test_onehot.shape}")    # Should be (10000, 10)
    
    return (x_train_flat, y_train_onehot), (x_test_flat, y_test_onehot)

# Load the data
(train_images, train_labels), (test_images, test_labels) = load_fashion_mnist()

# Class names for Fashion-MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Let's visualize a few samples to confirm it worked
import matplotlib.pyplot as plt # We'll use matplotlib for plotting, which is allowed
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    # Reshape the flattened vector back to an image for display
    plt.imshow(train_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(train_labels[i])]) # Get class name from one-hot label
plt.show()