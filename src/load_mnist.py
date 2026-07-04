import numpy as np
import tensorflow as tf

def load_mnist_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # download mnist data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # normalize pixels to [0, 1] because by default it's images so 0 to 255
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32) / 255.0

    # flatten images
    x_train = x_train.reshape(-1, 784)
    x_test  = x_test.reshape(-1, 784)

    # convert labels to binary
    y_train = (y_train % 2 == 0).astype(np.float32)
    y_test  = (y_test % 2 == 0).astype(np.float32)

    # add bias term 
    bias_train = np.ones((x_train.shape[0], 1), dtype=np.float32)
    bias_test  = np.ones((x_test.shape[0], 1), dtype=np.float32)

    x_train = np.hstack([x_train, bias_train]) 
    x_test  = np.hstack([x_test, bias_test])   

    return x_train, y_train, x_test, y_test

def shard_data(X: np.ndarray, y: np.ndarray, num_workers: int) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X_shards = np.array_split(X[indices], num_workers)
    y_shards = np.array_split(y[indices], num_workers)

    # return list of (x_shard, y_shard) tuples, one per worker 
    return list(zip(X_shards, y_shards))


    
