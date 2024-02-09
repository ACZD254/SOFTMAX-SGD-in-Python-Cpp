import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    print("I am just debugging bro")
    return x+y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    #Okay so let me think about this. We have the MNIST file format.
    #And it looks something like this.
    """
        [offset] [type]          [value]          [description]
        0000     32 bit integer  0x00000803(2051) magic number
        0004     32 bit integer  60000            number of images
        0008     32 bit integer  28               number of rows
        0012     32 bit integer  28               number of columns
        0016     unsigned byte   ??               pixel
    """
    #So the first 16 Bytes are the header.
    #And then we have the data.
    #Okay. What's with the number of rows and the number of columns?
    #Oh yeah, that's how many pixels are there in every image.
    #So each image is Row*Column long.
    #Okay so slowly but surely, a picture is forming.
    with gzip.open(image_filename) as image_bin:
        img_header = image_bin.read(16)
        img_magic_number, number_of_images, rows, cols = struct.unpack('>IIII',img_header)
        print("number of colums is ", cols)
        print("number of rows is ", rows)
        print("number of images is ", number_of_images)
        print("magic number is ", img_magic_number)
        #Okay so now I have an open file and I can do something about it. 
        #The header has been parsed. I have the data that I need.
        #Or atleast the metadata that I need. Now I Need to pack each image in a numpy array.
        #How do I do that?
        #I don't think that I'm supposed to use a for-loop. Right? How do you populate an array anyway?
        #So maybe something like preinitializing the array and then populating the fields?
        #What I have found is even more efficient.
        images = np.frombuffer(image_bin.read(number_of_images*rows*cols),dtype=np.uint8)
        images = images.reshape(number_of_images, rows*cols).astype(np.float32)/255


    with gzip.open(label_filename) as label_bin:
        lbl_header = label_bin.read(8)
        lbl_magic_number, number_of_items = struct.unpack('>II',lbl_header)
        print("magic_number for labels is ", lbl_magic_number)
        print("number of items is ", number_of_items)

        labels = np.frombuffer(label_bin.read(number_of_items),dtype=np.uint8)

    (X,y) = (images,labels)
    print(y[0:10])
    return (X,y)

    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    """
    Okay so what do I need to do again?
    I know the formula.
    its something like ->
    So we already have the logit predictions for each class.
    One of the things that is different from the last time I tried this is that
    y is no longer a vector of one-hot encodings but the true label of each class.
    This changes things.
    Okay, I understand what is happening here. 
    Lce = -log(softmax(Zi,yi))
    so we're using the fact that we have the true class given by y for each example i
    and then we're using yi as an indexing tool to access the softmax probablity
    And then the loss is computed by using the entropy formula (-log(probability))
    Okay, so how can we do this with a single line of code? 
    And apparently I don't need to use for-loops?
    """
    batch_size = Z.shape[0]
    #I really think that python's list comprehension might be the way to go
    #So lets say that we're computing individual losses L_i
    # Compute individual losses L_i for each example
    individual_losses = [-np.log(np.exp(Z[i, y[i]]) / np.sum(np.exp(Z[i]))) for i in range(batch_size)]
    # Compute the average loss over the batch
    average_loss = (1 / batch_size) * np.sum(individual_losses)
    return average_loss

    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    """
    So, what would the code look like? Something like this
    1. Create batches with batch_size
    2. for each batch in batches
        find the gradient of the loss function w.r.t theta
        compute new theta using l.r and gradient
        update theta 
        And the gradient is given by
        ∇Θℓsoftmax(XΘ,y)=(1/m)XT(Z−Iy)
        where
        Z=normalize(exp(XΘ))(normalization applied row-wise) 
        denotes the matrix of logits, and  Iy∈Rm×k  represents a concatenation of one-hot bases for the labels in  y.
    """
    #Z = np.array([[np.exp(X[i,j])/np.sum(np.exp(X[i])) for j in range(X.shape[1])] for i in range(X.shape[0])])
    #A better way to do this
    #Divide Data into batches
    for start_idx in range(0, X.shape[0], batch):
        # Create batches
        X_batch = X[start_idx:start_idx + batch]
        y_batch = y[start_idx:start_idx + batch]

        # Compute logits and apply softmax
        logits = np.dot(X_batch, theta) #this is the hypothesis h from the lectures.
        logits -= np.max(logits, axis=1, keepdims=True)  # For numerical stability though not strictly necessary
        Z = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

        # Create a one-hot encoding of the true labels
        Iy = np.zeros_like(Z)
        Iy[np.arange(X_batch.shape[0]), y_batch] = 1

        # Compute the gradient
        batch_gradient = (1 / X_batch.shape[0]) * np.dot(X_batch.T, (Z - Iy))

        # Update theta
        theta -= lr * batch_gradient
    
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    """
    Okay so this is not too hard, is it?
    I am not asked to write down the genralized update for a neural network.
    All I have to really do is update W1 and W2 on my own and that to with simply 
    two lines of python code.
    This should not take too long.
    
    This is what the equations look like:
    Z1∈Rm×d=ReLU(XW1)
    G2∈Rm×k=normalize(exp(Z1W2))−Iy
    G1∈Rm×d=1{Z1>0}∘(G2WT2) 

    where 1{Z1>0} is a binary matrix with entries equal to zero or one depending on 
    whether each term in Z1 is strictly positive and where ∘ denotes elementwise multiplication. 

    Then the gradients of the objective are given by
    ∇W1ℓsoftmax(ReLU(XW1)W2,y)=(1/m)X.T@G1
    ∇W2ℓsoftmax(ReLU(XW1)W2,y)=(1/m)Z.T1@G2.
    """

    #So lets write a ReLU function
    def ReLU(x):
        return np.maximum(0,x)
    
    #Okay, so now I have to think a bit about batches.
    #And apparently, it is okay to use for Loops so lets use them
    num_examples = X.shape[0]

    #Okay so lets try coding this out
    for idx in range(0, num_examples, batch):
        #What are we supposed to do?
    
    # for start_idx in range(0, num_examples, batch):
    #     end_idx = min(start_idx + batch, num_examples)
    #     X_batch = X[start_idx:end_idx]
    #     y_batch = y[start_idx:end_idx]

    #     #Forward pass
    #     Z1 = ReLU(np.dot(X_batch, W1))
    #     logits = np.dot(Z1, W2)

    #     #Softmax probabilities
    #     exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    #     Z2 = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    #     #One-hot encoding for y_batch
    #     Iy = np.zeros_like(Z2)
    #     Iy[np.arange(end_idx - start_idx), y_batch] = 1

    #     #Backward pass: compute gradients
    #     G2 = Z2 - Iy
    #     G1 = np.dot(G2, W2.T)
    #     G1[Z1 <= 0] = 0  # Gradient through ReLU

    #     #Gradients for weights
    #     grad_W1 = np.dot(X_batch.T, G1) / (end_idx - start_idx)
    #     grad_W2 = np.dot(Z1.T, G2) / (end_idx - start_idx)

    #     # Update weights
    #     W1 -= lr * grad_W1
    #     W2 -= lr * grad_W2

    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
