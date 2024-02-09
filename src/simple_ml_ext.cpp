#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


#include <functional> // For std::function

// Generic element-wise operator function
template <typename T>
void elementWiseOperator(const std::function<T(T)>& f, const T* array, size_t size, T* result) {
    // Pseudocode:
    // def element_wise_operator(f, array)
    // for each element of list:
    //     element[i] of new_array = f(array[i])
    // return new_array
    for (size_t i = 0; i < size; ++i) {
        result[i] = f(array[i]);
    }
}

// Special case: Exponentiation of array elements
template <typename T>
void expArray(const T* array, size_t size, T* result) {
    // Pseudocode:
    // def exp_list(array):
    // return element_wise_operator(exp, array)
    elementWiseOperator<T>(static_cast<T(*)(T)>(std::exp), array, size, result);
}

// Special case: Multiplication of array elements by a scalar
template <typename T>
T* multByNum(T scalar, const T* array, size_t size) {
    // Pseudocode:
    // def mult_by_num(some_num):
    // return lambda(x) (some_num * x)
    T* result = new T[size];
    elementWiseOperator<T>([scalar](T x) { return scalar * x; }, array, size, result);
    return result;
}



void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    //So essentially, I want to write numpy-styled code
    //Yeah, this is more of a coding assignment/ C refresher than the other ones.
    //Okay, so lets see what I can do

    /*
    So how am I going to do this?
    poorly, to start with, I suppose.
    The issue or perhaps the only issue is that C code is going to be more verbose than numpy
    And maybe a bit more spelt out.
    WHich is not the worst thing, especially if I know exactly what I need to do.
    Okay, lets try to write the C code so that it actually makes sense to me in a wishful thinking way
    WE're supposed to perform softmax, that's it. 
    So what is exactly happening during that?
    For any given batch, we compute the softmax
    Then we compute the gradient of the loss
    And then we update the theta parameters.
    WE do this for all samples of the data in a batch-wise form.
    */

   /*
   Pseudocode:

   Forward pass
   Z = dot(X_batch, theta) //I have to be careful about passing references or pointers here
   logit_exp = exp(Z)
   softmax_loss = each row of logit_exp/ sum of each row of logit_exp

   Computing Gradients
   Iy = one-hot-encoding(y_batch,k)
   Gradient = (1/batch_size) * dot(transpose(X_batch),softmax_loss-Iy)

   Updating Theta
   theta =- lr*Gradient
   */

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
