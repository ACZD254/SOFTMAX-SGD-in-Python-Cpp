#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


#include <functional> // For std::function

template <typename T>
T* transpose(const T* input_matrix, size_t input_rows, size_t input_columns){
    T* output_matrix = new T[input_columns*input_rows];
    for (size_t j = 0; j< input_columns; j++){
        for(size_t i = 0; i< input_rows; i++){
            output_matrix[j+(i*input_rows)] = input_matrix[i+(j*input_columns)];
        }
    }
    return output_matrix;
}

template <typename T>
T dot_product(const T* vector_1, const T* vector_2, size_t k)
{   
    /*
    Takes in two 1D vectors of length k and multiplies them
    */   
    T sum = 0;
    for (size_t i = 0; i<k; i++){
        sum += vector_1[i] * vector_2[i];
    }
    return sum;
}

template <typename T>
T* splice(const T* vector, size_t start, size_t end)
{
    size_t return_size = (end>start)? (end-start):0;
    T* spliced_array = new T[return_size];
    for (size_t i = 0; i< return_size; i++){
        //We're assuming zero based indexing
        spliced_array[i] = vector[i+start];
    }
    return spliced_array;
    
}


template <typename t>
T* mat_mult(const T* matrix_1, const T* matrix_2, size_t rows_1, size_t columns_1, size_t rows_2, size_t columns_2)
{
    T* out_matrix = new T[rows_1*columns_2];
    //Matrix 1 = m*k
    //Matrix 2 = k*n
    //The output is a vector m*n long
    // Check if columns in the first matrix are equal to rows in the second matrix
    if (columns_1 != rows_2) {
        std::cerr << "Error: The number of columns in the first matrix must equal the number of rows in the second matrix." << std::endl;
        return nullptr; // Return a null pointer to indicate failure
    }
    /* Main Loop
        for i in number of rows of first matrix
            Take row[i] of first matrix
            for j in number of columns of second matrix
                Take column[j] matrix
                output[i,j] = dot_product(row[i], column[j], k)

    */
    for (size_t i = 1; i < rows_1+1; i++){
        T* intermediate_vector = new T[rows_1];


        delete[] intermediate_vector;
    }
}
int main(void){
    int test_vector_1[] = {0,1,2,3,4,5};
    int test_vector_2[] = {1,1,1,1,1,1};
    int result = dot_product(vector1, vector2, length);

    // Print the result
    std::cout << "The dot product of the two vectors is: " << result << std::endl;

    return 0;
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
