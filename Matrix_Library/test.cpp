#include <cmath>
#include <iostream>
#include <functional> // For std::function

template <typename T>
T* transpose(const T* input_matrix, size_t input_rows, size_t input_columns) {
    T* output_matrix = new T[input_columns * input_rows];
    for (size_t j = 0; j < input_columns; j++) {
        for (size_t i = 0; i < input_rows; i++) {
            //std::cout << input_matrix[j + (i * input_columns)] << "\n";
            output_matrix[i + (j * input_rows)] = input_matrix[j + (i * input_columns)];
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
    for (size_t i = 0; i < k; i++) {
        sum += vector_1[i] * vector_2[i];
    }
    return sum;
}

template <typename T>
T* splice(const T* vector, size_t start, size_t end)
{
    size_t return_size = (end > start) ? (end - start) : 0;
    T* spliced_array = new T[return_size];
    for (size_t i = 0; i < return_size; i++) {
        //We're assuming zero based indexing
        spliced_array[i] = vector[i + start];
    }
    return spliced_array;

}

template <typename T>
void print_array(const T* array, size_t n) {
    for (size_t i = 0; i < n;i++) {
        std::cout << array[i] << " ";
    }
    std::cout << "\n";
}

template <typename T>
T* mat_mult(const T* matrix_1, const T* matrix_2, size_t rows_1, size_t columns_1, size_t rows_2, size_t columns_2)
{
    T* out_matrix = new T[rows_1 * columns_2];
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
    //For splicing purposes, it would be easier to transpose matrix 2

    std::cout << "Number of rows in first matrix is " << rows_1 << "\n";
    std::cout << "Number of columns in first matrix is " << columns_1 << "\n";

    std::cout << "Number of rows in second matrix is " << rows_2 << "\n";
    std::cout << "Number of columns in second matrix is " << columns_2 << "\n";

    T* matrix_2_transpose = transpose(matrix_2, rows_2, columns_2);
    size_t k = columns_1;
    std::cout << "K is " << k << "\n";
    size_t index = 0;
    for (size_t i = 0; i < rows_1; i++) {
        size_t start_row = i*k;
        size_t end_row = start_row + k;
        T* row_vector = splice(matrix_1, start_row, end_row);
        std::cout << "The row vector for this iteration is " << "\n";
        print_array(row_vector, k);
        for (size_t j = 0; j < columns_2; j++) {
            size_t start_col = j * k;
            size_t end_col = start_col + k;
            T* col_vector = splice(matrix_2_transpose, start_col, end_col);
            std::cout << "The col vector for this iteratio is " << "\n";
            print_array(col_vector, k);
            std::cout << "Their dot product is: " << "\n";
            T dot = dot_product(row_vector, col_vector, k);
            std::cout << dot << "\n";
            std::cout << "Index is " << index << "\n";
            out_matrix[index] = dot;
            index += 1;
            delete[] col_vector;
            }
        delete[] row_vector;
        }
    return out_matrix;

}
template <typename T>
T* elem_wise_operator(std::function<T(T)>& foo, const T* array, size_t n) {
    T* ret_array = new T[n];
    for (size_t i = 0;i < n;i++) {
        ret_array[i] = foo(array[i]);
    }
    return ret_array;
}

template <typename T>
T* exp_array(const T* array, size_t n) {
    auto exp_func = [](T val) -> T {return std::exp(val);};
    return elem_wise_operator(exp_func, array, n);
}


int main(void) {
    int test_vector1[] = { 0,1,2,3,4,5 };
    int test_vector2[] = { 1,1,1,1,1,1 };
    int test_rows = 3;
    int test_columns = 2;
    int length = 5;

    //Tests
    //Dot Product
    int result = dot_product(test_vector1, test_vector2, length);
    std::cout << "The dot product of the two vectors is: " << result << "\n";

    //Transpose
    int* transpose_test1 = transpose(test_vector1, test_rows, test_columns);
    std::cout << "The transpose of the matrix is: " << "\n";
    print_array(transpose_test1,6);
    
    //Splice
    int* spliced_test1 = splice(test_vector1, 2, 4);
    std::cout << "The spliced array is: " << "\n";
    print_array(spliced_test1, 2);

    //Mat_Mult
    int* out_matrix = mat_mult(test_vector1, test_vector2, 3, 2, 2, 3);
    print_array(out_matrix,9);

    //Exp Array
    int* exp_vector = exp_array(test_vector1, 6);

    delete[] transpose_test1;
    delete[] spliced_test1;

    return 0;
}