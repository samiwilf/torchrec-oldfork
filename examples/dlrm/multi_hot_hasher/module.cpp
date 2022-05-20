#include <iostream>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "openssl/evp.h"
#include <math.h>
#include "omp.h"

namespace py = pybind11;

//unsigned char hash[EVP_MAX_MD_SIZE];
const int HASH_MAX_SIZE = 62;
//static char hash[HASH_MAX_SIZE] = {'0'};
//static unsigned int lengthOfHash = 0;
bool computeHash(const std::string& unhashed, std::string& hashed)
{
    bool success = false;

    EVP_MD_CTX* context = EVP_MD_CTX_new();

    if(context != NULL)
    {
        if(EVP_DigestInit_ex(context, EVP_sha224(), NULL))
        {
            if(EVP_DigestUpdate(context, unhashed.c_str(), unhashed.length()))
            {
                unsigned char hash[HASH_MAX_SIZE];
                unsigned int lengthOfHash = 0;

                if(EVP_DigestFinal_ex(context, hash, &lengthOfHash))
                {
                    std::stringstream ss;
                    for(unsigned int i = 0; i < lengthOfHash; ++i)
                    {
                        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
                    }

                    hashed = ss.str();
                    success = true;
                }
            }
        }
        EVP_MD_CTX_free(context);
    }
    return success;
}

class Matrix {
public:
    Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
        m_data = new int[rows*cols];
    }
    int *data() { return m_data; }
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
private:
    size_t m_rows, m_cols;
    int *m_data;
};


// struct hash_function_args {
//     int N;
//     int batch_size;
//     int input_nums;
//     int desired_vector_sizes;
// };

#define VERBOSE_MODE false

Matrix hash_single_vals_to_vecs(int emb_table_index, int N, int batch_size, py::array input_nums, int desired_vector_sizes) {

	py::buffer_info info = input_nums.request();
	int * input_data = static_cast<int *>(info.ptr);

    int hash_piece_size = 1 + (int)log2((double)N)/4;
    if (VERBOSE_MODE)
        std::cout<<"hash_piece_size: "<<hash_piece_size<<"\n\n";
    Matrix output_vectors = Matrix(batch_size, desired_vector_sizes);
    int * output_data = output_vectors.data();

	#pragma omp parallel for
	for (int i = 0; i < batch_size; i++) {

        std::string key = std::to_string(input_data[i]) + "_" + std::to_string(emb_table_index);
        std::string computed_hash;
        bool was_successful = computeHash(key, computed_hash);

        //int max_allowed_vector_size = computed_hash.length() / hash_piece_size;

        if (VERBOSE_MODE) {
            std::cout<<"Hash "<<i<<"\n";
            std::cout<<computed_hash<<"\n";
        }

        int jj = 0;
        for (int j = 0; j < desired_vector_sizes*hash_piece_size; j += hash_piece_size) {
            std::string piece = computed_hash.substr(j, hash_piece_size);

            if (VERBOSE_MODE)
                std::cout<<piece;

            unsigned int random_num;
            std::istringstream iss(piece);
            iss >> std::hex >> random_num;
            output_data[ i * desired_vector_sizes + jj++] = random_num % N;

            if (VERBOSE_MODE) {
                std::cout<<piece;
                std::cout<<" ("<<random_num<<")   ";
            }
        }
        if (VERBOSE_MODE)
            std::cout<<std::endl<<std::endl;
    }
    return output_vectors;
}

PYBIND11_MODULE(multi_hot_hasher, m) {
    m.def("meta_1_hot_hasher", &hash_single_vals_to_vecs);
    pybind11::class_<Matrix>(m, "Matrix", py::buffer_protocol())
   .def_buffer([](Matrix &m) -> py::buffer_info {
        return py::buffer_info(
            m.data(),                               /* Pointer to buffer */
            sizeof(int),                          /* Size of one scalar */
            py::format_descriptor<int>::format(), /* Python struct-style format descriptor */
            2,                                      /* Number of dimensions */
            { m.rows(), m.cols() },                 /* Buffer dimensions */
            { sizeof(int) * m.cols(),             /* Strides (in bytes) for each index */
              sizeof(int) }
        );
    });
}


/*

import multi_hot_hasher
import numpy as np
emb_table_index = 0
batch_size = 2048
N = 10000000
input_nums = np.random.randint(0,N, size=(batch_size))
desired_vector_sizes = 10


results = multi_hot_hasher.hash_single_vals_to_vecs(emb_table_index, N, batch_size, input_nums, desired_vector_sizes)
results = np.array(results)




//https://alexsm.com/pybind11-buffer-protocol-opencv-to-numpy/
python-side usage:
im = get_image
np.array(im, copy=False)
*/


/*
//interesting alternatives
https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
https://gist.github.com/abidrahmank/61fcc5230454403c77bd4b407b79f2a5 <-- straight to numpy
#https://docs.microsoft.com/en-us/visualstudio/python/working-with-c-cpp-python-in-visual-studio?view=vs-2019
#from ^, run with pip install .
#C:\_bsln\B\B>pip install .
*/