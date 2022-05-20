// Run using g++ main.cpp -I/home/ubuntu/anaconda3/envs/torchrec/include/ -L/usr/lib/x86_64-linux-gnu -l:libcrypto.so.1.1

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <openssl/evp.h>

#include <math.h> /* log2 */

bool computeHash(const std::string& unhashed, std::string& hashed)
{
    bool success = false;

    EVP_MD_CTX* context = EVP_MD_CTX_new();

    if(context != NULL)
    {
        if(EVP_DigestInit_ex(context, EVP_sha256(), NULL))
        {
            if(EVP_DigestUpdate(context, unhashed.c_str(), unhashed.length()))
            {
                unsigned char hash[EVP_MAX_MD_SIZE];
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

#define VERBOSE_MODE true

Matrix hash_single_vals_to_vecs(int N, int batch_size, Matrix input_nums, int desired_vector_sizes) {

    int hash_piece_size = 1 + (int)log2((double)N)/4;
    if (VERBOSE_MODE)
        std::cout<<"hash_piece_size: "<<hash_piece_size<<"\n\n";
    Matrix output_vectors = Matrix(batch_size, desired_vector_sizes);
    int * output_data = output_vectors.data();

    int * intput_data = input_nums.data();

    for (int i = 0; i < batch_size; i++) {

        std::string key = std::to_string(intput_data[i]);
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



int main(int, char**)
{
    int batch_size = 4;
    int N = 1000;
    Matrix input_nums = Matrix(1,batch_size);
    int desired_vector_sizes = 10;

    int * input_data = input_nums.data();
    for (int i = 0; i < batch_size; i++)
        input_data[i] = i % 2;

    Matrix output_vecs = hash_single_vals_to_vecs(N, batch_size, input_nums, desired_vector_sizes);

    int * output_data = output_vecs.data();

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < desired_vector_sizes; j++) {
            std::cout<<output_data[i*desired_vector_sizes + j]<<",";
        }
        std::cout<<"\n";
    }

    /*
    std::string pw1 = "password1", pw1hashed;
    std::string pw2 = "password2", pw2hashed;
    std::string pw3 = "password3", pw3hashed;
    std::string pw4 = "password4", pw4hashed;

    computeHash(pw1, pw1hashed);
    computeHash(pw2, pw2hashed);
    computeHash(pw3, pw3hashed);
    computeHash(pw4, pw4hashed);

    std::cout << pw1hashed << std::endl;
    std::cout << pw2hashed << std::endl;
    std::cout << pw3hashed << std::endl;
    std::cout << pw4hashed << std::endl;
    */
    return 0;
}








#if 0



import hashlib
import numpy as np

def hash_single_vals_to_vecs(N, batch_size, input_nums, desired_vector_sizes=10):
    hash_piece_size = int(1 + np.log2(N)//4)
    output_vectors = []
    for input_num in input_nums:
        hashobj = hashlib.sha256(str(input_num).encode('utf-8'))
        val_hex = hashobj.hexdigest()
        output_vector = []
        for i in range(desired_vector_sizes):
            piece = val_hex[ i*hash_piece_size : (i+1)*hash_piece_size ]
            output_vector.append(int.from_bytes(piece.encode('utf-8'), 'little') % N)
        output_vectors.append(output_vector)
    return np.array(output_vectors)


batch_size = 2048
N = 10000000
input_nums = np.random.randint(0,N, size=(batch_size))
desired_vector_sizes = 10

output_vecs = hash_single_vals_to_vecs(N, batch_size, input_nums, desired_vector_sizes)
print(output_vecs.shape)







#include <iostream>
//#include "/home/ubuntu/anaconda3/pkgs/openssl-1.0.2p-h14c3975_0/include/openssl/sha.h"
#include <openssl/sha.h>
int main() {
    sha256 = OpenSSL::Digest::SHA256.new
    std::cout<<"hello world!";
    return 0;
}
#endif