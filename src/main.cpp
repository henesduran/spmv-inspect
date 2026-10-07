#include <iostream>
#include <omp.h>
#include "mmio.hpp"

int main() {
    COOMatrix mat = read_matrix("samples/tiny.mtx");
    std::cout << "Matrix read with success" << std::endl;
    std::cout << "Row: " << mat.num_rows 
              << ", col: " << mat.num_cols 
              << ", NNZ: " << mat.nnz << std::endl;

    for (const auto& el : mat.elements) {
        std::cout << "(" << el.row << ", " << el.col << ") -> " << el.val << "\n";
    }
    return 0;
}