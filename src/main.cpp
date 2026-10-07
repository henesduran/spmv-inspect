#include <iostream>
#include <omp.h>
#include "mmio.hpp"
#include "csr_matrix.hpp"

int main() {
    auto coo = read_matrix("samples/tiny.mtx");
    auto csr = coo_to_csr(coo);

    std::cout << "--- CSR TEST ---" << std::endl;
    std::cout << "DIMS: " << csr.num_rows << "x" << csr.num_cols << ", NNZ: " << csr.nnz << "\n\n";

    std::cout << "row_ptr : ";
    for (int p : csr.row_ptr) std::cout << p << " ";
    std::cout << "\n";

    std::cout << "col_idx: ";
    for (int c : csr.col_idx) std::cout << c << " ";
    std::cout << "\n";

    std::cout << "values : ";
    for (double v : csr.values) std::cout << v << " ";
    std::cout << "\n";

    return 0;
}