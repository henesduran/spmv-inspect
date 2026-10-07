#include <iostream>
#include <omp.h>

int main() {
    std::cout << "spmv-inspect baslatildi!" << std::endl;
    std::cout << "Maksimum Kullanilabilir Thread: " << omp_get_max_threads() << std::endl;
    return 0;
}