#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace spmv{
    
struct COOElement {
    int row;
    int col;
    double val;

    COOElement(int r,int c,double v):row(r),col(c),val(v){}
};

struct COOMatrix {
    int num_rows = 0;
    int num_cols = 0;
    size_t nnz = 0;
    std::vector<COOElement> elements;

    COOMatrix(int num_rows,int num_cols,size_t nnz,std::vector<COOElement> elmnts): num_rows(num_rows),num_cols(num_cols),nnz(nnz),elements(std::move(elmnts)){}
};

inline COOMatrix read_matrix(const std::string& filepath){
    std::ifstream file(filepath);
    if(!file.is_open()){
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    std::string line;
    std::getline(file, line);
    for (char &ch : line) ch = static_cast<char>(std::tolower(ch));
    bool is_symmetric = line.find("symmetric") != std::string::npos;
    bool is_pattern = line.find("pattern") != std::string::npos;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (line[0] != '%') break;
    }
    std::stringstream ss(line);
    
    int num_rows = 0, num_cols = 0;
    size_t nnz = 0;
    ss >> num_rows >> num_cols >> nnz;

    std::vector<COOElement> elements;
    elements.reserve(is_symmetric ? nnz * 2 : nnz);

    int r,c;
    double val = 1.0;

    if (is_pattern) {
        while (file >> r >> c) {
            elements.push_back(COOElement(r - 1, c - 1, 1.0));
            if (is_symmetric && r != c) {
                elements.push_back(COOElement(c - 1, r - 1, 1.0));
            }
        }
    } else {
        while (file >> r >> c >> val) {
            elements.push_back(COOElement(r - 1, c - 1, val));
            if (is_symmetric && r != c) {
                elements.push_back(COOElement(c - 1, r - 1, val));
            }
        }
    }

    return COOMatrix(num_rows,num_cols,elements.size(),std::move(elements));

}
}