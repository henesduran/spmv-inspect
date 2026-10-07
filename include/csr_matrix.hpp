#pragma once

#include <vector>
#include <algorithm>
#include "mmio.hpp"

struct CSRMatrix {
    int num_rows = 0;
    int num_cols = 0;
    size_t nnz = 0;
    
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;

    CSRMatrix(int num_r,int num_c,size_t nnz, std::vector<int> row_Ptr,std::vector<int> col_Idx,std::vector<double> vals): num_rows(num_r),num_cols(num_c),nnz(nnz),row_ptr(row_Ptr),col_idx(col_Idx),values(vals){}
};

inline bool compare(COOElement &a,COOElement &b){
    if(a.row != b.row){
        return a.row < b.row;
    }
    return a.col < b.col;
}

inline CSRMatrix coo_to_csr(COOMatrix &coo){

    std::sort(coo.elements.begin(),coo.elements.end(),compare);

    std::vector<int> row_ptr(coo.num_rows+1,0);
    std::vector<int> col_idx;
    std::vector<double> values;

    values.reserve(coo.elements.size());
    col_idx.reserve(coo.elements.size());

    for(const auto&el :  coo.elements){
        values.push_back(el.val);
        col_idx.push_back(el.col);
        row_ptr[el.row+1]++;
    }

    for (size_t i = 0; i < coo.num_rows; i++)
    {
        row_ptr[i+1] += row_ptr[i];
    }

    return CSRMatrix(coo.num_rows,coo.num_cols,coo.nnz,row_ptr,col_idx,values);
}   
