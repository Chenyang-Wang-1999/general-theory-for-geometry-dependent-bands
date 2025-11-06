/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-07-29
 * @lastModified  2023-09-07
 * Copyright © Department of Physics, Tsinghua University. All rights reserved 
 */
#ifndef DEBUG_TOOLS_HPP
#define DEBUG_TOOLS_HPP
#include "basic_defs.h"
#include <string>
#include <iostream>
#include <fstream>

#include <array>
template <typename Iterable>
void print_vector(Iterable vec)
{
    for(auto it = vec.begin(); it!=vec.end(); it++)
    {
        STDERR << (*it) << ',';
    }
}

template<typename Iterator>
void save_iter_to_file(std::string file_name, const Iterator & vec, int with_real_part)
{
    /*
        with_real_part:
            0: real vector
            1: complex vector, output real part
            2: complex vector, output imaginary part
    */
    std::ofstream out_file_pointer(file_name, std::ios::out);
    for(size_t j = 0; j < vec.size() - 1; j ++)
    {
        auto x = vec[j];
        switch (with_real_part)
        {
        case 0:
            out_file_pointer << x << ',';
            break;
        
        case 1:
            out_file_pointer << x.real() << ',';
            break;
        
        case 2:
            out_file_pointer << x.imag() << ',';
            break;
        }
    }
    auto x = vec.back();
    switch (with_real_part)
    {
    case 0:
        out_file_pointer << x ;
        break;
    
    case 1:
        out_file_pointer << x.real() ;
        break;
    
    case 2:
        out_file_pointer << x.imag() ;
        break;
    }
    out_file_pointer.close();
    
}



#endif