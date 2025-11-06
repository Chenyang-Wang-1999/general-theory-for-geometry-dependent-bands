/*
 * @author        wangchenyang <cy-wang21@mails.tsinghua.edu.cn>
 * @date          2023-07-16
 * @lastModified  2023-07-17
 * Copyright © Department of Physics, Tsinghua University.  All rights reserved 
 */

#include <iostream>

template<typename... Types>
struct MyStruct
{
    // template<Types... args>
    // Types... args;
    // template<typename T>
    // T& operator[](unsigned int var_id)
    // {
    //     return args[var_id];
    // }
};


int main()
{
    MyStruct<double> struct_int;
    std::cout<< sizeof(struct_int) <<'\n';
}