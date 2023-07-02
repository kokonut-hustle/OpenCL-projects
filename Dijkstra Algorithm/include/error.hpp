#pragma once

#include <iostream>

void check_error_file_line(int err_num, int expected, const char *file, const int line_number) {
    if (err_num != expected) {
        std::cerr << "Error at line " << line_number << " in file " << file << std::endl;
        exit(1);
    }
}

#define check_error(a, b) check_error_file_line(a, b, __FILE__, __LINE__)
