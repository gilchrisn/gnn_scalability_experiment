#ifndef COMPARE_H
#define COMPARE_H

#include <utility>
bool cmp_max(std::pair<double, unsigned int> x, std::pair<double, unsigned int> y){
    return x.first > y.first;
}

bool cmp_max_unsigned(std::pair<unsigned int, unsigned int> x, std::pair<unsigned int, unsigned int> y){
    return x.first > y.first;
}

#endif