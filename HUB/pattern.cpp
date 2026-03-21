#ifndef PATTERN
#define PATTERN

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <queue>
#include <fstream>
#include <set>
#include <math.h>
#include <sstream>

class Pattern{
public:
    std::vector<int> NTypes;
    std::vector<int> ETypes;
    std::vector<int> EDirect;
    int instance;
    explicit Pattern(){
        this->instance = -1;
        NTypes.push_back(-1);
    }

    void print(){
        std::cout<<"% ";
        for(int i=0;i<ETypes.size();i++){
            std::cout<<ETypes[i];
            if(this->EDirect[i] == 1){
                std::cout<<"->";
            }
            else{
                std::cout<<"<-";
            }
        }
        std::cout<<"\tinstance:"<<this->instance<<std::endl;
    }

    void clear(){
        NTypes.clear();
        ETypes.clear();
        EDirect.clear();
        instance = -1;

        NTypes.push_back(-1);
    }
};

#endif