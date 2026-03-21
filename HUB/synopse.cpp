#ifndef SYNOPSE
#define SYNOPSE

#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <algorithm>
#include <queue>
#include <fstream>
#include <set>
#include <math.h>
#include <sstream>
#include <unordered_map>
#include <chrono>
#include "pattern.cpp"
#include "hin.cpp"
#include "param.h"
#include "hidden_edge.cpp"


namespace sy{
    struct Synopse {
        std::vector<unsigned int> mink;
        unsigned int size;
        unsigned int p;
        unsigned int r;  // the random value corresponding to "p"
    };

    void init_synopse(Synopse *s, unsigned int _p) {
        s->mink.resize(K, 0);
        s->size = 0;
        s->p = _p;
    }

    void clear(Synopse *s) {
        for (unsigned int i = 0; i < s->size; i++) s->mink[i] = 0;
        s->size = 0;
    }

    unsigned int choice(Synopse *s, unsigned int rand_num, std::unordered_map<unsigned int, unsigned int>* rand2p){
        unsigned int pos = rand_num % s->size;
        return rand2p->at(s->mink[pos]);
    }

    std::vector<unsigned int>* sample(Synopse *s, unsigned int k, std::random_device dev,
            std::unordered_map<unsigned int, unsigned int>* rand2p){
        std::vector<unsigned int> values;
        for(unsigned int i=0;i<s->size;i++) values.push_back(s->mink[i]);
        std::shuffle(values.begin(), values.end(), dev);
        auto sampl = new std::vector<unsigned int>();
        if(k>= s->size){
            for(unsigned int i=0;i<s->size;i++){
                unsigned int value = values[i];
                sampl->push_back(rand2p->at(value));
            }
        }
        else{
            for(unsigned int i=0;i<k;i++){
                unsigned int value = values[i];
                sampl->push_back(rand2p->at(value));
            }
        }
        return sampl;
    }

    void add(Synopse *s, unsigned int x){
        if(s->r > 0 && s->r == x) return;

        int mink_size = s->size;
        for (unsigned int i = 0; i < s->size; i++) if (s->mink[i] == x) return;
        if (mink_size < K) {
            s->mink[s->size] = x;
            s->size += 1;
            return;
        }
        int top_pos = 0, top = 0;
        for (unsigned int i = 0; i < mink_size; i++) {
            if (s->mink[i] > top) {
                top = s->mink[i];
                top_pos = i;
            }
        }
        if (top > x) s->mink[top_pos] = x;
    }

    double estimate(Synopse *s){
        if (s->size < K) return s->size;
        else {
            int top = 0;
            for (unsigned int i = 0; i < s->size; i++) if (s->mink[i] > top) top = s->mink[i];
            double r = static_cast<double>(top) / RAND_MAX;
            return (K - 1) / r;
        }
    }

    std::map<unsigned int, double> *synopses(unsigned int meta_layer, std::vector<HiddenEdge> *hidden_edges,
                                             std::vector<std::vector<Synopse> *> *synopses,
                                             std::vector<bool> *qualified_nodes = nullptr) {
        auto degs = new std::map<unsigned int, double>();
        std::random_device dev;
        std::mt19937 generator(dev());
        std::uniform_int_distribution<std::mt19937::result_type> distribute(1, RAND_MAX);

        int meta_layer_count = -1;
        for (unsigned int ec = 0; ec < hidden_edges->size(); ec++) {
            unsigned int p = hidden_edges->at(ec).s;
            unsigned int nbr = hidden_edges->at(ec).t;
            unsigned int l = hidden_edges->at(ec).l;
            if (l == 0 && synopses->at(0)->at(p).size == 0) {
                unsigned int n=synopses->at(0)->at(p).p;
                if(qualified_nodes == nullptr || qualified_nodes->at(n)) {
                    int rand = distribute(generator);
                    add(&synopses->at(0)->at(p), (unsigned int) rand);
                }
            }
            if (l >= meta_layer) {
                meta_layer_count = ec - 1;
                break;
            }
            for (unsigned int h = 0; h < synopses->at(l)->at(p).size; h++)
                add(&synopses->at(l + 1)->at(nbr), synopses->at(l)->at(p).mink[h]);
        }

        if (meta_layer_count < 0) meta_layer_count = hidden_edges->size() - 1;

        for (unsigned int l = 0; l < meta_layer; l++) for (auto &i: *synopses->at(l)) clear(&i);

        for (int ec = meta_layer_count; ec >= 0; ec--) {
            unsigned int p = hidden_edges->at((unsigned int) ec).t;
            unsigned int nbr = hidden_edges->at((unsigned int) ec).s;
            unsigned int l = 1 + hidden_edges->at((unsigned int) ec).l;

            for (unsigned int h = 0; h < synopses->at(l)->at(p).size; h++)
                add(&synopses->at(l - 1)->at(nbr), synopses->at(l)->at(p).mink[h]);
        }

        for (auto &i : *synopses->at(0)) {
            unsigned int _p = i.p;
            if (qualified_nodes == nullptr || qualified_nodes->at(_p))
                degs->insert(std::pair<unsigned int, double>(_p, estimate(&i)));
        }
        for (unsigned int l = 0; l <= meta_layer; l++) for (auto &i : *synopses->at(l)) clear(&i);
        return degs;
    }
}
#endif