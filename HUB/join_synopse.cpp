#ifndef JOINTSYNOPSE
#define JOINTSYNOPSE

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
#include <limits>
#include <map>
#include <chrono>
#include "pattern.cpp"
#include "hin.cpp"
#include "param.h"
#include "hidden_edge.cpp"

namespace jsy {  //join-synopses
    struct Synopse {
        std::vector<unsigned int> minks;
        std::vector<unsigned int> sizes;
        unsigned int p;

        std::vector<unsigned int> rs;  // the random values corresponding to "p"

        double clique_est;
    };


    void init_synopse(Synopse *s, unsigned int _p) {
        s->minks.resize(K * L, 0); 
        s->sizes.resize(L, 0);    
        s->rs.resize(L, 0);       
        
        s->p = _p;
        s->clique_est = 0;
    }

    void clear(Synopse *s, bool clear_rs) {
        for (unsigned int i = 0; i < L; i++) {
            for (unsigned int j = 0; j < s->sizes[i]; j++) s->minks[i * K + j] = 0;
            s->sizes[i] = 0;
            if(clear_rs) s->rs[i] = 0;
        }
    }

//    bool self_ref(Synopse *s){
//        for(unsigned int l=0;l<L;l++){
//            for(unsigned int k=0;k<K;k++){
//                unsigned int bias = K*l + k;
//                unsigned int rand = s->minks[bias];
//                if(rand == s->rs[l]) return true;
//            }
//        }
//        return false;
//    }
//    }
//
//    bool rs_check(Synopse *s, std::vector<std::unordered_map<unsigned int, unsigned int>*>* rand2ps){
//        for(unsigned int l=0;l<L;l++){
//            unsigned int rand = s->rs[l];
//            unsigned int pp = rand2ps->at(l)->at(rand);
//            if(pp != s->p) return false;
//        }
//        return true;
//    }

    // choice a random nbr of "p" from its last synopse
    unsigned int choice(Synopse *s, unsigned int rand_num /* the range of "rand_num" is [1, RAND_MAX]*/,
            std::vector<std::unordered_map<unsigned int, unsigned int>*>* rand2ps){
        unsigned int pos = rand_num % s->sizes[L-1];
        return rand2ps->at(L-1)->at(s->minks[K * (L-1) + pos]);
    }

    std::vector<unsigned int>* sample(Synopse *s, unsigned int num_sampl, std::mt19937 rg, unsigned int layer,
                                      std::vector<std::unordered_map<unsigned int, unsigned int>*>* rand2ps){
        std::vector<unsigned int> values;
        for(unsigned int i=0;i<s->sizes[layer];i++) values.push_back(s->minks[i + K * layer]);
        std::shuffle(values.begin(), values.end(), rg);

        auto sampl = new std::vector<unsigned int>();
        if(num_sampl >= s->sizes[layer]){
            for(unsigned int i=0;i<s->sizes[layer];i++){
                unsigned int value = values[i];
                sampl->push_back(rand2ps->at(layer)->at(value));
            }
        }
        else{
            for(unsigned int i=0;i<num_sampl;i++){
                unsigned int value = values[i];
                sampl->push_back(rand2ps->at(layer)->at(value));
            }
        }
        return sampl;
    }

    void add(Synopse *s, Synopse *t) {
        // add Synopse t into Synopse s
        for (unsigned int l = 0; l < L; l++) {
            for (unsigned int i = 0; i < t->sizes[l]; i++) {
                unsigned int x = t->minks[l * K + i];

                if(s->rs[l] > 0 && s->rs[l] == x) continue;

                bool exists = false;
                for (unsigned int j = 0; j < s->sizes[l]; j++)
                    if (s->minks[l * K + j] == x) {
                        exists = true;
                        break;
                    }
                if (exists) continue;
                if (s->sizes[l] < K) {
                    s->minks[l * K + s->sizes[l]] = x;
                    s->sizes[l]++;
                    exists = true;
                }
                if (exists) continue;
                int top_pos = 0, top = 0;
                for (unsigned int j = 0; j < s->sizes[l]; j++) {
                    if (s->minks[l * K + j] > top) {
                        top = s->minks[l * K + j];
                        top_pos = j;
                    }
                }
                if (top > x) s->minks[l * K + top_pos] = x;
            }
        }
    }

    double estimate(Synopse *s) {
        double est = 0;
        for (unsigned int l = 0; l < L; l++) {
            if (s->sizes[l] < K) est += s->sizes[l];
            else {
                int top = 0;
                for (unsigned int i = 0; i < s->sizes[l]; i++) if (s->minks[l * K + i] > top) top = s->minks[l * K + i];
                double r = static_cast<double>(top) / RAND_MAX;
                est += ((K - 1) / r);
            }
        }
        est /= L;
        return est;
    }

    std::map<unsigned int, double>* synopses_fp_deg(unsigned int meta_layer, std::vector<HiddenEdge> *hidden_edges,
                                                std::vector<std::vector<Synopse> *>* synopses, unsigned int q_deg,
                                                double topr, unsigned int peer_size,
                                                std::vector<bool>* qualified_nodes = nullptr){
        auto degs = new std::map<unsigned int, double>();
        
        std::mt19937 generator(SEED);
        std::uniform_int_distribution<std::mt19937::result_type> distribute(1, RAND_MAX);

        int meta_layer_count = -1;
        for (unsigned int ec = 0; ec < hidden_edges->size(); ec++) {
            unsigned int p = hidden_edges->at(ec).s;
            unsigned int nbr = hidden_edges->at(ec).t;
            unsigned int l = hidden_edges->at(ec).l;
            if(l == 0 && synopses->at(0)->at(p).sizes[0] == 0){
                unsigned int n=synopses->at(0)->at(p).p;
                if(qualified_nodes == nullptr || qualified_nodes->at(n))
                    for (unsigned int ll = 0; ll < L; ll++) {
                        int rand = distribute(generator);
                        synopses->at(0)->at(p).minks[ll * K] = (unsigned int) rand;
                        synopses->at(0)->at(p).sizes[ll] = 1;
                    }
            }
            if (l >= meta_layer) {
                meta_layer_count = ec - 1;
                break;
            }
            add(&synopses->at(l + 1)->at(nbr), &synopses->at(l)->at(p));
        }

        for (auto &i: *synopses->at(meta_layer)) {
            double clique_size = estimate(&i);
            if(clique_size >= q_deg && clique_size > peer_size * topr) return nullptr;
        }

        for(unsigned int l=1;l<meta_layer;l++) for(auto &i: *synopses->at(l)) i.clique_est = estimate(&i);

        if (meta_layer_count < 0) meta_layer_count = hidden_edges->size() - 1;

        for (unsigned int l = 0; l < meta_layer; l++) for (auto &i: *synopses->at(l)) clear(&i, true);

        unsigned int pre_nbr = hidden_edges->at((unsigned int)meta_layer_count).s;
        unsigned int pre_l = 1 + hidden_edges->at((unsigned int)meta_layer_count).l;

        for (int ec = meta_layer_count; ec >= 0; ec--) {
            unsigned int p = hidden_edges->at((unsigned int) ec).t;
            unsigned int nbr = hidden_edges->at((unsigned int) ec).s;
            unsigned int l = 1 + hidden_edges->at((unsigned int) ec).l;

            if((pre_nbr != nbr || pre_l != l) && l > 1){
                if(synopses->at(pre_l-1)->at(pre_nbr).clique_est > peer_size * topr &&
                    estimate(&synopses->at(pre_l-1)->at(pre_nbr))>=q_deg) return nullptr;
                pre_nbr = nbr;
                pre_l = l;
            }

            add(&synopses->at(l - 1)->at(nbr), &synopses->at(l)->at(p));
        }

        for (auto &i : *synopses->at(0)) {
            unsigned int _p = i.p;
            if (qualified_nodes == nullptr || qualified_nodes->at(_p))
                degs->insert(std::pair<unsigned int, double>(_p, estimate(&i)));
        }
        for (unsigned int l = 0; l <= meta_layer; l++) for (auto &i : *synopses->at(l)) clear(&i, true);
        return degs;
    }

    std::map<unsigned int, double>* synopses_fp(unsigned int meta_layer, std::vector<HiddenEdge> *hidden_edges,
                                                std::vector<std::vector<Synopse> *>* synopses, unsigned int q_deg,
                                                double topr, unsigned int peer_size, double & mcs,
                                                std::vector<bool> *qualified_nodes = nullptr){
        auto degs = new std::map<unsigned int, double>();
        
        std::mt19937 generator(SEED);
        std::uniform_int_distribution<std::mt19937::result_type> distribute(1, RAND_MAX);

        int meta_layer_count = -1;
        for (unsigned int ec = 0; ec < hidden_edges->size(); ec++) {
            unsigned int p = hidden_edges->at(ec).s;
            unsigned int nbr = hidden_edges->at(ec).t;
            unsigned int l = hidden_edges->at(ec).l;
            if(l == 0 && synopses->at(0)->at(p).sizes[0] == 0){
                unsigned int n=synopses->at(0)->at(p).p;
                if(qualified_nodes == nullptr || qualified_nodes->at(n))
                    for (unsigned int ll = 0; ll < L; ll++) {
                        int rand = distribute(generator);
                        synopses->at(0)->at(p).minks[ll * K] = (unsigned int) rand;
                        synopses->at(0)->at(p).sizes[ll] = 1;
                    }
            }
            if (l >= meta_layer) {
                meta_layer_count = ec - 1;
                break;
            }
            add(&synopses->at(l + 1)->at(nbr), &synopses->at(l)->at(p));
        }

        mcs = 0;

        for (auto &i: *synopses->at(meta_layer)) {
            unsigned int _p = i.p;
            double clique_size = estimate(&i);
            if(clique_size > mcs) mcs = clique_size;
            if(clique_size >= q_deg && clique_size > peer_size * topr) return nullptr;
        }

        if (meta_layer_count < 0) meta_layer_count = hidden_edges->size() - 1;

        for (unsigned int l = 0; l < meta_layer; l++) for (auto &i: *synopses->at(l)) clear(&i, true);

        for (int ec = meta_layer_count; ec >= 0; ec--) {
            unsigned int p = hidden_edges->at((unsigned int) ec).t;
            unsigned int nbr = hidden_edges->at((unsigned int) ec).s;
            unsigned int l = 1 + hidden_edges->at((unsigned int) ec).l;
            add(&synopses->at(l - 1)->at(nbr), &synopses->at(l)->at(p));
        }

        for (auto &i : *synopses->at(0)) {
            unsigned int _p = i.p;
            if (qualified_nodes == nullptr || qualified_nodes->at(_p))
                degs->insert(std::pair<unsigned int, double>(_p, estimate(&i)));
        }
        for (unsigned int l = 0; l <= meta_layer; l++) for (auto &i : *synopses->at(l)) clear(&i, true);
        return degs;
    }

    std::vector<std::unordered_map<unsigned int, unsigned int>*> *gnn_synopses(unsigned int meta_layer,
                                                std::vector<HiddenEdge> *hidden_edges,
                                                std::vector<std::vector<Synopse> *> *synopses,
                                                std::vector<bool> *qualified_nodes = nullptr) {


        std::mt19937 generator(SEED);
        // Use full 32-bit range (not RAND_MAX which is only 32767 on 32-bit MinGW)
        static constexpr unsigned int GNN_HASH_MAX = std::numeric_limits<unsigned int>::max();
        std::uniform_int_distribution<std::mt19937::result_type> distribute(1, GNN_HASH_MAX);

        auto rand2ps = new std::vector<std::unordered_map<unsigned int, unsigned int>*>();
        for(unsigned int l=0;l<L;l++) rand2ps->push_back(new std::unordered_map<unsigned int, unsigned int>());

        int meta_layer_count = -1;
        for (unsigned int ec = 0; ec < hidden_edges->size(); ec++) {
            unsigned int p = hidden_edges->at(ec).s;
            unsigned int nbr = hidden_edges->at(ec).t;
            unsigned int l = hidden_edges->at(ec).l;
            if(l == 0 && synopses->at(0)->at(p).sizes[0] == 0){
                unsigned int n=synopses->at(0)->at(p).p;
                if(qualified_nodes == nullptr || qualified_nodes->at(n))
                    for (unsigned int ll = 0; ll < L; ll++){
                        int rand = distribute(generator);
                        while(rand2ps->at(ll)->count(rand) > 0) rand = distribute(generator);
                        synopses->at(0)->at(p).minks[ll * K] = (unsigned int) rand;
                        synopses->at(0)->at(p).sizes[ll] = 1;
                        (*rand2ps->at(ll))[rand] = n;
                        synopses->at(0)->at(p).rs[ll] = (unsigned int)rand;
                    }
            }
            if (l >= meta_layer) {
                meta_layer_count = ec - 1;
                break;
            }
            add(&synopses->at(l + 1)->at(nbr), &synopses->at(l)->at(p));
        }

        if (meta_layer_count < 0) meta_layer_count = hidden_edges->size() - 1;

        for (unsigned int l = 0; l < meta_layer; l++) for (auto &i: *synopses->at(l)) clear(&i, false);

        for (int ec = meta_layer_count; ec >= 0; ec--) {
            unsigned int p = hidden_edges->at((unsigned int) ec).t;
            unsigned int nbr = hidden_edges->at((unsigned int) ec).s;
            unsigned int l = 1 + hidden_edges->at((unsigned int) ec).l;
            add(&synopses->at(l - 1)->at(nbr), &synopses->at(l)->at(p));
        }
        return rand2ps;
    }

    std::map<unsigned int, double> *synopses(unsigned int meta_layer, std::vector<HiddenEdge> *hidden_edges,
                                             std::vector<std::vector<Synopse> *> *synopses,
                                             std::vector<bool> *qualified_nodes = nullptr) {
        auto degs = new std::map<unsigned int, double>();
        
        std::mt19937 generator(SEED);
        std::uniform_int_distribution<std::mt19937::result_type> distribute(1, RAND_MAX);

        int meta_layer_count = -1;
        for (unsigned int ec = 0; ec < hidden_edges->size(); ec++) {
            unsigned int p = hidden_edges->at(ec).s;
            unsigned int nbr = hidden_edges->at(ec).t;
            unsigned int l = hidden_edges->at(ec).l;
            if(l == 0 && synopses->at(0)->at(p).sizes[0] == 0){
                unsigned int n=synopses->at(0)->at(p).p;
                if(qualified_nodes == nullptr || qualified_nodes->at(n))
                    for (unsigned int ll = 0; ll < L; ll++) {
                        int rand = distribute(generator);
                        synopses->at(0)->at(p).minks[ll * K] = (unsigned int) rand;
                        synopses->at(0)->at(p).sizes[ll] = 1;
                    }
            }
            if (l >= meta_layer) {
                meta_layer_count = ec - 1;
                break;
            }
            add(&synopses->at(l + 1)->at(nbr), &synopses->at(l)->at(p));
        }

        if (meta_layer_count < 0) meta_layer_count = hidden_edges->size() - 1;

        for (unsigned int l = 0; l < meta_layer; l++) for (auto &i: *synopses->at(l)) clear(&i, true);

        for (int ec = meta_layer_count; ec >= 0; ec--) {
            unsigned int p = hidden_edges->at((unsigned int) ec).t;
            unsigned int nbr = hidden_edges->at((unsigned int) ec).s;
            unsigned int l = 1 + hidden_edges->at((unsigned int) ec).l;
            add(&synopses->at(l - 1)->at(nbr), &synopses->at(l)->at(p));
        }

        for (auto &i : *synopses->at(0)) {
            unsigned int _p = i.p;
            //if (qualified_nodes == nullptr || qualified_nodes->at(_p)) 
            degs->insert(std::pair<unsigned int, double>(_p, estimate(&i)));
        }
        for (unsigned int l = 0; l <= meta_layer; l++) for (auto &i : *synopses->at(l)) clear(&i, true);
        return degs;
    }
}
#endif