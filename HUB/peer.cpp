
#ifndef PEER
#define PEER

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <queue>
#include <fstream>
#include <set>
#include <math.h>
#include <sstream>
#include "pattern.cpp"
#include "hin.cpp"
#include <chrono>
#include "param.h"

std::vector<unsigned int>* MidHop(unsigned int s, Pattern *pattern, HeterGraph *g, unsigned int meta_layer,
                                      std::vector<std::vector<bool> *> *visited,
                                      std::vector<std::pair<unsigned int, unsigned int>> *visited_mod,
                                      std::vector<std::vector<unsigned int> *>* ractive){
    auto frontiers = new std::vector<unsigned int>();
    std::queue<unsigned int> que;
    std::queue<unsigned int> layer_que;

    que.push(s);
    layer_que.push(0);

    while(!que.empty()){
        unsigned int v = que.front();
        que.pop();
        unsigned int pattern_pos = layer_que.front();
        layer_que.pop();

        if(!visited->at(pattern_pos)->at(v)){
            visited->at(pattern_pos)->at(v) = true;
            visited_mod->push_back(std::pair<unsigned int, unsigned int>(pattern_pos, v));
            if(pattern_pos == meta_layer){
                frontiers->push_back(v);
                continue;
            }
            if(pattern->EDirect[pattern_pos]==1){
                int guard_begin=-1, guard_end=-1;
                for (auto &i : *g->NGET[v]) {
                    unsigned int et = i.etype;
                    if (et == pattern->ETypes[pattern_pos]){
                        guard_begin = i.begin;
                        guard_end = i.end;
                        break;
                    }
                }
                if(guard_begin < 0) {
                    continue;
                }
                for(auto i=(unsigned int)guard_begin;i<guard_end;i++){
                    unsigned int n = g->ETL[pattern->ETypes[pattern_pos]]->at(i);
                    if(ractive->at(pattern_pos+1)->at(n) > 0) {
                        que.push(n);
                        layer_que.push(pattern_pos + 1);
                    }
                }
            }
            else{
                int guard_begin=-1, guard_end=-1;
                for (auto &i : *g->rNGET[v]) {
                    unsigned int et = i.etype;
                    if (et == pattern->ETypes[pattern_pos]){
                        guard_begin = i.begin;
                        guard_end = i.end;
                        break;
                    }
                }
                if(guard_begin < 0) {
                    continue;
                }
                for(auto i=(unsigned int)guard_begin;i<guard_end;i++){
                    unsigned int n = g->rETL[pattern->ETypes[pattern_pos]]->at(i);
                    if(ractive->at(pattern_pos+1)->at(n) > 0) {
                        que.push(n);
                        layer_que.push(pattern_pos + 1);
                    }
                }
            }

        }
    }
    return frontiers;
}

std::vector<unsigned int>* rMidHop(unsigned int s, Pattern *pattern, HeterGraph *g, unsigned int meta_layer,
                                       std::vector<std::vector<bool> *> *visited,
                                       std::vector<std::pair<unsigned int, unsigned int>> *visited_mod,
                                       std::vector<std::vector<unsigned int> *>* ractive){
    auto peers = new std::vector<unsigned int>();
    std::queue<unsigned int> que;
    std::queue<unsigned int> layer_que;
    que.push(s);
    layer_que.push(meta_layer);

    while(!que.empty()){
        unsigned int v = que.front();
        que.pop();
        unsigned int pattern_pos = layer_que.front();
        layer_que.pop();

        if(!visited->at(pattern_pos)->at(v)){
            visited->at(pattern_pos)->at(v) = true;
            visited_mod->push_back(std::pair<unsigned int, unsigned int>(pattern_pos, v));
            if(pattern_pos == 0){
                peers->push_back(v);
                continue;
            }

            if(pattern->EDirect[pattern_pos-1]==1){
                int guard_begin=-1, guard_end = -1;
                for (auto &i : *g->rNGET[v]) {
                    unsigned int et = i.etype;
                    if (et == pattern->ETypes[pattern_pos-1]){
                        guard_begin = i.begin;
                        guard_end = i.end;
                        break;
                    }
                }
                if(guard_begin < 0) continue;
                for(auto i=(unsigned int)guard_begin;i<guard_end;i++){
                    unsigned int n = g->rETL[pattern->ETypes[pattern_pos-1]]->at(i);
                    if(ractive->at(pattern_pos-1)->at(n) > 0) {
                        que.push(n);
                        layer_que.push(pattern_pos - 1);
                    }
                }
            }
            else{
                int guard_begin=-1, guard_end = -1;
                for (auto &i : *g->NGET[v]) {
                    unsigned int et = i.etype;
                    if (et == pattern->ETypes[pattern_pos-1]){
                        guard_begin = i.begin;
                        guard_end = i.end;
                        break;
                    }
                }
                if(guard_begin < 0) continue;
                for(auto i=(unsigned int)guard_begin;i<guard_end;i++){
                    unsigned int n = g->ETL[pattern->ETypes[pattern_pos-1]]->at(i);
                    if(ractive->at(pattern_pos-1)->at(n) > 0) {
                        que.push(n);
                        layer_que.push(pattern_pos - 1);
                    }
                }
            }
        }
    }
    return peers;
}

std::vector<unsigned int>* PatternHop(unsigned int s, Pattern *pattern, HeterGraph *g,
                                   std::vector<std::vector<bool> *> *visited,
                                   std::vector<std::pair<unsigned int, unsigned int>> *visited_mod){
    auto frontiers = new std::vector<unsigned int>();
    std::queue<unsigned int> que;
    std::queue<unsigned int> layer_que;

    que.push(s);
    layer_que.push(0);

    while(!que.empty()){
        unsigned int v = que.front();
        que.pop();
        unsigned int pattern_pos = layer_que.front();
        layer_que.pop();

        if(!visited->at(pattern_pos)->at(v)){
            visited->at(pattern_pos)->at(v) = true;
            visited_mod->push_back(std::pair<unsigned int, unsigned int>(pattern_pos, v));
            if(pattern_pos == pattern->ETypes.size()){
                frontiers->push_back(v);
                continue;
            }

            if(pattern->EDirect[pattern_pos]==1){
                int guard_begin=-1, guard_end=-1;
                for (auto &i : *g->NGET[v]) {
                    unsigned int et = i.etype;
                    if (et == pattern->ETypes[pattern_pos]){
                        guard_begin = i.begin;
                        guard_end = i.end;
                        break;
                    }
                }
                if(guard_begin < 0) {
                    continue;
                }
                for(auto i=(unsigned int)guard_begin;i<guard_end;i++){
                    que.push(g->ETL[pattern->ETypes[pattern_pos]]->at(i));
                    layer_que.push(pattern_pos + 1);
                }
            }
            else{
                int guard_begin=-1, guard_end=-1;
                for (auto &i : *g->rNGET[v]) {
                    unsigned int et = i.etype;
                    if (et == pattern->ETypes[pattern_pos]){
                        guard_begin = i.begin;
                        guard_end = i.end;
                        break;
                    }
                }

                if(guard_begin < 0) {
                    continue;
                }
                for(auto i=(unsigned int)guard_begin;i<guard_end;i++){
                    que.push(g->rETL[pattern->ETypes[pattern_pos]]->at(i));
                    layer_que.push(pattern_pos + 1);
                }
            }

        }
    }
    return frontiers;
}

std::vector<unsigned int>* rPatternHop(unsigned int s, Pattern *pattern, HeterGraph *g,
                                    std::vector<std::vector<bool> *> *visited,
                                    std::vector<std::pair<unsigned int, unsigned int>> *visited_mod){
    auto peers = new std::vector<unsigned int>();
    std::queue<unsigned int> que;
    std::queue<unsigned int> layer_que;
    que.push(s);
    layer_que.push(pattern->ETypes.size());
    while(!que.empty()){
        unsigned int v = que.front();
        que.pop();
        unsigned int pattern_pos = layer_que.front();
        layer_que.pop();

        if(!visited->at(pattern_pos)->at(v)){
            visited->at(pattern_pos)->at(v) = true;

            visited_mod->push_back(std::pair<unsigned int, unsigned int>(pattern_pos, v));

            if(pattern_pos == 0){
                peers->push_back(v);
                continue;
            }

            if(pattern->EDirect[pattern_pos-1]==1){
                int guard_begin=-1, guard_end = -1;
                for (auto &i : *g->rNGET[v]) {
                    unsigned int et = i.etype;
                    if (et == pattern->ETypes[pattern_pos-1]){
                        guard_begin = i.begin;
                        guard_end = i.end;
                        break;
                    }
                }
                if(guard_begin < 0) continue;
                for(auto i=(unsigned int)guard_begin;i<guard_end;i++){
                    que.push(g->rETL[pattern->ETypes[pattern_pos-1]]->at(i));
                    layer_que.push(pattern_pos-1);
                }
            }
            else{
                int guard_begin=-1, guard_end = -1;
                for (auto &i : *g->NGET[v]) {
                    unsigned int et = i.etype;
                    if (et == pattern->ETypes[pattern_pos-1]){
                        guard_begin = i.begin;
                        guard_end = i.end;
                        break;
                    }
                }
                if(guard_begin < 0) continue;
                for(auto i=(unsigned int)guard_begin;i<guard_end;i++){
                    que.push(g->ETL[pattern->ETypes[pattern_pos-1]]->at(i));
                    layer_que.push(pattern_pos-1);
                }
            }
        }
    }
    return peers;
}

std::vector<unsigned int>* Peers(unsigned int qn, Pattern *pattern, unsigned int meta_layer, HeterGraph *g,
        std::vector<std::vector<bool>*> *visited, std::vector<std::vector<unsigned int> *>* ractive){
    auto peers = new std::vector<unsigned int>();
    auto visited_mod = new std::vector<std::pair<unsigned int, unsigned int>>();
    auto midfrontiers = MidHop(qn, pattern, g, meta_layer, visited, visited_mod, ractive);
    for (auto &it : *visited_mod) visited->at(it.first)->at(it.second) = false;
    visited_mod->clear();
    for(unsigned int f: *midfrontiers){
        auto peers_new=rMidHop(f, pattern, g, meta_layer, visited, visited_mod, ractive);
        for(unsigned int p: *peers_new){
            peers->push_back(p);
        }
        delete peers_new;
    }
    for(auto &it: *visited_mod) visited->at(it.first)->at(it.second)=false;
    visited_mod->clear();
    return peers;
}

void Peers(Pattern *pattern, HeterGraph *g, std::vector<unsigned int> *frontiers, std::vector<unsigned int> *peers,
           std::vector<std::vector<bool> *> *visited){
    auto visited_mod = new std::vector<std::pair<unsigned int, unsigned int>>();
    if(pattern->instance == -1) {
        for (unsigned int u = 0; u < g->NT.size(); u++){
            if (pattern->NTypes[pattern->ETypes.size()] != -1 &&
                std::find(g->NT[u]->begin(), g->NT[u]->end(), pattern->NTypes[pattern->ETypes.size()]) == g->NT[u]->end())
                continue;
            auto peers_from_u = rPatternHop(u, pattern, g, visited, visited_mod);
            for (unsigned int p : *peers_from_u) peers->push_back(p);
            frontiers->push_back(u);
            delete peers_from_u;
        }
    }
    else{
        frontiers->push_back((unsigned int)pattern->instance);
        auto ipeers = rPatternHop((unsigned int) pattern->instance, pattern, g, visited, visited_mod);
        for(unsigned int p: *ipeers) peers->push_back(p);
        delete ipeers;
    }
    for (auto &it: *visited_mod) visited->at(it.first)->at(it.second) = false;
    delete visited_mod;
    
}

std::vector<std::set<unsigned int>* >* ActiveMidNodes(std::vector<unsigned int> *peers,
                                                      std::vector<unsigned int> *frontiers, Pattern *pattern,
                                                      HeterGraph *g,
                                                      std::vector<std::vector<bool> *> *forward_visited,
                                                      std::vector<std::vector<bool> *> *backward_visited){
    auto active = new std::vector<std::set<unsigned int>* >();

    auto forward_visited_mod = new std::vector<std::pair<unsigned int, unsigned int>>();
    auto backward_visited_mod = new std::vector<std::pair<unsigned int, unsigned int>>();

    for (unsigned int p : *peers) {
        auto temp = PatternHop(p, pattern, g, forward_visited, forward_visited_mod);
        delete temp;
    }
    for (unsigned int frontier : *frontiers) {
        auto temp = rPatternHop(frontier, pattern, g, backward_visited, backward_visited_mod);
        delete temp;
    }

    for(unsigned int n=0;n<g->NT.size();n++){
        active->push_back(new std::set<unsigned int>());
        for(unsigned int l=0;l<=pattern->ETypes.size();l++)
            if(forward_visited->at(l)->at(n) && backward_visited->at(l)->at(n)) active->at(n)->insert(l);
    }

    for (auto &it : *forward_visited_mod) forward_visited->at(it.first)->at(it.second) = false;
    for (auto &it : *backward_visited_mod) backward_visited->at(it.first)->at(it.second) = false;
    delete forward_visited_mod;
    delete backward_visited_mod;
    return active;
}

#endif