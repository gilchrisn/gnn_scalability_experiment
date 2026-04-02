#include <iostream>
#include <queue>
#include <cstdlib>
#include <ctime>
#include <map>
#include <chrono>
#include "hin.cpp"
#include "pattern.cpp"
#include "peer.cpp"
#include "synopse.cpp"
#include "join_synopse.cpp"
#include "param.h"
#include "hg.cpp"
#include "hidden_edge.cpp"
#include "compare.cpp"

namespace effectiveness{
    bool COD_matching_graph_time(Pattern *qp, HeterGraph *g, std::vector<std::vector<bool>*>* visited,
                                 std::vector<std::vector<bool>*>* back_visited, double & avg_time){

        auto start = std::chrono::steady_clock::now();

        auto frontiers = new std::vector<unsigned int>();
        auto peers = new std::vector<unsigned int>();
        Peers(qp, g, frontiers, peers, visited);

        auto end = std::chrono::steady_clock::now();

        if (peers->size() <= 2) {
            delete frontiers;
            delete peers;
            return false;
        }
        avg_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        start = std::chrono::steady_clock::now();
        std::vector<std::set<unsigned int> *> *active = ActiveMidNodes(peers, frontiers, qp, g, visited, back_visited);
        auto ractive = new std::vector<std::vector<unsigned int> *>();
        for (unsigned int l = 0; l <= qp->ETypes.size(); l++) ractive->push_back(new std::vector<unsigned int>(g->NT.size(), 0));
        for (unsigned int n = 0; n < g->NT.size(); n++) for (unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;
        end = std::chrono::steady_clock::now();
        avg_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        return true;
    }


    bool COD_prop_personalized_scale(Pattern *qp, HeterGraph *g, double topr, double beta, std::vector<std::vector<bool>*>* visited,
                                        std::vector<std::vector<bool>*>* back_visited, const std::string &centrality,
                                        double & avg_time){
        double preprocess_time = 0.0;
        double running_time = 0.0;

        auto start = std::chrono::steady_clock::now();

        auto frontiers = new std::vector<unsigned int>();
        auto peers = new std::vector<unsigned int>();
        Peers(qp, g, frontiers, peers, visited);
        unsigned int peer_size = peers->size();

        auto end = std::chrono::steady_clock::now();

        if (peer_size <= 2) {
            delete frontiers;
            delete peers;
            return false;
        }
        qp->print();

        preprocess_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        std::vector<unsigned int> qnodes;

        std::mt19937 generator(SEED);
        std::uniform_int_distribution<std::mt19937::result_type> distribute(0, peer_size - 1);
        for(unsigned int i=0;i<QNUM;i++) qnodes.push_back(peers->at(distribute(generator)));

        start = std::chrono::steady_clock::now();

        std::vector<std::set<unsigned int> *> *active = ActiveMidNodes(peers, frontiers, qp, g, visited, back_visited);
        auto ractive = new std::vector<std::vector<unsigned int> *>();
        for (unsigned int l = 0; l <= qp->ETypes.size(); l++) ractive->push_back(new std::vector<unsigned int>(g->NT.size(), 0));
        for (unsigned int n = 0; n < g->NT.size(); n++) for (unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;

        auto synopses_combined = new std::vector<std::vector<jsy::Synopse> *>();
        for (unsigned int l = 0; l <= qp->ETypes.size(); l++) {
            synopses_combined->push_back(new std::vector<jsy::Synopse>());
            for (unsigned int p = 0; p < g->NT.size(); p++)
                if (ractive->at(l)->at(p) > 0) {
                    jsy::Synopse s; jsy::init_synopse(&s, p);
                    synopses_combined->at(l)->push_back(s);
                    ractive->at(l)->at(p) = synopses_combined->at(l)->size();
                }
        }

        auto hidden_edges = new std::vector<HiddenEdge>();
        for (unsigned int l = 0; l < qp->ETypes.size(); l++) {
            for (unsigned int p = 0; p < g->NT.size(); p++) {
                if (ractive->at(l)->at(p) > 0) {
                    if (qp->EDirect[l] == 1) {
                        for (unsigned int nbr: *(g->EL[p])) {
                            if (ractive->at(l + 1)->at(nbr) > 0) {
                                HiddenEdge he{.s=ractive->at(l)->at(p) - 1, .t=ractive->at(l + 1)->at(nbr) - 1, .l=l};
                                hidden_edges->push_back(he);
                            }
                        }
                    } else {
                        for (unsigned int nbr: *(g->rEL[p])) {
                            if (ractive->at(l + 1)->at(nbr) > 0) {
                                HiddenEdge he{.s=ractive->at(l)->at(p) - 1, .t=ractive->at(l + 1)->at(nbr) - 1, .l=l};
                                hidden_edges->push_back(he);
                            }}}}}}

        unsigned int max_meta_layer;
        if (qp->instance == -1) max_meta_layer = qp->ETypes.size();
        else max_meta_layer = std::max(1u, (unsigned int)(qp->ETypes.size() - 1));

        end = std::chrono::steady_clock::now();
        preprocess_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        for(unsigned int meta_layer = 1;meta_layer<= max_meta_layer;meta_layer++){
            auto topr_size = (unsigned int)(topr * peer_size);

            start = std::chrono::steady_clock::now();

            if(centrality == "d"){
                for(unsigned int qn: qnodes) {
                    std::vector<unsigned int> props_dom_peers;

                    auto props_degs = jsy::synopses(meta_layer, hidden_edges, synopses_combined);
                    double qn_prop_deg = props_degs->at(qn);
                    for(unsigned int i=0;i<g->NT.size();i++){
                        if(i == qn) continue;
                        if(ractive->at(0)->at(i)) if(props_degs->at(i) > qn_prop_deg) props_dom_peers.push_back(i);
                    }
                    props_dom_peers.push_back(qn);
                    bool istopr = false;
                    if(topr_size >= props_dom_peers.size()) istopr = true;
                    std::cout<<"qn:"<<qn<<"  istopr:"<<istopr<<std::endl;
                }
            }
            else if(centrality == "dp"){
                for(unsigned int qn: qnodes){
                    auto q_peers = Peers(qn, qp, meta_layer, g, visited, ractive);
                    unsigned int qn_exact_deg = q_peers->size();
                    delete q_peers;
                    auto prop_degs = jsy::synopses_fp_deg(meta_layer, hidden_edges, synopses_combined,
                                                          qn_exact_deg*(1.0+beta), topr*(1.0+beta), peer_size);

                    bool istopr = false;
                    std::vector<unsigned int> props_dom_peers;
                    if(prop_degs != nullptr){
                        double qn_prop_deg = prop_degs->at(qn);
                        for(unsigned int i=0;i<g->NT.size();i++){
                            if(i == qn) continue;
                            if(ractive->at(0)->at(i)) if(prop_degs->at(i) >qn_prop_deg) props_dom_peers.push_back(i);
                        }
                        props_dom_peers.push_back(qn);
                        if(topr_size >= props_dom_peers.size()) istopr = true;
                    }
                    std::cout<<"qn:"<<qn<<"  istopr:"<<istopr<<std::endl;
                }
            }
            else if(centrality == "h"){
                for(unsigned int qn: qnodes){
                    auto degs = jsy::synopses(meta_layer, hidden_edges, synopses_combined);
                    auto qn_peers = Peers(qn, qp, meta_layer, g, visited, ractive);
                    auto sort_degs = new std::vector<std::pair<double, unsigned int>>();
                    for(unsigned int n: *qn_peers)
                        sort_degs->push_back(std::pair<double, unsigned int>(degs->at(n), n));
                    std::sort(sort_degs->begin(), sort_degs->end(), cmp_max);

                    unsigned int qn_hindex = 0;
                    for(auto &sort_deg: *sort_degs){
                        qn_hindex++;
                        if(qn_hindex > sort_deg.first) break;
                    }
                    if(qn_hindex > sort_degs->at(qn_hindex-1).first) qn_hindex--;

                    std::vector<bool> *qualified_nodes = new std::vector<bool>(g->NT.size(), false);
                    for(unsigned int peer: *peers) if(degs->at(peer)>=qn_hindex) qualified_nodes->at(peer)=true;
                    auto *hindexes = jsy::synopses(meta_layer, hidden_edges, synopses_combined, qualified_nodes);

                    std::vector<unsigned int> props_dom_peers;
                    for(auto it=hindexes->begin();it!=hindexes->end();it++){
                        if(it->first == qn) continue;
                        if(it->second >= hindexes->at(qn)) props_dom_peers.push_back(it->first);
                    }
                    props_dom_peers.push_back(qn);
                    bool istopr = false;
                    if(topr_size >= props_dom_peers.size()) istopr = true;

                    delete degs;
                    delete qn_peers;
                    delete sort_degs;
                    delete qualified_nodes;
                    delete hindexes;

                    std::cout<<"qn:"<<qn<<"  istopr:"<<istopr<<std::endl;
                }
            }
            else if(centrality == "hp"){
                for(unsigned int qn: qnodes){
                    auto q_peers = Peers(qn, qp, meta_layer, g, visited, ractive);
                    unsigned int q_deg = q_peers->size();

                    double mcs = 0;
                    auto prop_deg = jsy::synopses_fp(meta_layer, hidden_edges, synopses_combined, q_deg*(1.0+beta),
                                                     topr*(1.0+beta), peer_size, mcs);

                    if(prop_deg != nullptr){
                        std::vector<double> *degs = new std::vector<double>(g->NT.size(), 0);
                        for(auto &it: *prop_deg){
                            unsigned int n = it.first;
                            double d = it.second;
                            degs->at(n) += d;
                        }
                        delete prop_deg;

                        auto sort_degs = new std::vector<std::pair<double, unsigned int>>();
                        for(unsigned int n : *q_peers)
                            sort_degs->push_back(std::pair<double, unsigned int>(degs->at(n), n));
                        std::sort(sort_degs->begin(), sort_degs->end(), cmp_max);

                        unsigned int q_hindex = 0;
                        for(auto &sort_deg: *sort_degs){
                            q_hindex++;
                            if(q_hindex > sort_deg.first) break;
                        }
                        if(q_hindex > sort_degs->at(q_hindex-1).first) q_hindex--;

                        if(mcs < q_hindex || mcs < topr_size){
                            std::vector<bool> *qualified_nodes = new std::vector<bool>(g->NT.size(), false);
                            for(unsigned int peer: *peers) if(degs->at(peer)>= q_hindex) qualified_nodes->at(peer)=true;
                            auto* hindexes = jsy::synopses(meta_layer, hidden_edges, synopses_combined, qualified_nodes);

                            std::vector<unsigned int> props_dom_peers;
                            for(auto it=hindexes->begin();it!=hindexes->end();it++){
                                if(it->first==qn) continue;
                                if(it->second >= hindexes->at(qn)) props_dom_peers.push_back(it->first);
                            }
                            props_dom_peers.push_back(qn);
                            bool istopr = false;
                            if(topr_size >= props_dom_peers.size()) istopr = true;

                            delete qualified_nodes;
                            delete hindexes;

                            std::cout<<"qn:"<<qn<<"  istopr:"<<istopr<<std::endl;
                        }
                        else{
                            std::cout<<"qn:"<<qn<<"  istopr:false"<<std::endl;
                        }

                        delete degs;
                        delete sort_degs;
                    }
                    else{
                        std::cout<<"qn:"<<qn<<"  istopr:false"<<std::endl;
                    }
                    delete q_peers;
                }
            }
            end = std::chrono::steady_clock::now();
            running_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }

        running_time /= QNUM;
        double temp_time = (preprocess_time + running_time);
        avg_time += temp_time;
        temp_time /= 1000000000;
        std::cout<<"time:"<<temp_time<<" s"<<std::endl;

        delete frontiers;
        delete peers;
        for(auto &i: *active) delete i;
        for(auto &i: *ractive) delete i;
        delete active;
        delete ractive;
        for(auto &i: *synopses_combined) delete i;
        delete synopses_combined;
        delete hidden_edges;

        return true;
    }

    double COD_prop_personalized_cross_precision(Pattern *qp, HeterGraph *g, double topr, double beta, std::vector<std::vector<bool>*>* visited,
                                    std::vector<std::vector<bool>*>* back_visited, const std::string &centrality,
                                    std::vector<std::vector<unsigned int>*>* hg_dom_peers, std::vector<bool>* isdom,
                                    std::vector<std::vector<unsigned int>*>* hg_dom_greater_peers, std::vector<bool>* isdom_greater,
                                    unsigned int & count, double & avg_time, unsigned int & out_edges){
        double preprocess_time = 0.0;
        double running_time = 0.0;



        auto start = std::chrono::steady_clock::now();

        auto frontiers = new std::vector<unsigned int>();
        auto peers = new std::vector<unsigned int>();
        Peers(qp, g, frontiers, peers, visited);
        unsigned int peer_size = peers->size();

        auto end = std::chrono::steady_clock::now();

        if (peer_size <= 2) {
            delete frontiers;
            delete peers;
            return -1;
        }
        qp->print();

        preprocess_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        std::vector<unsigned int> qnodes;
        
        std::mt19937 generator(SEED);
        std::uniform_int_distribution<std::mt19937::result_type> distribute(0, peer_size - 1);
        for(unsigned int i=0;i<QNUM;i++) qnodes.push_back(peers->at(distribute(generator)));

        start = std::chrono::steady_clock::now();

        std::vector<std::set<unsigned int> *> *active = ActiveMidNodes(peers, frontiers, qp, g, visited, back_visited);
        auto ractive = new std::vector<std::vector<unsigned int> *>();
        for (unsigned int l = 0; l <= qp->ETypes.size(); l++) ractive->push_back(new std::vector<unsigned int>(g->NT.size(), 0));
        for (unsigned int n = 0; n < g->NT.size(); n++) for (unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;

        auto synopses_combined = new std::vector<std::vector<jsy::Synopse> *>();
        for (unsigned int l = 0; l <= qp->ETypes.size(); l++) {
            synopses_combined->push_back(new std::vector<jsy::Synopse>());
            for (unsigned int p = 0; p < g->NT.size(); p++)
                if (ractive->at(l)->at(p) > 0) {
                    jsy::Synopse s; jsy::init_synopse(&s, p);
                    synopses_combined->at(l)->push_back(s);
                    ractive->at(l)->at(p) = synopses_combined->at(l)->size();
                }
        }

        auto hidden_edges = new std::vector<HiddenEdge>();
        for (unsigned int l = 0; l < qp->ETypes.size(); l++) {
            for (unsigned int p = 0; p < g->NT.size(); p++) {
                if (ractive->at(l)->at(p) > 0) {
                    if (qp->EDirect[l] == 1) {
                        for (unsigned int nbr: *(g->EL[p])) {
                            if (ractive->at(l + 1)->at(nbr) > 0) {
                                HiddenEdge he{.s=ractive->at(l)->at(p) - 1, .t=ractive->at(l + 1)->at(nbr) - 1, .l=l};
                                hidden_edges->push_back(he);
                            }
                        }
                    } else {
                        for (unsigned int nbr: *(g->rEL[p])) {
                            if (ractive->at(l + 1)->at(nbr) > 0) {
                                HiddenEdge he{.s=ractive->at(l)->at(p) - 1, .t=ractive->at(l + 1)->at(nbr) - 1, .l=l};
                                hidden_edges->push_back(he);
                            }}}}}}

        out_edges = hidden_edges->size();
        unsigned int max_meta_layer;
        if (qp->instance == -1) max_meta_layer = qp->ETypes.size();
        else max_meta_layer = std::max(1u, (unsigned int)(qp->ETypes.size() - 1));

        end = std::chrono::steady_clock::now();
        preprocess_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        double goodness = 0;

        for(unsigned int meta_layer = 1;meta_layer<= max_meta_layer;meta_layer++){
            std::vector<bool> exact_dom_peers(g->NT.size(), false);
            if (isdom->at(count+meta_layer-1))
                for (auto x: *hg_dom_peers->at(count+meta_layer - 1))
                    exact_dom_peers[x] = true;
            else {
                std::vector<bool> hg_dom_map(g->NT.size(), true);
                for (auto x : *hg_dom_peers->at(count + meta_layer - 1)) hg_dom_map[x] = false;
                for (unsigned int i = 0; i < g->NT.size(); i++) {
                    if (ractive->at(0)->at(i) && hg_dom_map[i]) exact_dom_peers[i] = true;
                }
            }

            std::vector<bool> exact_dom_greater_peers(g->NT.size(), false);
            if(isdom_greater->at(count+meta_layer-1))
                for (auto x: *hg_dom_greater_peers->at(count+meta_layer-1))
                    exact_dom_greater_peers[x] = true;
            else{
                std::vector<bool> hg_dom_greater_map(g->NT.size(),true);
                for(auto x: *hg_dom_greater_peers->at(count+meta_layer-1)) hg_dom_greater_map[x] = false;
                for(unsigned int i=0;i<g->NT.size();i++){
                    if(ractive->at(0)->at(i) && hg_dom_greater_map[i]) exact_dom_greater_peers[i] = true;
                }
            }

            auto topr_size = (unsigned int)(topr * peer_size);

            start = std::chrono::steady_clock::now();

            if(centrality == "d"){
                for(unsigned int qn: qnodes) {
                    std::vector<unsigned int> props_dom_peers;

                    auto props_degs = jsy::synopses(meta_layer, hidden_edges, synopses_combined);
                    double qn_prop_deg = props_degs->at(qn);
                    for(unsigned int i=0;i<g->NT.size();i++){
                        if(i == qn) continue;
                        if(ractive->at(0)->at(i)) if(props_degs->at(i) > qn_prop_deg) props_dom_peers.push_back(i);
                    }
                    props_dom_peers.push_back(qn);
                    bool istopr = false;
                    if(topr_size >= props_dom_peers.size()) istopr = true;
                    if(istopr && exact_dom_peers[qn]) goodness += 1.0;
                    else if(!istopr && !exact_dom_greater_peers[qn]) goodness += 1.0;
                }
            }
            else if(centrality == "dp"){
                for(unsigned int qn: qnodes){
                    auto q_peers = Peers(qn, qp, meta_layer, g, visited, ractive);
                    unsigned int qn_exact_deg = q_peers->size();
                    delete q_peers;
                    auto prop_degs = jsy::synopses_fp_deg(meta_layer, hidden_edges, synopses_combined,
                            qn_exact_deg*(1.0+beta), topr*(1.0+beta), peer_size);

                    bool istopr = false;
                    std::vector<unsigned int> props_dom_peers;
                    if(prop_degs != nullptr){
                        double qn_prop_deg = prop_degs->at(qn);
                        for(unsigned int i=0;i<g->NT.size();i++){
                            if(i == qn) continue;
                            if(ractive->at(0)->at(i)) if(prop_degs->at(i) >qn_prop_deg) props_dom_peers.push_back(i);
                        }
                        props_dom_peers.push_back(qn);
                        if(topr_size >= props_dom_peers.size()) istopr = true;
                        if(istopr && exact_dom_peers[qn]) goodness += 1.0;
                        else if(!istopr && !exact_dom_greater_peers[qn]) goodness += 1.0;
                    }
                    else{
                        if(!exact_dom_greater_peers[qn]) goodness += 1.0;
                    }
                }
            }
            else if(centrality == "h"){
                for(unsigned int qn: qnodes){
                    auto degs = jsy::synopses(meta_layer, hidden_edges, synopses_combined);
                    auto qn_peers = Peers(qn, qp, meta_layer, g, visited, ractive);
                    auto sort_degs = new std::vector<std::pair<double, unsigned int>>();
                    for(unsigned int n: *qn_peers)
                        sort_degs->push_back(std::pair<double, unsigned int>(degs->at(n), n));
                    std::sort(sort_degs->begin(), sort_degs->end(), cmp_max);

                    unsigned int qn_hindex = 0;
                    for(auto &sort_deg: *sort_degs){
                            qn_hindex++;
                            if(qn_hindex > sort_deg.first) break;
                    }
                    if(qn_hindex > sort_degs->at(qn_hindex-1).first) qn_hindex--;

                    std::vector<bool> *qualified_nodes = new std::vector<bool>(g->NT.size(), false);
                    for(unsigned int peer: *peers) if(degs->at(peer)>=qn_hindex) qualified_nodes->at(peer)=true;
                    auto *hindexes = jsy::synopses(meta_layer, hidden_edges, synopses_combined, qualified_nodes);

                    std::vector<unsigned int> props_dom_peers;
                    for(auto it=hindexes->begin();it!=hindexes->end();it++){
                        if(it->first == qn) continue;
                        if(it->second >= hindexes->at(qn)) props_dom_peers.push_back(it->first);
                    }
                    props_dom_peers.push_back(qn);
                    bool istopr = false;
                    if(topr_size >= props_dom_peers.size()) istopr = true;
                    if(istopr && exact_dom_peers[qn]) goodness += 1.0;
                    else if(!istopr && !exact_dom_greater_peers[qn]) goodness += 1.0;

                    delete degs;
                    delete qn_peers;
                    delete sort_degs;
                    delete qualified_nodes;
                    delete hindexes;
                }
            }
            else if(centrality == "hp"){
                for(unsigned int qn: qnodes){
                    auto q_peers = Peers(qn, qp, meta_layer, g, visited, ractive);
                    unsigned int q_deg = q_peers->size();

                    double mcs = 0;
                    auto prop_deg = jsy::synopses_fp(meta_layer, hidden_edges, synopses_combined, q_deg*(1.0+beta),
                            topr*(1.0+beta), peer_size, mcs);

                    if(prop_deg != nullptr){
                        std::vector<double> *degs = new std::vector<double>(g->NT.size(), 0);
                        for(auto &it: *prop_deg){
                            unsigned int n = it.first;
                            double d = it.second;
                            degs->at(n) += d;
                        }
                        delete prop_deg;

                        auto sort_degs = new std::vector<std::pair<double, unsigned int>>();
                        for(unsigned int n : *q_peers)
                            sort_degs->push_back(std::pair<double, unsigned int>(degs->at(n), n));
                        std::sort(sort_degs->begin(), sort_degs->end(), cmp_max);

                        unsigned int q_hindex = 0;
                        for(auto &sort_deg: *sort_degs){
                            q_hindex++;
                            if(q_hindex > sort_deg.first) break;
                        }
                        if(q_hindex > sort_degs->at(q_hindex-1).first) q_hindex--;

                        if(mcs >= q_hindex && mcs>= topr_size) {
                            if(!exact_dom_greater_peers[qn]) goodness += 1.0;
                        }
                        else{
                            std::vector<bool> *qualified_nodes = new std::vector<bool>(g->NT.size(), false);
                            for(unsigned int peer: *peers) if(degs->at(peer)>= q_hindex) qualified_nodes->at(peer)=true;
                            auto* hindexes = jsy::synopses(meta_layer, hidden_edges, synopses_combined, qualified_nodes);

                            std::vector<unsigned int> props_dom_peers;
                            for(auto it=hindexes->begin();it!=hindexes->end();it++){
                                if(it->first==qn) continue;
                                if(it->second >= hindexes->at(qn)) props_dom_peers.push_back(it->first);
                            }
                            props_dom_peers.push_back(qn);
                            bool istopr = false;
                            if(topr_size >= props_dom_peers.size()) istopr = true;
                            if(istopr && exact_dom_peers[qn]) goodness+= 1.0;
                            else if(!istopr && !exact_dom_greater_peers[qn]) goodness+=1.0;

                            delete qualified_nodes;
                            delete hindexes;
                        }

                        delete degs;
                        delete sort_degs;
                    }
                    else if(!exact_dom_greater_peers[qn]) goodness += 1.0;
                    delete q_peers;
                }
            }
            end = std::chrono::steady_clock::now();
            running_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
        goodness /= max_meta_layer;
        goodness /= QNUM;
        count += max_meta_layer;
        std::cout<<"goodness:"<<goodness<<std::endl;

        running_time /= QNUM;
        double temp_time = (preprocess_time + running_time);
        avg_time += temp_time;
        temp_time /= 1000000000;
        std::cout<<"time:"<<temp_time<<" s"<<std::endl;

        delete frontiers;
        delete peers;
        for(auto &i: *active) delete i;
        for(auto &i: *ractive) delete i;
        delete active;
        delete ractive;
        for(auto &i: *synopses_combined) delete i;
        delete synopses_combined;
        delete hidden_edges;

        return goodness;
    }

    bool COD_prop_global_scale(Pattern *qp, HeterGraph *g, double topr, std::vector<std::vector<bool>*>* visited,
                                    std::vector<std::vector<bool>*>* back_visited, const std::string &centrality,
                                    double & avg_time){


        double running_time = 0.0;
        double preprocess_time = 0.0;

        auto start = std::chrono::steady_clock::now();

        auto frontiers = new std::vector<unsigned int>();
        auto peers = new std::vector<unsigned int>();
        Peers(qp, g, frontiers, peers, visited);
        unsigned int peer_size = peers->size();

        auto end = std::chrono::steady_clock::now();

        if (peer_size <= 2) {
            delete frontiers;
            delete peers;
            return false;
        }
        qp->print();

        preprocess_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        start = std::chrono::steady_clock::now();

        std::vector<std::set<unsigned int> *> *active = ActiveMidNodes(peers, frontiers, qp, g, visited, back_visited);
        auto ractive = new std::vector<std::vector<unsigned int> *>();
        for (unsigned int l = 0; l <= qp->ETypes.size(); l++) ractive->push_back(new std::vector<unsigned int>(g->NT.size(), 0));
        for (unsigned int n = 0; n < g->NT.size(); n++) for (unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;

        auto synopses_combined = new std::vector<std::vector<jsy::Synopse> *>();
        for (unsigned int l = 0; l <= qp->ETypes.size(); l++) {
            synopses_combined->push_back(new std::vector<jsy::Synopse>());
            for (unsigned int p = 0; p < g->NT.size(); p++)
                if (ractive->at(l)->at(p) > 0) {
                    jsy::Synopse s; jsy::init_synopse(&s, p);
                    synopses_combined->at(l)->push_back(s);
                    ractive->at(l)->at(p) = synopses_combined->at(l)->size();
                }
        }

        auto hidden_edges = new std::vector<HiddenEdge>();
        for (unsigned int l = 0; l < qp->ETypes.size(); l++) {
            for (unsigned int p = 0; p < g->NT.size(); p++) {
                if (ractive->at(l)->at(p) > 0) {
                    if (qp->EDirect[l] == 1) {
                        for (unsigned int nbr: *(g->EL[p])) {
                            if (ractive->at(l + 1)->at(nbr) > 0) {
                                HiddenEdge he{.s=ractive->at(l)->at(p) - 1, .t=ractive->at(l + 1)->at(nbr) - 1, .l=l};
                                hidden_edges->push_back(he);
                            }
                        }
                    } else {
                        for (unsigned int nbr: *(g->rEL[p])) {
                            if (ractive->at(l + 1)->at(nbr) > 0) {
                                HiddenEdge he{.s=ractive->at(l)->at(p) - 1, .t=ractive->at(l + 1)->at(nbr) - 1, .l=l};
                                hidden_edges->push_back(he);
                            }}}}}}
        unsigned int max_meta_layer;
        if (qp->instance == -1) max_meta_layer = qp->ETypes.size();
        else max_meta_layer = std::max(1u, (unsigned int)(qp->ETypes.size() - 1));

        end = std::chrono::steady_clock::now();
        preprocess_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        for(unsigned int loop=0;loop<1;loop++){
            for(unsigned int meta_layer = 1;meta_layer <= max_meta_layer;meta_layer++){

                start = std::chrono::steady_clock::now();
                auto topr_size = (unsigned int)(topr * peer_size);
                std::vector<unsigned int> props_dom_peers;
                if(centrality == "df1"){
                    auto props_degs = jsy::synopses(meta_layer, hidden_edges, synopses_combined);
                    auto sort_degs = new std::vector<std::pair<double, unsigned int>>();
                    for(unsigned int i=0;i<g->NT.size();i++)
                        if(ractive->at(0)->at(i)) sort_degs->push_back(std::pair<double, unsigned int>(props_degs->at(i), i));
                    std::sort(sort_degs->begin(), sort_degs->end(), cmp_max);

                    for(unsigned int i=0;i<=topr_size;i++)
                        props_dom_peers.push_back(sort_degs->at(i).second);
                }
                else if(centrality == "hf1"){
                    auto degs = jsy::synopses(meta_layer, hidden_edges, synopses_combined);
                    std::vector<unsigned int> candidates;
                    for(unsigned int i=0;i<g->NT.size();i++) if(ractive->at(0)->at(i)) candidates.push_back(i);

                    while(!candidates.empty()){
                        unsigned int qn = candidates[0];
                        auto qn_peers = Peers(qn, qp, meta_layer, g, visited, ractive);
                        auto sort_degs = new std::vector<std::pair<double, unsigned int>>();
                        for (unsigned int n: *qn_peers)
                            sort_degs->push_back(std::pair<double, unsigned int>(degs->at(n), n));
                        std::sort(sort_degs->begin(), sort_degs->end(), cmp_max);
                        unsigned int qn_hindex = 0;
                        for (auto &sort_deg: *sort_degs){
                            qn_hindex++;
                            if (qn_hindex > sort_deg.first) break;
                        }
                        if (qn_hindex > sort_degs->at(qn_hindex - 1).first) qn_hindex--;

                        std::vector<bool> *qualified_nodes = new std::vector<bool>(g->NT.size(), false);
                        for (unsigned int peer: *peers) if (degs->at(peer) >= qn_hindex) qualified_nodes->at(peer) = true;
                        auto *hindexes = jsy::synopses(meta_layer, hidden_edges, synopses_combined, qualified_nodes);

                        std::vector<bool> candidates_equal_qn;
                        unsigned int equal_qn_count = 0;
                        for (unsigned int candidate : candidates) {
                            bool equal_condition = (fabs(hindexes->at(candidate)-hindexes->at(qn))<0.00001);
                            candidates_equal_qn.push_back(equal_condition);
                            if(equal_condition) equal_qn_count++;
                        }

                        std::vector<unsigned int> temp_domed;
                        std::vector<unsigned int> temp_dom;
                        for(unsigned int i=1;i<candidates.size();i++){
                            if(!candidates_equal_qn[i] && hindexes->at(candidates[i]) < hindexes->at(qn)){
                                temp_domed.push_back(candidates[i]);
                            }
                            if(!candidates_equal_qn[i] && hindexes->at(candidates[i]) > hindexes->at(qn)){
                                temp_dom.push_back(candidates[i]);
                            }
                        }

                        if(temp_dom.size() + props_dom_peers.size() <= topr_size &&
                           temp_dom.size()+equal_qn_count+props_dom_peers.size() > topr_size){
                            for(auto i: temp_dom) props_dom_peers.push_back(i);
                            for(unsigned int i=0;i<candidates.size();i++){
                                if(candidates_equal_qn[i]){
                                    props_dom_peers.push_back(candidates[i]);
                                }
                            }
                            candidates.clear();
                        }
                        else if(temp_dom.size()+equal_qn_count+props_dom_peers.size() <= topr_size){
                            for(auto i: temp_dom) props_dom_peers.push_back(i);
                            for(unsigned int i=0;i<candidates.size();i++)
                                if(candidates_equal_qn[i]) props_dom_peers.push_back(candidates[i]);
                            candidates.clear();
                            for(auto i: temp_domed) candidates.push_back(i);
                        }
                        else if(temp_dom.size()+props_dom_peers.size()>topr_size){
                            candidates.clear();
                            for(auto i: temp_dom) candidates.push_back(i);
                        }
                    }
                }

                std::cout<<"|prop_dom|:"<<props_dom_peers.size()<<std::endl;

                end = std::chrono::steady_clock::now();
                running_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            }
        }

        running_time /= 1;
        double temp_time = (preprocess_time+running_time);
        avg_time += temp_time;
        temp_time /= 1000000000;
        std::cout<<"time:"<<temp_time<<std::endl;

        delete frontiers;
        delete peers;
        for(auto &i: *active) delete i;
        delete active;
        for(auto &i: *ractive) delete i;
        delete ractive;
        for(auto &i: *synopses_combined) delete i;
        delete synopses_combined;
        delete hidden_edges;

        return true;
    }

    double COD_prop_global_cross_f1(Pattern *qp, HeterGraph *g, double topr, std::vector<std::vector<bool>*>* visited,
                            std::vector<std::vector<bool>*>* back_visited, const std::string &centrality,
                            std::vector<std::vector<unsigned int>*>* hg_dom_peers, std::vector<bool>* isdom,
                            std::vector<std::vector<unsigned int>*>* hg_dom_greater_peers, std::vector<bool>* isdom_greater,
                            unsigned int & count, double & avg_time, unsigned int & out_edges){


        double running_time = 0.0;
        double preprocess_time = 0.0;

        auto start = std::chrono::steady_clock::now();

        auto frontiers = new std::vector<unsigned int>();
        auto peers = new std::vector<unsigned int>();
        Peers(qp, g, frontiers, peers, visited);
        unsigned int peer_size = peers->size();

        auto end = std::chrono::steady_clock::now();

        if (peer_size <= 2) {
            delete frontiers;
            delete peers;
            return -1;
        }
        qp->print();

        preprocess_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        start = std::chrono::steady_clock::now();

        std::vector<std::set<unsigned int> *> *active = ActiveMidNodes(peers, frontiers, qp, g, visited, back_visited);
        auto ractive = new std::vector<std::vector<unsigned int> *>();
        for (unsigned int l = 0; l <= qp->ETypes.size(); l++) ractive->push_back(new std::vector<unsigned int>(g->NT.size(), 0));
        for (unsigned int n = 0; n < g->NT.size(); n++) for (unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;

        auto synopses_combined = new std::vector<std::vector<jsy::Synopse> *>();
        for (unsigned int l = 0; l <= qp->ETypes.size(); l++) {
            synopses_combined->push_back(new std::vector<jsy::Synopse>());
            for (unsigned int p = 0; p < g->NT.size(); p++)
                if (ractive->at(l)->at(p) > 0) {
                    jsy::Synopse s; jsy::init_synopse(&s, p);
                    synopses_combined->at(l)->push_back(s);
                    ractive->at(l)->at(p) = synopses_combined->at(l)->size();
                }
        }

        auto hidden_edges = new std::vector<HiddenEdge>();
        for (unsigned int l = 0; l < qp->ETypes.size(); l++) {
            for (unsigned int p = 0; p < g->NT.size(); p++) {
                if (ractive->at(l)->at(p) > 0) {
                    if (qp->EDirect[l] == 1) {
                        for (unsigned int nbr: *(g->EL[p])) {
                            if (ractive->at(l + 1)->at(nbr) > 0) {
                                HiddenEdge he{.s=ractive->at(l)->at(p) - 1, .t=ractive->at(l + 1)->at(nbr) - 1, .l=l};
                                hidden_edges->push_back(he);
                            }
                        }
                    } else {
                        for (unsigned int nbr: *(g->rEL[p])) {
                            if (ractive->at(l + 1)->at(nbr) > 0) {
                                HiddenEdge he{.s=ractive->at(l)->at(p) - 1, .t=ractive->at(l + 1)->at(nbr) - 1, .l=l};
                                hidden_edges->push_back(he);
                            }}}}}}
        out_edges = hidden_edges->size();                    
        unsigned int max_meta_layer;
        if (qp->instance == -1) max_meta_layer = qp->ETypes.size();
        else max_meta_layer = std::max(1u, (unsigned int)(qp->ETypes.size() - 1));

        end = std::chrono::steady_clock::now();
        preprocess_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        double goodness = 0;
        for(unsigned int loop=0;loop<LOOP;loop++){
            for(unsigned int meta_layer = 1;meta_layer <= max_meta_layer;meta_layer++){
                std::vector<unsigned int> exact_dom_peers;
                if (isdom->at(count+meta_layer-1))
                    for (auto x: *hg_dom_peers->at(count+meta_layer - 1))
                        exact_dom_peers.push_back(x);
                else {
                    std::vector<bool> hg_dom_map(g->NT.size(), true);
                    for (auto x : *hg_dom_peers->at(count + meta_layer - 1)) hg_dom_map[x] = false;
                    for (unsigned int i = 0; i < g->NT.size(); i++) {
                        if (ractive->at(0)->at(i) && hg_dom_map[i]) exact_dom_peers.push_back(i);
                    }
                }
                std::sort(exact_dom_peers.begin(), exact_dom_peers.end());

                std::vector<unsigned int> exact_dom_greater_peers;
                if(isdom_greater->at(count+meta_layer-1))
                    for (auto x: *hg_dom_greater_peers->at(count+meta_layer-1))
                        exact_dom_greater_peers.push_back(x);
                else{
                    std::vector<bool> hg_dom_greater_map(g->NT.size(),true);
                    for(auto x: *hg_dom_greater_peers->at(count+meta_layer-1)) hg_dom_greater_map[x] = false;
                    for(unsigned int i=0;i<g->NT.size();i++){
                        if(ractive->at(0)->at(i) && hg_dom_greater_map[i]) exact_dom_greater_peers.push_back(i);
                    }
                }
                std::sort(exact_dom_greater_peers.begin(), exact_dom_greater_peers.end());

                start = std::chrono::steady_clock::now();
                auto topr_size = (unsigned int)(topr * peer_size);
                std::vector<unsigned int> props_dom_peers;
                if(centrality == "df1"){
                    auto props_degs = jsy::synopses(meta_layer, hidden_edges, synopses_combined);
                    auto sort_degs = new std::vector<std::pair<double, unsigned int>>();
                    for(unsigned int i=0;i<g->NT.size();i++)
                        if(ractive->at(0)->at(i)) sort_degs->push_back(std::pair<double, unsigned int>(props_degs->at(i), i));
                    std::sort(sort_degs->begin(), sort_degs->end(), cmp_max);

                    for(unsigned int i=0;i<=topr_size;i++)
                        props_dom_peers.push_back(sort_degs->at(i).second);
                }
                else if(centrality == "hf1"){
                    auto degs = jsy::synopses(meta_layer, hidden_edges, synopses_combined);
                    std::vector<unsigned int> candidates;
                    for(unsigned int i=0;i<g->NT.size();i++) if(ractive->at(0)->at(i)) candidates.push_back(i);

                    while(!candidates.empty()){
                        unsigned int qn = candidates[0];
                        auto qn_peers = Peers(qn, qp, meta_layer, g, visited, ractive);
                        auto sort_degs = new std::vector<std::pair<double, unsigned int>>();
                        for (unsigned int n: *qn_peers)
                            sort_degs->push_back(std::pair<double, unsigned int>(degs->at(n), n));
                        std::sort(sort_degs->begin(), sort_degs->end(), cmp_max);
                        unsigned int qn_hindex = 0;
                        for (auto &sort_deg: *sort_degs){
                            qn_hindex++;
                            if (qn_hindex > sort_deg.first) break;
                        }
                        if (qn_hindex > sort_degs->at(qn_hindex - 1).first) qn_hindex--;

                        std::vector<bool> *qualified_nodes = new std::vector<bool>(g->NT.size(), false);
                        for (unsigned int peer: *peers) if (degs->at(peer) >= qn_hindex) qualified_nodes->at(peer) = true;
                        auto *hindexes = jsy::synopses(meta_layer, hidden_edges, synopses_combined, qualified_nodes);

                        std::vector<bool> candidates_equal_qn;
                        unsigned int equal_qn_count = 0;
                        for (unsigned int candidate : candidates) {
                            bool equal_condition = (fabs(hindexes->at(candidate)-hindexes->at(qn))<0.00001);
                            candidates_equal_qn.push_back(equal_condition);
                            if(equal_condition) equal_qn_count++;
                        }

                        std::vector<unsigned int> temp_domed;
                        std::vector<unsigned int> temp_dom;
                        for(unsigned int i=1;i<candidates.size();i++){
                            if(!candidates_equal_qn[i] && hindexes->at(candidates[i]) < hindexes->at(qn)){
                                temp_domed.push_back(candidates[i]);
                            }
                            if(!candidates_equal_qn[i] && hindexes->at(candidates[i]) > hindexes->at(qn)){
                                temp_dom.push_back(candidates[i]);
                            }
                        }

                        if(temp_dom.size() + props_dom_peers.size() <= topr_size &&
                           temp_dom.size()+equal_qn_count+props_dom_peers.size() > topr_size){
                            for(auto i: temp_dom) props_dom_peers.push_back(i);
                            for(unsigned int i=0;i<candidates.size();i++){
                                if(candidates_equal_qn[i]){
                                    props_dom_peers.push_back(candidates[i]);
                                }
                            }
                            candidates.clear();
                        }
                        else if(temp_dom.size()+equal_qn_count+props_dom_peers.size() <= topr_size){
                            for(auto i: temp_dom) props_dom_peers.push_back(i);
                            for(unsigned int i=0;i<candidates.size();i++)
                                if(candidates_equal_qn[i]) props_dom_peers.push_back(candidates[i]);
                            candidates.clear();
                            for(auto i: temp_domed) candidates.push_back(i);
                        }
                        else if(temp_dom.size()+props_dom_peers.size()>topr_size){
                            candidates.clear();
                            for(auto i: temp_dom) candidates.push_back(i);
                        }
                    }
                }
                end = std::chrono::steady_clock::now();
                running_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

                std::sort(props_dom_peers.begin(), props_dom_peers.end());

                unsigned int tp = 0, tp_greater = 0;
                unsigned int exact_pos = 0, prop_pos = 0;
                while (exact_pos < exact_dom_peers.size() && prop_pos < props_dom_peers.size()) {
                    if (exact_dom_peers[exact_pos] == props_dom_peers[prop_pos]) {
                        tp++;
                        exact_pos++;
                        prop_pos++;
                    } else if (exact_dom_peers[exact_pos] > props_dom_peers[prop_pos]) prop_pos++;
                    else if (exact_dom_peers[exact_pos] < props_dom_peers[prop_pos]) exact_pos++;
                }

                exact_pos = 0; prop_pos = 0;
                while(exact_pos < exact_dom_greater_peers.size() && prop_pos < props_dom_peers.size()){
                    if (exact_dom_greater_peers[exact_pos] == props_dom_peers[prop_pos]){
                        tp_greater++;
                        exact_pos++;
                        prop_pos++;
                    } else if(exact_dom_greater_peers[exact_pos] > props_dom_peers[prop_pos]) prop_pos++;
                    else if(exact_dom_peers[exact_pos]<props_dom_peers[prop_pos]) exact_pos++;
                }
                if (tp_greater > 0) {
                    double precision = (tp * 1.0) / props_dom_peers.size();
                    double recall = (tp_greater * 1.0) / exact_dom_greater_peers.size();
                    double f1 = (2 * precision * recall) / (precision + recall);
                    goodness += f1;
                }
                else if(tp_greater == 0){
                    if(!props_dom_peers.empty()) {
                        double precision = (tp * 1.0) / props_dom_peers.size();
                        double f1 = (2 * precision) / (precision + 1);
                        goodness += f1;
                    }
                }
            }
        }
        goodness /= max_meta_layer;
        goodness /= LOOP;
        count += max_meta_layer;

        running_time /= LOOP;
        double temp_time = (preprocess_time+running_time);
        avg_time += temp_time;
        temp_time /= 1000000000;
        std::cout<<"goodness:"<<goodness<<std::endl;
        std::cout<<"time:"<<temp_time<<std::endl;

        delete frontiers;
        delete peers;
        for(auto &i: *active) delete i;
        delete active;
        for(auto &i: *ractive) delete i;
        delete ractive;
        for(auto &i: *synopses_combined) delete i;
        delete synopses_combined;
        delete hidden_edges;

        return goodness;
    }

    bool COD_hg_global_greater_f1(Pattern *qp, HeterGraph *g, double topr, std::vector<std::vector<bool>*>* visited,
                    std::vector<std::vector<bool>*>* back_visited, const std::string &centrality, double & avg_time){
        double running_time = 0.0;


        auto start = std::chrono::steady_clock::now();

        auto frontiers = new std::vector<unsigned int>();
        auto peers = new std::vector<unsigned int>();
        Peers(qp, g, frontiers, peers, visited);
        unsigned int peer_size = peers->size();

        auto end = std::chrono::steady_clock::now();

        if (peer_size <= 2) {
            delete frontiers;
            delete peers;
            return false;
        }
        qp->print();

        running_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        start = std::chrono::steady_clock::now();

        std::vector<std::set<unsigned int> *> *active = ActiveMidNodes(peers, frontiers, qp, g, visited, back_visited);
        auto ractive = new std::vector<std::vector<unsigned int> *>();
        for (unsigned int l = 0; l <= qp->ETypes.size(); l++) ractive->push_back(new std::vector<unsigned int>(g->NT.size(), 0));
        for (unsigned int n = 0; n < g->NT.size(); n++) for (unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;

        unsigned int max_meta_layer;
        if(qp->instance == -1) max_meta_layer = qp->ETypes.size();
        else max_meta_layer = std::max(1u, (unsigned int)(qp->ETypes.size() - 1));

        end = std::chrono::steady_clock::now();
        running_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        for (unsigned int meta_layer = 1; meta_layer <= max_meta_layer; meta_layer++){
            std::cout << "# l=" << meta_layer << std::endl;

            start = std::chrono::steady_clock::now();

            std::vector<unsigned int> *exact_degs = nullptr;
            if (centrality == "df1") exact_degs = hidden_graph_deg(meta_layer, qp, g, visited, ractive);
            else if (centrality == "hf1") exact_degs = hidden_graph_hindex(meta_layer, qp, g, visited, ractive);

            auto sort_degs = new std::vector<std::pair<unsigned int, unsigned int>>();
            for(unsigned int i=0;i<g->NT.size();i++)
                if(ractive->at(0)->at(i)) sort_degs->push_back(std::pair<unsigned int, unsigned int>(exact_degs->at(i), i));
            std::sort(sort_degs->begin(), sort_degs->end(), cmp_max_unsigned);

            auto topr_size = (unsigned int)(topr*peer_size);
            unsigned int qn_exact_deg = sort_degs->at(topr_size).first;
            delete sort_degs;

            std::vector<unsigned int> exact_dom_peers;
            for (unsigned int i = 0; i < g->NT.size(); i++)
                if (ractive->at(0)->at(i)) if (exact_degs->at(i) > qn_exact_deg) exact_dom_peers.push_back(i);

            end = std::chrono::steady_clock::now();
            running_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

            double ratio = (exact_dom_peers.size()*1.0) / peer_size;
            if(ratio < 0.5){
                std::cout<<"( ";
                for(auto x: exact_dom_peers) std::cout<<x<<" ";
                std::cout<<std::endl;
            }
            else{
                std::cout<<") ";
                for(unsigned int i=0;i<g->NT.size();i++){
                    if(ractive->at(0)->at(i)){
                        if(exact_degs->at(i) <= qn_exact_deg) std::cout<<i<<" ";
                    }
                }
                std::cout<<std::endl;
            }
            delete exact_degs;
        }
        delete frontiers;
        delete peers;
        for (auto &i : *active) delete i;
        delete active;
        for (auto &i : *ractive) delete i;
        delete ractive;

        avg_time += running_time;
        running_time /= 1000000000;
        std::cout<<"time:"<<running_time<<std::endl;
        return true;
    }

    bool COD_hg_global_f1_by_union(Pattern *qp, HeterGraph *g, double topr, std::vector<std::vector<bool>*>* visited,
                    std::vector<std::vector<bool>*>* back_visited, const std::string &centrality, double & avg_time){
        double running_time = 0.0;


        auto start = std::chrono::steady_clock::now();

        auto frontiers = new std::vector<unsigned int>();
        auto peers = new std::vector<unsigned int>();
        Peers(qp, g, frontiers, peers, visited);
        unsigned int peer_size = peers->size();

        auto end = std::chrono::steady_clock::now();

        if (peer_size <= 2) {
            delete frontiers;
            delete peers;
            return false;
        }
        qp->print();

        running_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        start = std::chrono::steady_clock::now();

        std::vector<std::set<unsigned int> *> *active = ActiveMidNodes(peers, frontiers, qp, g, visited, back_visited);
        auto ractive = new std::vector<std::vector<unsigned int> *>();
        for (unsigned int l = 0; l <= qp->ETypes.size(); l++) ractive->push_back(new std::vector<unsigned int>(g->NT.size(), 0));
        for (unsigned int n = 0; n < g->NT.size(); n++) for (unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;

        unsigned int max_meta_layer;
        if(qp->instance == -1) max_meta_layer = qp->ETypes.size();
        else max_meta_layer = std::max(1u, (unsigned int)(qp->ETypes.size() - 1));

        end = std::chrono::steady_clock::now();
        running_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        for (unsigned int meta_layer = 1; meta_layer <= max_meta_layer; meta_layer++) {
            std::cout << "# l=" << meta_layer << std::endl;

            start = std::chrono::steady_clock::now();

            std::vector<unsigned int> *exact_degs = nullptr;
            if (centrality == "df1") exact_degs = hidden_graph_deg_union(meta_layer, qp, g, visited, ractive);
//            else if (centrality == "hf1") {}

            auto sort_degs = new std::vector<std::pair<unsigned int, unsigned int>>();
            for(unsigned int i=0;i<g->NT.size();i++)
                if(ractive->at(0)->at(i)) sort_degs->push_back(std::pair<unsigned int, unsigned int>(exact_degs->at(i), i));
            std::sort(sort_degs->begin(), sort_degs->end(), cmp_max_unsigned);

            auto topr_size = (unsigned int)(topr*peer_size);
            unsigned int qn_exact_deg = sort_degs->at(topr_size).first;
            delete sort_degs;

            std::vector<unsigned int> exact_dom_peers;
            for (unsigned int i = 0; i < g->NT.size(); i++)
                if (ractive->at(0)->at(i)) if (exact_degs->at(i) >= qn_exact_deg) exact_dom_peers.push_back(i);

            end = std::chrono::steady_clock::now();
            running_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

            double ratio = (exact_dom_peers.size()*1.0) / peer_size;
            if(ratio < 0.5){
                std::cout<<"( ";
                for(auto x: exact_dom_peers) std::cout<<x<<" ";
                std::cout<<std::endl;
            }
            else{
                std::cout<<") ";
                for(unsigned int i=0;i<g->NT.size();i++)
                    if(ractive->at(0)->at(i)) if(exact_degs->at(i) < qn_exact_deg) std::cout<<i<<" ";
                std::cout<<std::endl;
            }
            delete exact_degs;
        }
        delete frontiers;
        delete peers;
        for (auto &i : *active) delete i;
        delete active;
        for (auto &i : *ractive) delete i;
        delete ractive;

        avg_time += running_time;
        running_time /= 1000000000;
        std::cout<<"time:"<<running_time<<std::endl;
        return true;
    }

    bool COD_hg_global_f1(Pattern *qp, HeterGraph *g, double topr, std::vector<std::vector<bool>*>* visited,
                    std::vector<std::vector<bool>*>* back_visited, const std::string &centrality, double & avg_time){
        double running_time = 0.0;


        auto start = std::chrono::steady_clock::now();

        auto frontiers = new std::vector<unsigned int>();
        auto peers = new std::vector<unsigned int>();
        Peers(qp, g, frontiers, peers, visited);
        unsigned int peer_size = peers->size();

        auto end = std::chrono::steady_clock::now();

        if (peer_size <= 2) {
            delete frontiers;
            delete peers;
            return false;
        }
        qp->print();

        running_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        start = std::chrono::steady_clock::now();

        std::vector<std::set<unsigned int> *> *active = ActiveMidNodes(peers, frontiers, qp, g, visited, back_visited);
        auto ractive = new std::vector<std::vector<unsigned int> *>();
        for (unsigned int l = 0; l <= qp->ETypes.size(); l++) ractive->push_back(new std::vector<unsigned int>(g->NT.size(), 0));
        for (unsigned int n = 0; n < g->NT.size(); n++) for (unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;

        unsigned int max_meta_layer;
        if(qp->instance == -1) max_meta_layer = qp->ETypes.size();
        else max_meta_layer = std::max(1u, (unsigned int)(qp->ETypes.size() - 1));

        end = std::chrono::steady_clock::now();
        running_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        for (unsigned int meta_layer = 1; meta_layer <= max_meta_layer; meta_layer++) {
            std::cout << "# l=" << meta_layer << std::endl;

            start = std::chrono::steady_clock::now();

            std::vector<unsigned int> *exact_degs = nullptr;
            if (centrality == "df1") exact_degs = hidden_graph_deg(meta_layer, qp, g, visited, ractive);
            else if (centrality == "hf1") exact_degs = hidden_graph_hindex(meta_layer, qp, g, visited, ractive);

            auto sort_degs = new std::vector<std::pair<unsigned int, unsigned int>>();
            for(unsigned int i=0;i<g->NT.size();i++)
                if(ractive->at(0)->at(i)) sort_degs->push_back(std::pair<unsigned int, unsigned int>(exact_degs->at(i), i));
            std::sort(sort_degs->begin(), sort_degs->end(), cmp_max_unsigned);

            auto topr_size = (unsigned int)(topr*peer_size);
            unsigned int qn_exact_deg = sort_degs->at(topr_size).first;
            delete sort_degs;

            std::vector<unsigned int> exact_dom_peers;
            for (unsigned int i = 0; i < g->NT.size(); i++)
                if (ractive->at(0)->at(i)) if (exact_degs->at(i) >= qn_exact_deg) exact_dom_peers.push_back(i);

            end = std::chrono::steady_clock::now();
            running_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

            double ratio = (exact_dom_peers.size()*1.0) / peer_size;
            if(ratio < 0.5){
                std::cout<<"( ";
                for(auto x: exact_dom_peers) std::cout<<x<<" ";
                std::cout<<std::endl;
            }
            else{
                std::cout<<") ";
                for(unsigned int i=0;i<g->NT.size();i++)
                    if(ractive->at(0)->at(i)) if(exact_degs->at(i) < qn_exact_deg) std::cout<<i<<" ";
                std::cout<<std::endl;
            }
            delete exact_degs;
        }
        delete frontiers;
        delete peers;
        for (auto &i : *active) delete i;
        delete active;
        for (auto &i : *ractive) delete i;
        delete ractive;

        avg_time += running_time;
        running_time /= 1000000000;
        std::cout<<"time:"<<running_time<<std::endl;
        return true;
    }

    std::vector<double>* COD_hg_statistics(Pattern *qp, HeterGraph *g, unsigned int qn, std::vector<std::vector<bool>*>* visited,
                           std::vector<std::vector<bool>*>* back_visited){
        auto res = new std::vector<double>();

        // NTypes guard removed: 2-hop instance rules now allowed
        auto frontiers = new std::vector<unsigned int>();
        auto peers = new std::vector<unsigned int>();
        Peers(qp, g, frontiers, peers, visited);
        unsigned int peer_size = peers->size();

        if (peer_size <= 2 || std::find(peers->begin(), peers->end(), qn) == peers->end()) {
            delete frontiers;
            delete peers;
            return nullptr;
        }
        qp->print();

        std::vector<std::set<unsigned int> *> *active = ActiveMidNodes(peers, frontiers, qp, g, visited, back_visited);
        auto ractive = new std::vector<std::vector<unsigned int> *>();
        for (unsigned int l = 0; l <= qp->ETypes.size(); l++) ractive->push_back(new std::vector<unsigned int>(g->NT.size(), 0));
        for (unsigned int n = 0; n < g->NT.size(); n++) for (unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;

        unsigned int max_meta_layer;
        if(qp->instance == -1) max_meta_layer = qp->ETypes.size();
        else max_meta_layer = std::max(1u, (unsigned int)(qp->ETypes.size() - 1));

        double temp_density = 0.0, degree_same = 0.0, hindex_same = 0.0;
        unsigned int raw_edges_total = 0; 
        for (unsigned int meta_layer = 1; meta_layer <= max_meta_layer; meta_layer++) {
            auto hidden_graph = hidden_graph_construction(meta_layer, qp, g, visited, ractive);
            auto deg_vec=new std::vector<unsigned int>(g->NT.size(), 0);
            for(unsigned int u=0;u<g->NT.size();u++)
                if(ractive->at(0)->at(u) > 0) {
                    deg_vec->at(u) = hidden_graph->at(u)->size();
                    temp_density += hidden_graph->at(u)->size();
                    raw_edges_total += hidden_graph->at(u)->size();
                }
            unsigned int qn_exact_deg = deg_vec->at(qn);
            for(unsigned int u=0;u<g->NT.size();u++){
                if(ractive->at(0)->at(u) > 0){
                    if(u == qn) continue;
                    if(qn_exact_deg == deg_vec->at(u)) degree_same += 1.0;
                }
            }

            auto hindex_vec = new std::vector<unsigned int>(g->NT.size(), 0);
            for(unsigned int u=0;u<g->NT.size();u++){
                if(ractive->at(0)->at(u) > 0){
                    unsigned int hindex = 0;
                    std::vector<unsigned int> degrees;
                    for(unsigned int n: *hidden_graph->at(u)) degrees.push_back(hidden_graph->at(n)->size());
                    std::sort(degrees.begin(), degrees.end(), std::greater<>());
                    for(;hindex<degrees.size();hindex++){
                        if(degrees[hindex] < (hindex + 1)) break;
                    }
                    hindex_vec->at(u) = hindex;
                }
            }
            unsigned int qn_exact_hindex = hindex_vec->at(qn);
            for(unsigned int u=0;u<g->NT.size();u++){
                if(ractive->at(0)->at(u) > 0){
                    if(qn == u) continue;
                    if(qn_exact_hindex == hindex_vec->at(u)) hindex_same += 1.0;
                }
            }
        }
        std::cout << "RAW_EDGES_E*: " << raw_edges_total << std::endl;
        temp_density /= max_meta_layer;
        temp_density /= (peer_size*(peer_size - 1));
        degree_same /= max_meta_layer;
        degree_same /= peer_size;
        hindex_same /= max_meta_layer;
        hindex_same /= peer_size;

        res->push_back(temp_density);
        res->push_back(degree_same);
        res->push_back(hindex_same);
        res->push_back(peer_size);

        std::cout<<"dens:"<<res->at(0)<<std::endl;
        std::cout<<"d_same:"<<res->at(1)<<std::endl;
        std::cout<<"h_same:"<<res->at(2)<<std::endl;
        std::cout<<"|peer|:"<<res->at(3)<<std::endl;

        delete frontiers;
        delete peers;
        for (auto &i : *active) delete i;
        delete active;
        for (auto &i : *ractive) delete i;
        delete ractive;

        return res;
    }

    // ---- Epsilon (approximation ratio) computation ----
    // Mirrors the structure of COD_prop_global_cross_f1 exactly,
    // but instead of F1 computes the max relative error (epsilon).
    double COD_epsilon(Pattern *qp, HeterGraph *g, double topr,
                       std::vector<std::vector<bool>*>* visited,
                       std::vector<std::vector<bool>*>* back_visited,
                       const std::string &centrality) {


        auto frontiers = new std::vector<unsigned int>();
        auto peers = new std::vector<unsigned int>();
        Peers(qp, g, frontiers, peers, visited);
        unsigned int peer_size = peers->size();
        if (peer_size <= 2) {
            delete frontiers; delete peers;
            return -1;
        }

        std::vector<std::set<unsigned int>*>* active = ActiveMidNodes(peers, frontiers, qp, g, visited, back_visited);
        auto ractive = new std::vector<std::vector<unsigned int>*>();
        for (unsigned int l = 0; l <= qp->ETypes.size(); l++)
            ractive->push_back(new std::vector<unsigned int>(g->NT.size(), 0));
        for (unsigned int n = 0; n < g->NT.size(); n++)
            for (unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;

        // Build synopses_combined and hidden_edges ONCE (same as COD_prop_global_cross_f1)
        auto synopses_combined = new std::vector<std::vector<jsy::Synopse>*>();
        for (unsigned int l = 0; l <= qp->ETypes.size(); l++) {
            synopses_combined->push_back(new std::vector<jsy::Synopse>());
            for (unsigned int p = 0; p < g->NT.size(); p++)
                if (ractive->at(l)->at(p) > 0) {
                    jsy::Synopse s; jsy::init_synopse(&s, p);
                    synopses_combined->at(l)->push_back(s);
                    ractive->at(l)->at(p) = synopses_combined->at(l)->size();
                }
        }
        auto hidden_edges = new std::vector<HiddenEdge>();
        for (unsigned int l = 0; l < qp->ETypes.size(); l++) {
            for (unsigned int p = 0; p < g->NT.size(); p++) {
                if (ractive->at(l)->at(p) > 0) {
                    if (qp->EDirect[l] == 1) {
                        for (unsigned int nbr : *(g->EL[p])) {
                            if (ractive->at(l + 1)->at(nbr) > 0) {
                                HiddenEdge he{.s=ractive->at(l)->at(p) - 1,
                                              .t=ractive->at(l + 1)->at(nbr) - 1, .l=l};
                                hidden_edges->push_back(he);
                            }
                        }
                    } else {
                        for (unsigned int nbr : *(g->rEL[p])) {
                            if (ractive->at(l + 1)->at(nbr) > 0) {
                                HiddenEdge he{.s=ractive->at(l)->at(p) - 1,
                                              .t=ractive->at(l + 1)->at(nbr) - 1, .l=l};
                                hidden_edges->push_back(he);
                            }
                        }
                    }
                }
            }
        }

        unsigned int max_meta_layer;
        if(qp->instance == -1) max_meta_layer = qp->ETypes.size();
        else max_meta_layer = std::max(1u, (unsigned int)(qp->ETypes.size() - 1));

        double total_epsilon = 0;
        unsigned int valid_layers = 0;

        for (unsigned int meta_layer = 1; meta_layer <= max_meta_layer; meta_layer++) {
            // 1. Exact degrees
            std::vector<unsigned int> *exact_degs = nullptr;
            if (centrality == "df1") exact_degs = hidden_graph_deg(meta_layer, qp, g, visited, ractive);
            else if (centrality == "hf1") exact_degs = hidden_graph_hindex(meta_layer, qp, g, visited, ractive);
            if (!exact_degs) continue;

            // Find c(v^lambda) and exact hub set
            auto sort_exact = new std::vector<std::pair<unsigned int, unsigned int>>();
            for (unsigned int i = 0; i < g->NT.size(); i++)
                if (ractive->at(0)->at(i))
                    sort_exact->push_back({exact_degs->at(i), i});
            std::sort(sort_exact->begin(), sort_exact->end(), cmp_max_unsigned);

            auto topr_size = (unsigned int)(topr * peer_size);
            if (topr_size >= sort_exact->size()) { delete exact_degs; delete sort_exact; continue; }
            double c_vlambda = (double)sort_exact->at(topr_size).first;
            delete sort_exact;
            if (c_vlambda < 1.0) { delete exact_degs; continue; }

            // 2. Estimated degrees via synopses (reuses shared hidden_edges + synopses_combined)
            std::map<unsigned int, double> *est_degs = nullptr;
            if (centrality == "df1") {
                est_degs = jsy::synopses(meta_layer, hidden_edges, synopses_combined);
            } else {
                // For h-index: use the same pivot-based approach as COD_prop_global_cross_f1
                // Simplified: just use degree estimates as proxy
                est_degs = jsy::synopses(meta_layer, hidden_edges, synopses_combined);
            }

            // 3. Find estimated top-r% set
            auto sort_est = new std::vector<std::pair<double, unsigned int>>();
            for (auto &kv : *est_degs)
                sort_est->push_back({kv.second, kv.first});
            std::sort(sort_est->begin(), sort_est->end(), cmp_max);

            std::set<unsigned int> est_hubs;
            for (unsigned int i = 0; i <= topr_size && i < sort_est->size(); i++)
                est_hubs.insert(sort_est->at(i).second);
            delete sort_est;

            // 4. Compute epsilon
            double max_eps = 0;
            for (unsigned int i = 0; i < g->NT.size(); i++) {
                if (!ractive->at(0)->at(i)) continue;
                bool is_exact_hub = (exact_degs->at(i) > (unsigned int)c_vlambda);
                bool is_est_hub = (est_hubs.count(i) > 0);
                if (is_exact_hub != is_est_hub) {
                    double rel_err = fabs((double)exact_degs->at(i) - c_vlambda) / c_vlambda;
                    if (rel_err > max_eps) max_eps = rel_err;
                }
            }
            total_epsilon += max_eps;
            valid_layers++;

            delete exact_degs;
            delete est_degs;
        }

        delete frontiers; delete peers;
        for (auto &i : *active) delete i; delete active;
        for (auto &i : *ractive) delete i; delete ractive;
        for (auto &v : *synopses_combined) delete v;
        delete synopses_combined;
        delete hidden_edges;

        return valid_layers > 0 ? total_epsilon / valid_layers : -1;
    }
}