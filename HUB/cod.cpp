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

namespace hg{
    bool COD_global(Pattern *qp, HeterGraph *g, std::vector<std::vector<bool>*>* visited,
            std::vector<std::vector<bool>*>* back_visited, const std::string &centrality, double & avg_time){
        double topr = 0.1;

        // NTypes guard removed: 2-hop instance rules now allowed

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
        avg_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        start = std::chrono::steady_clock::now();
        std::vector<std::set<unsigned int> *> *active = ActiveMidNodes(peers, frontiers, qp, g, visited, back_visited);
        auto ractive = new std::vector<std::vector<unsigned int> *>();
        for (unsigned int l = 0; l <= qp->ETypes.size(); l++) ractive->push_back(new std::vector<unsigned int>(g->NT.size(), 0));
        for (unsigned int n = 0; n < g->NT.size(); n++) for (unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;

        unsigned int max_meta_layer;
        if(qp->instance == -1) max_meta_layer = qp->ETypes.size();
        else max_meta_layer = std::max(1u, (unsigned int)(qp->ETypes.size() - 1));

        for(unsigned int meta_layer = 1;meta_layer <= max_meta_layer;meta_layer++){
            std::cout<<"# l="<<meta_layer<<std::endl;
            std::vector<unsigned int>* centrality_vec = nullptr;
            if(centrality == "d")
                centrality_vec = hidden_graph_deg(meta_layer, qp, g, visited, ractive);
            else if(centrality == "h")
                centrality_vec = hidden_graph_hindex(meta_layer, qp, g, visited, ractive);

            auto sort_degs = new std::vector<std::pair<unsigned int, unsigned int>>();
            for(unsigned int i=0;i<g->NT.size();i++)
                if(ractive->at(0)->at(i)) sort_degs->push_back(std::pair<unsigned int, unsigned int>(centrality_vec->at(i), i));
            std::sort(sort_degs->begin(), sort_degs->end(), cmp_max_unsigned);
            auto topr_size = (unsigned int)(topr*peer_size);
            unsigned int v_lambda = sort_degs->at(topr_size).second;

            std::cout<<"v_0.1:"<<v_lambda<<std::endl;

            delete centrality_vec;
            delete sort_degs;
        }

        delete frontiers;
        delete peers;
        for(auto &i: *active) delete i;
        delete active;
        for(auto &i: *ractive) delete i;
        delete ractive;

        end = std::chrono::steady_clock::now();
        avg_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        return true;
    }

//    bool COD(Pattern *qp, HeterGraph *g, unsigned int qn, double topr, std::vector<std::vector<bool>*>* visited,
//             std::vector<std::vector<bool>*>* back_visited, const std::string &centrality,
//             std::vector<double>* time_statistic){
//        // NTypes guard removed: 2-hop instance rules now allowed
//
//        auto start = std::chrono::steady_clock::now();
//        auto frontiers = new std::vector<unsigned int>();
//        auto peers = new std::vector<unsigned int>();
//        Peers(qp, g, frontiers, peers, visited);
//        unsigned int peer_size = peers->size();
//        auto end = std::chrono::steady_clock::now();
//
//        if (peer_size <= 2 || std::find(peers->begin(), peers->end(), qn) == peers->end()) {
//            delete frontiers;
//            delete peers;
//            return false;
//        }
//
//        qp->print();
//        time_statistic->at(0) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
//
//        start = std::chrono::steady_clock::now();
//
//        std::vector<std::set<unsigned int> *> *active = ActiveMidNodes(peers, frontiers, qp, g, visited, back_visited);
//        auto ractive = new std::vector<std::vector<unsigned int> *>();
//        for (unsigned int l = 0; l <= qp->ETypes.size(); l++) ractive->push_back(new std::vector<unsigned int>(g->NT.size(), 0));
//        for (unsigned int n = 0; n < g->NT.size(); n++) for (unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;
//
//        unsigned int max_meta_layer;
//        if(qp->instance == -1) max_meta_layer = qp->ETypes.size();
//        else max_meta_layer = std::max(1u, (unsigned int)(qp->ETypes.size() - 1));
//
//        for (unsigned int meta_layer = 1; meta_layer <= max_meta_layer; meta_layer++) {
//            std::cout << "# l=" << meta_layer << std::endl;
//
//            std::vector<unsigned int>* centrality_vec = nullptr;
//
//            if(centrality == "d")
//                centrality_vec = hidden_graph_deg(meta_layer, qp, g, visited, ractive);
//            else if(centrality == "h")
//                centrality_vec = hidden_graph_hindex(meta_layer, qp, g, visited, ractive);
//
//            unsigned int rank = 0;
//            unsigned int q_centrality = centrality_vec->at(qn);
//            for(unsigned int i=0;i<g->NT.size();i++){
//                if(ractive->at(0)->at(i)){
//                    if(centrality_vec->at(i)>=q_centrality){
//                        rank++;
//                    }
//                }
//            }
//            std::cout<<"topr:"<<((rank*1.0)/peer_size)<<std::endl;
//            delete centrality_vec;
//        }
//
//        delete frontiers;
//        delete peers;
//        for (auto &i : *active) delete i;
//        delete active;
//        for (auto &i : *ractive) delete i;
//        delete ractive;
//
//        end = std::chrono::steady_clock::now();
//        time_statistic->at(0) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
//
//        return true;
//    }
}

namespace prop {
    void density(Pattern *qp, HeterGraph *g, std::vector<std::vector<bool>*>* visited,
                 std::vector<std::vector<bool>*>* back_visited, double dens_thresh){
        // NTypes guard removed: 2-hop instance rules now allowed
        auto frontiers = new std::vector<unsigned int>();
        auto peers = new std::vector<unsigned int>();
        Peers(qp, g, frontiers, peers, visited);
        unsigned int peer_size = peers->size();

        if(peer_size <= 2){
            delete frontiers;
            delete peers;
            return;
        }

        std::vector<std::set<unsigned int>* >* active = ActiveMidNodes(peers, frontiers, qp, g, visited, back_visited);
        auto ractive = new std::vector<std::vector<unsigned int>*>();
        for(unsigned int l=0; l<=qp->ETypes.size();l++) ractive->push_back(new std::vector<unsigned int>(g->NT.size(), 0));
        for(unsigned int n=0;n<g->NT.size();n++) for(unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;

        auto synopses_cache_opt = new std::vector<std::vector<jsy::Synopse>*>();
        for(unsigned int l = 0;l <= qp->ETypes.size();l++){
            synopses_cache_opt->push_back(new std::vector<jsy::Synopse>());
            for(unsigned int p=0;p<g->NT.size();p++)
                if(ractive->at(l)->at(p) > 0){
                    jsy::Synopse s; init_synopse(&s, p);
                    synopses_cache_opt->at(l)->push_back(s);
                    ractive->at(l)->at(p) = synopses_cache_opt->at(l)->size();
                }
        }

        auto hidden_edges = new std::vector<HiddenEdge>();
        for(unsigned int l=0;l < qp->ETypes.size();l++){
            for(unsigned int p=0;p < g->NT.size();p++){
                if(ractive->at(l)->at(p) > 0){
                    if (qp->EDirect[l] == 1) {
                        for (unsigned int nbr: *(g->EL[p])) {
                            if (ractive->at(l + 1)->at(nbr) > 0) {
                                HiddenEdge he{.s=ractive->at(l)->at(p) - 1, .t=ractive->at(l+1)->at(nbr) - 1, .l=l};
                                hidden_edges->push_back(he);
                            }}}
                    else {
                        for (unsigned int nbr: *(g->rEL[p])) {
                            if (ractive->at(l + 1)->at(nbr) > 0) {
                                HiddenEdge he{.s=ractive->at(l)->at(p) - 1, .t=ractive->at(l+1)->at(nbr) - 1, .l=l};
                                hidden_edges->push_back(he); }}}}}}

        unsigned int max_meta_layer;
        if(qp->instance == -1) max_meta_layer = qp->ETypes.size();
        else max_meta_layer = std::max(1u, (unsigned int)(qp->ETypes.size() - 1));

        for(unsigned int meta_layer = 1;meta_layer<= max_meta_layer;meta_layer++){
            auto degs= jsy::synopses(meta_layer, hidden_edges, synopses_cache_opt);
            double dens = 0;
            for(unsigned int i=0;i<g->NT.size();i++) if(ractive->at(0)->at(i)) dens += degs->at(i);
            dens /= (peer_size * (peer_size - 1));
            dens *= 2;

            if(dens <= dens_thresh){
                qp->print();
                std::cout<<"asso="<<meta_layer<<std::endl;
                std::cout<<"dens="<<dens<<std::endl;
                std::cout<<"|peers|="<<peer_size<<std::endl;
            }
        }
    }

    bool COD_global(Pattern *qp, HeterGraph *g, double topr, std::vector<std::vector<bool>*>* visited,
            std::vector<std::vector<bool>*>* back_visited, const std::string &method,
            std::vector<double>* time_statistics){
        // NTypes guard removed: 2-hop instance rules now allowed

        auto start = std::chrono::steady_clock::now();
        auto frontiers = new std::vector<unsigned int>();
        auto peers = new std::vector<unsigned int>();
        Peers(qp, g, frontiers, peers, visited);
        unsigned int peer_size = peers->size();
        auto end = std::chrono::steady_clock::now();

        if(peer_size <= 2){
            delete frontiers;
            delete peers;
            return false;
        }
        qp->print();

        time_statistics->at(0) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        start = std::chrono::steady_clock::now();
        std::vector<std::set<unsigned int>* >* active = ActiveMidNodes(peers, frontiers, qp, g, visited, back_visited);
        auto ractive = new std::vector<std::vector<unsigned int>*>();
        for(unsigned int l=0; l<=qp->ETypes.size();l++) ractive->push_back(new std::vector<unsigned int>(g->NT.size(), 0));
        for(unsigned int n=0;n<g->NT.size();n++) for(unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;

        auto synopses_cache_opt = new std::vector<std::vector<jsy::Synopse>*>();
        for(unsigned int l = 0;l <= qp->ETypes.size();l++){
            synopses_cache_opt->push_back(new std::vector<jsy::Synopse>());
            for(unsigned int p=0;p<g->NT.size();p++)
                if(ractive->at(l)->at(p) > 0){
                    jsy::Synopse s; init_synopse(&s, p);
                    synopses_cache_opt->at(l)->push_back(s);
                    ractive->at(l)->at(p) = synopses_cache_opt->at(l)->size();
                }
        }

        auto hidden_edges = new std::vector<HiddenEdge>();
        for(unsigned int l=0;l < qp->ETypes.size();l++){
            for(unsigned int p=0;p < g->NT.size();p++){
                if(ractive->at(l)->at(p) > 0){
                    if (qp->EDirect[l] == 1) {
                        for (unsigned int nbr: *(g->EL[p])) {
                            if (ractive->at(l + 1)->at(nbr) > 0) {
                                HiddenEdge he{.s=ractive->at(l)->at(p) - 1, .t=ractive->at(l+1)->at(nbr) - 1, .l=l};
                                hidden_edges->push_back(he);
                            }}}
                    else {
                        for (unsigned int nbr: *(g->rEL[p])) {
                            if (ractive->at(l + 1)->at(nbr) > 0) {
                                HiddenEdge he{.s=ractive->at(l)->at(p) - 1, .t=ractive->at(l+1)->at(nbr) - 1, .l=l};
                                hidden_edges->push_back(he); }}}}}}

        unsigned int max_meta_layer;
        if(qp->instance == -1) max_meta_layer = qp->ETypes.size();
        else max_meta_layer = std::max(1u, (unsigned int)(qp->ETypes.size() - 1));

        for(unsigned int meta_layer = 1;meta_layer<= max_meta_layer;meta_layer++){
            std::cout<<"# l="<<meta_layer<<std::endl;
            auto topr_size = (unsigned int)(topr * peer_size);
            std::vector<unsigned int> props_dom_peers;

            if(method == "d"){
                auto props_degs = jsy::synopses(meta_layer, hidden_edges, synopses_cache_opt);
                auto sort_degs = new std::vector<std::pair<double, unsigned int>>();
                for(unsigned int i=0;i<g->NT.size();i++)
                    if(ractive->at(0)->at(i)) sort_degs->push_back(std::pair<double, unsigned int>(props_degs->at(i), i));
                std::sort(sort_degs->begin(), sort_degs->end(), cmp_max);

                for(unsigned int i=0;i<=topr_size;i++)
                    props_dom_peers.push_back(sort_degs->at(i).second);
            }
            else if(method == "h"){
                auto degs = jsy::synopses(meta_layer, hidden_edges, synopses_cache_opt);
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
                    auto *hindexes = jsy::synopses(meta_layer, hidden_edges, synopses_cache_opt, qualified_nodes);

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
            std::cout<<"prop_size:"<<props_dom_peers.size()<<std::endl;
        }
        end = std::chrono::steady_clock::now();
        time_statistics->at(0) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        delete frontiers;
        delete peers;
        for (auto &i : *active) delete i;
        delete active;
        for (auto &i : *ractive) delete i;
        delete ractive;
        for (auto &i : *synopses_cache_opt) delete i;
        delete synopses_cache_opt;
        delete hidden_edges;

        return true;
    }

    bool COD(Pattern *qp, HeterGraph *g, unsigned int qn, double topr, double delta_r, double delta_qd,
                std::vector<std::vector<bool>*>* visited,
                std::vector<std::vector<bool>*>* back_visited, const std::string &method,
                std::vector<double>* time_statistics){
        // NTypes guard removed: 2-hop instance rules now allowed

        auto start = std::chrono::steady_clock::now();
        auto frontiers = new std::vector<unsigned int>();
        auto peers = new std::vector<unsigned int>();
        Peers(qp, g, frontiers, peers, visited);
        unsigned int peer_size = peers->size();
        auto end = std::chrono::steady_clock::now();

        if(peer_size <= 2 || std::find(peers->begin(), peers->end(), qn)==peers->end()){
            delete frontiers;
            delete peers;
            return false;
        }
        qp->print();

        time_statistics->at(0) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        start = std::chrono::steady_clock::now();
        std::vector<std::set<unsigned int>* >* active = ActiveMidNodes(peers, frontiers, qp, g, visited, back_visited);
        auto ractive = new std::vector<std::vector<unsigned int>*>();
        for(unsigned int l=0; l<=qp->ETypes.size();l++) ractive->push_back(new std::vector<unsigned int>(g->NT.size(), 0));
        for(unsigned int n=0;n<g->NT.size();n++) for(unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;

        auto synopses_cache_opt = new std::vector<std::vector<jsy::Synopse>*>();
        for(unsigned int l = 0;l <= qp->ETypes.size();l++){
            synopses_cache_opt->push_back(new std::vector<jsy::Synopse>());
            for(unsigned int p=0;p<g->NT.size();p++)
                if(ractive->at(l)->at(p) > 0){
                    jsy::Synopse s; init_synopse(&s, p);
                    synopses_cache_opt->at(l)->push_back(s);
                    ractive->at(l)->at(p) = synopses_cache_opt->at(l)->size();
                }
        }

        auto hidden_edges = new std::vector<HiddenEdge>();
        for(unsigned int l=0;l < qp->ETypes.size();l++){
            for(unsigned int p=0;p < g->NT.size();p++){
                if(ractive->at(l)->at(p) > 0){
                    if (qp->EDirect[l] == 1) {
                        for (unsigned int nbr: *(g->EL[p])) {
                            if (ractive->at(l + 1)->at(nbr) > 0) {
                                HiddenEdge he{.s=ractive->at(l)->at(p) - 1, .t=ractive->at(l+1)->at(nbr) - 1, .l=l};
                                hidden_edges->push_back(he);
                            }}}
                    else {
                        for (unsigned int nbr: *(g->rEL[p])) {
                            if (ractive->at(l + 1)->at(nbr) > 0) {
                                HiddenEdge he{.s=ractive->at(l)->at(p) - 1, .t=ractive->at(l+1)->at(nbr) - 1, .l=l};
                                hidden_edges->push_back(he); }}}}}}

        unsigned int max_meta_layer;
        if(qp->instance == -1) max_meta_layer = qp->ETypes.size();
        else max_meta_layer = std::max(1u, (unsigned int)(qp->ETypes.size() - 1));

        for(unsigned int meta_layer = 1;meta_layer<= max_meta_layer;meta_layer++){
            std::cout<<"# l="<<meta_layer<<std::endl;

            if(method == "d"){
                auto degs = jsy::synopses(meta_layer, hidden_edges, synopses_cache_opt);

                unsigned int rank = 0;
                double q_deg = degs->at(qn);
                for(unsigned int i=0;i<g->NT.size();i++){
                    if(ractive->at(0)->at(i)){
                        if(degs->at(i) >= q_deg){
                            rank++;
                        }
                    }
                }
                std::cout<<"topr:"<<((rank*1.0)/peer_size)<<std::endl;
                delete degs;
            }
            else if (method == "dp"){
                auto q_peers = Peers(qn, qp, meta_layer, g, visited, ractive);
                unsigned int q_deg_exact = q_peers->size();
                delete q_peers;

                auto degs = jsy::synopses_fp_deg(meta_layer, hidden_edges, synopses_cache_opt, q_deg_exact*(1.0+delta_qd), topr*(1.0+delta_r), peer_size);
                if(degs == nullptr) {
                    std::cout << "topr: pruned" << std::endl;
                    continue;
                }
                unsigned int rank = 0;
                double q_deg = degs->at(qn);
                for(unsigned int i=0;i<g->NT.size();i++){
                    if(ractive->at(0)->at(i)){
                        if(degs->at(i) >= q_deg){
                            rank++;
                        }
                    }
                }
                std::cout<<"topr:"<<((rank*1.0)/peer_size)<<std::endl;
                delete degs;
            }
            else if (method == "h"){
                auto qn_peers = Peers(qn, qp, meta_layer, g, visited, ractive);

                auto degs = jsy::synopses(meta_layer, hidden_edges, synopses_cache_opt);
                auto sort_degs = new std::vector<std::pair<double, unsigned int>>();
                for(unsigned int n: *qn_peers)
                    sort_degs->push_back(std::pair<double, unsigned int>(degs->at(n), n));
                delete qn_peers;
                std::sort(sort_degs->begin(), sort_degs->end(), cmp_max);

                unsigned int qn_hindex = 0;
                for(auto &sort_deg: *sort_degs){
                    qn_hindex++;
                    if (qn_hindex > sort_deg.first) break;
                }
                if(qn_hindex > sort_degs->at(qn_hindex-1).first) qn_hindex--;
                delete sort_degs;

                std::vector<bool> *qualified_nodes = new std::vector<bool>(g->NT.size(), false);
                for(unsigned int peer: *peers) if(degs->at(peer)>qn_hindex) qualified_nodes->at(peer) = true;
                auto *hindexes = jsy::synopses(meta_layer, hidden_edges, synopses_cache_opt, qualified_nodes);
                delete qualified_nodes;

                unsigned int qn_hindex_rank = 0;
                for (auto &hindexe : *hindexes) {
                    if(hindexe.second >= hindexes->at(qn)) qn_hindex_rank++;
                }
                delete hindexes;
                delete degs;
                std::cout<<"topr:"<<((qn_hindex_rank*1.0)/peer_size)<<std::endl;
            }
            else if(method == "hp"){
                auto q_peers = Peers(qn, qp, meta_layer, g, visited, ractive);
                unsigned int q_deg = q_peers->size();

                double mcs = 0;

                // compute the degress of peers
                auto temp_deg = jsy::synopses_fp(meta_layer, hidden_edges, synopses_cache_opt, q_deg*(1.0+delta_qd),
                        topr*(1.0+delta_r), peer_size, mcs);
                if(temp_deg == nullptr) {
                    delete q_peers;
                    std::cout<<"topr: pruned"<<std::endl;
                    continue;
                }

                std::vector<double> *degs = new std::vector<double>(g->NT.size(), 0);
                for(auto &it: *temp_deg){
                    unsigned int n=it.first;
                    double d = it.second;
                    degs->at(n) += d;
                }

                delete temp_deg;

                //compute the h-index of qn
                auto sort_degs = new std::vector<std::pair<double, unsigned int>>();
                for(unsigned int n: *q_peers)
                    sort_degs->push_back(std::pair<double, unsigned int>(degs->at(n), n));
                delete q_peers;
                std::sort(sort_degs->begin(), sort_degs->end(), cmp_max);

                unsigned int q_hindex = 0;
                for(auto &sort_deg: *sort_degs){
                    q_hindex++;
                    if(q_hindex > sort_deg.first) break;
                }
                if(q_hindex > sort_degs->at(q_hindex-1).first) q_hindex--;

                if(mcs>=q_hindex && mcs>peer_size * topr) {
                    delete degs;
                    delete sort_degs;
                    continue;
                }

                //compute peers with h-index larger than qn
                std::vector<double> *hindexes = new std::vector<double>(g->NT.size(), 0);
                std::vector<bool> *qualified_nodes = new std::vector<bool>(g->NT.size(), false);
                for (unsigned int peer: *peers) if (degs->at(peer) > q_hindex) qualified_nodes->at(peer) = true;
                auto temp_hindex = jsy::synopses(meta_layer, hidden_edges, synopses_cache_opt, qualified_nodes);
                for (auto &it: *temp_hindex) {
                    unsigned int n = it.first;
                    double d = it.second;
                    hindexes->at(n) += d;
                }
                delete temp_hindex;

                unsigned int q_hindex_rank = 0;
                for (unsigned int i = 0; i < hindexes->size(); i++) if (qualified_nodes->at(i))
                        if (hindexes->at(i) >= hindexes->at(qn)) q_hindex_rank++;

                std::cout<<"topr:"<<((q_hindex_rank*1.0)/peer_size)<<std::endl;
                delete hindexes;
                delete qualified_nodes;
                delete degs;
                delete sort_degs;
            }
        }
        end = std::chrono::steady_clock::now();
        time_statistics->at(0) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

        delete frontiers;
        delete peers;
        for (auto &i : *active) delete i;
        delete active;
        for (auto &i : *ractive) delete i;
        delete ractive;
        for (auto &i : *synopses_cache_opt) delete i;
        delete synopses_cache_opt;
        delete hidden_edges;

        return true;
    }
}