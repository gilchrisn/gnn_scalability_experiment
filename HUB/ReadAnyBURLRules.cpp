#include "cod.cpp"
#include "effectiveness.cpp"
#include <chrono>
#include <thread>
#include <iostream>
#include <string>


using namespace std;

std::vector<std::string> split(const std::string& str, const std::string &pattern, bool app){
    std::vector<std::string> res;
    if(str == "") return res;

    std::string::size_type temp_pos;
    std::string appstr;
    if(app) appstr = str + pattern;
    else appstr = str;

    int size = appstr.size();
    for(unsigned int i=0;i<size;i++){
        temp_pos = appstr.find(pattern, i);
        if(temp_pos < size){
            std::string substr = appstr.substr(i, temp_pos - i);
            res.push_back(substr);
            i = temp_pos;
        }
    }
    return res;
}

//void ReadAnyBURLRules_DensityFilter(const std::string &choice, double dens_threshold){
//    HeterGraph g(choice);
//
//    std::string rules_path = (choice + "/cod-gnn-rules.dat");
//    std::ifstream rules_in;
//    rules_in.open(rules_path);
//    std::string rules_line;
//
//    unsigned int rule_count = 0;
//
//    auto visited = new std::vector<std::vector<bool>*>();
//    auto back_visited = new std::vector<std::vector<bool>*>();
//    auto qp = new Pattern();
//
//    getline(rules_in, rules_line);
//    int state = -1;
//    // state == 0: next int represents a variable rule;
//    // state == 1: next int represents a instance rule;
//
//    std::string::size_type temp_pos;
//    int size = rules_line.size();
//    for(unsigned int i = 0; i<size;i++) {
//        temp_pos = rules_line.find(' ', i);
//        if (temp_pos < size) {
//            int sub = stoi(rules_line.substr(i, temp_pos - i));
//
//            if (sub == -1) state = 0; // next int indicates a variable rule
//            else if (sub == -2) qp->EDirect.push_back(1); // ->
//            else if (sub == -3) qp->EDirect.push_back(-1); // <-
//            else if (sub == -4) { // pop
//                qp->EDirect.pop_back();
//                qp->ETypes.pop_back();
//                qp->NTypes.pop_back();}
//            else if (sub == -5) state = 1; // next int indicates a instance rule
//            else {
//                if (state == 1) { // instance rule
//                    qp->instance = sub;
//                    while(visited->size()< qp->ETypes.size()+1){
//                        visited->push_back(new std::vector<bool>(g.NT.size(), false));
//                        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
//                    }
//                    prop::density(qp, &g, visited, back_visited, dens_threshold);
//                    qp->instance = -1;
//                    state = -1;
//                } else if (state == 0) { // variable rule
//                    qp->ETypes.push_back(sub);
//                    qp->NTypes.push_back(-1);
//                    while(visited->size()< qp->ETypes.size()+1){
//                        visited->push_back(new std::vector<bool>(g.NT.size(), false));
//                        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
//                    }
//                    prop::density(qp, &g, visited, back_visited, dens_threshold);
//                    state = -1;
//                } else {
//                    qp->ETypes.push_back(sub);
//                    qp->NTypes.push_back(-1);
//                }
//            }
//            i = temp_pos;
//        }
//    }
//    rules_in.close();
//
//    for (auto &it : *visited) delete it;
//    for (auto &it : *back_visited) delete it;
//    delete visited;
//    delete back_visited;
//}

void MetaPathLenSplit(const std::string &choice){
    std::string rules_path = (choice + "/"+choice+"-cod-global-rules.dat");
    std::ifstream rules_in;
    rules_in.open(rules_path);
    std::string rules_line;

    std::string rules_path_len1 = (choice + "/"+choice+"-cod-global-rules.dat1");
    std::ofstream rules_out1;
    rules_out1.open(rules_path_len1);

    std::string rules_path_len2 = (choice + "/"+choice+"-cod-global-rules.dat2");
    std::ofstream rules_out2;
    rules_out2.open(rules_path_len2);

    std::string rules_path_len3 = (choice + "/"+choice+"-cod-global-rules.dat3");
    std::ofstream rules_out3;
    rules_out3.open(rules_path_len3);

    auto qp = new Pattern();

    getline(rules_in, rules_line);
    rules_in.close();

    int state = -1;
    std::string::size_type temp_pos;
    int size = rules_line.size();

    for(unsigned int i = 0; i<size;i++) {
        temp_pos = rules_line.find(' ', i);
        if (temp_pos < size) {
            int sub = stoi(rules_line.substr(i, temp_pos - i));

            if (sub == -1) state = 0; // next int indicates a variable rule
            else if (sub == -2) qp->EDirect.push_back(1); // ->
            else if (sub == -3) qp->EDirect.push_back(-1); // <-
            else if (sub == -4) { // pop
                qp->EDirect.pop_back();
                qp->ETypes.pop_back();
                qp->NTypes.pop_back();
            } else if (sub == -5) state = 1; // next int indicates a instance rule
            else {
                if (state == 1) { // instance rule
                    qp->instance = sub;

                    unsigned int len = qp->ETypes.size() - 1;
                    if(len == 1){
                        for(int jj=0;jj<qp->ETypes.size();jj++){
                            if(qp->EDirect[jj] == 1) rules_out1 << "-2 ";
                            else rules_out1<< "-3 ";
                            rules_out1<<qp->ETypes[jj]<<" ";
                        }
                        rules_out1<<"-5 "<<qp->instance<<" ";
                        for(int jj=0;jj<qp->ETypes.size();jj++) rules_out1<<"-4 ";
                    }
                    else if(len == 2){
                        for(int jj=0;jj<qp->ETypes.size();jj++){
                            if(qp->EDirect[jj] == 1) rules_out2 << "-2 ";
                            else rules_out2<< "-3 ";
                            rules_out2<<qp->ETypes[jj]<<" ";
                        }
                        rules_out2<<"-5 "<<qp->instance<<" ";
                        for(int jj=0;jj<qp->ETypes.size();jj++) rules_out2<<"-4 ";
                    }
                    else if(len == 3){
                        for(int jj=0;jj<qp->ETypes.size();jj++){
                            if(qp->EDirect[jj] == 1) rules_out3 << "-2 ";
                            else rules_out3<< "-3 ";
                            rules_out3<<qp->ETypes[jj]<<" ";
                        }
                        rules_out3<<"-5 "<<qp->instance<<" ";
                        for(int jj=0;jj<qp->ETypes.size();jj++) rules_out3<<"-4 ";
                    }

                    qp->instance = -1;
                    state = -1;
                } else if (state == 0) { // variable rule
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);

                    unsigned int len = qp->ETypes.size();
                    if(len == 1){
                        for(int jj=0;jj<qp->ETypes.size();jj++){
                            if(jj == qp->ETypes.size()-1) rules_out1 <<"-1 ";
                            if(qp->EDirect[jj] == 1) rules_out1 << "-2 ";
                            else rules_out1<< "-3 ";
                            rules_out1<<qp->ETypes[jj]<<" ";
                        }
                        for(int jj=0;jj<qp->ETypes.size();jj++) rules_out1<<"-4 ";
                    }
                    else if(len == 2){
                        for(int jj=0;jj<qp->ETypes.size();jj++){
                            if(jj == qp->ETypes.size()-1) rules_out2 <<"-1 ";
                            if(qp->EDirect[jj] == 1) rules_out2 << "-2 ";
                            else rules_out2<< "-3 ";
                            rules_out2<<qp->ETypes[jj]<<" ";
                        }
                        for(int jj=0;jj<qp->ETypes.size();jj++) rules_out2<<"-4 ";
                    }
                    else if(len == 3){
                        for(int jj=0;jj<qp->ETypes.size();jj++){
                            if(jj == qp->ETypes.size()-1) rules_out3 <<"-1 ";
                            if(qp->EDirect[jj] == 1) rules_out3 << "-2 ";
                            else rules_out3<< "-3 ";
                            rules_out3<<qp->ETypes[jj]<<" ";
                        }
                        for(int jj=0;jj<qp->ETypes.size();jj++) rules_out3<<"-4 ";
                    }
                    state = -1;
                } else {
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);
                }
            }
            i = temp_pos;
        }
    }
    rules_out1.close();
    rules_out2.close();
    rules_out3.close();
}

void MetaPathCount(const std::string &choice, unsigned int limited){
    unsigned int LEN = 5;

    std::string rules_path;
    if (limited == 0) rules_path = (choice + "/"+choice+"-cod-global-rules.dat");
    else rules_path = (choice + "/"+choice+"-cod-global-rules.limit");

    std::ifstream rules_in;
    rules_in.open(rules_path);
    std::string rules_line;

    auto qp = new Pattern();
    unsigned int rule_count = 0;

    getline(rules_in, rules_line);
    rules_in.close();

    std::vector<unsigned int> path_len_count;
    for(unsigned int i=0;i<=LEN;i++) path_len_count.push_back(0);

    int state = -1;

    std::string::size_type temp_pos;
    int size = rules_line.size();

    for(unsigned int i = 0; i<size;i++) {
        temp_pos = rules_line.find(' ', i);
        if (temp_pos < size) {
            int sub = stoi(rules_line.substr(i, temp_pos - i));

            if (sub == -1) state = 0; // next int indicates a variable rule
            else if (sub == -2) qp->EDirect.push_back(1); // ->
            else if (sub == -3) qp->EDirect.push_back(-1); // <-
            else if (sub == -4) { // pop
                    qp->EDirect.pop_back();
                    qp->ETypes.pop_back();
                    qp->NTypes.pop_back();
            } else if (sub == -5) state = 1; // next int indicates a instance rule
            else {
                if (state == 1) { // instance rule
                    qp->instance = sub;

                    unsigned int len = qp->ETypes.size() - 1;
                    if(len < LEN) path_len_count[len] += 1;

                    rule_count++;
                    qp->instance = -1;
                    state = -1;
                } else if (state == 0) { // variable rule
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);

                    unsigned int len = qp->ETypes.size();
                    if(len < LEN) path_len_count[len] += 1;

                    rule_count++;
                    state = -1;
                } else {
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);
                }
            }
            i = temp_pos;
        }
    }
    std::cout<<"Length Count:"<<std::endl;
    for(unsigned int i=1;i<=LEN;i++)
        std::cout<<"len="<<i<<"  count:"<<path_len_count[i]<<std::endl;
    std::cout<<std::endl;
    std::cout<<"rule_count:"<<rule_count<<std::endl;
}

void MatchingGraphTime(const std::string &choice){
    HeterGraph g(choice);

    std::string rules_path = (choice + "/"+choice+"-cod-global-rules.dat");
    std::ifstream rules_in;
    rules_in.open(rules_path);
    std::string rules_line;

    auto visited = new std::vector<std::vector<bool>*>();
    auto back_visited = new std::vector<std::vector<bool>*>();
    auto qp = new Pattern();

    unsigned int rule_count = 0;

    getline(rules_in, rules_line);
    rules_in.close();

    double avg_time = 0.0;

    int state = -1;
    std::string::size_type temp_pos;
    int size = rules_line.size();

    for(unsigned int i = 0; i<size;i++) {
        temp_pos = rules_line.find(' ', i);
        if (temp_pos < size) {
            int sub = stoi(rules_line.substr(i, temp_pos - i));

            if (sub == -1) state = 0; // next int indicates a variable rule
            else if (sub == -2) qp->EDirect.push_back(1); // ->
            else if (sub == -3) qp->EDirect.push_back(-1); // <-
            else if (sub == -4) { // pop
                qp->EDirect.pop_back();
                qp->ETypes.pop_back();
                qp->NTypes.pop_back();
            } else if (sub == -5) state = 1; // next int indicates a instance rule
            else {
                if (state == 1) { // instance rule
                    qp->instance = sub;
                    while(visited->size()< qp->ETypes.size()+1){
                        visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                    }

                    bool found = effectiveness::COD_matching_graph_time(qp, &g, visited, back_visited, avg_time);
                    if(found) rule_count++;

                    qp->instance = -1;
                    state = -1;
                } else if (state == 0) { // variable rule
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);
                    while(visited->size()< qp->ETypes.size()+1){
                        visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                    }
                    bool found = effectiveness::COD_matching_graph_time(qp, &g, visited, back_visited, avg_time);
                    if(found) rule_count++;
                    state = -1;
                } else {
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);
                }
            }
            i = temp_pos;
        }
    }
    qp->clear();

    for (auto &it : *visited) delete it;
    for (auto &it : *back_visited) delete it;
    delete visited;
    delete back_visited;

    avg_time /= rule_count;
    avg_time /= 1000000;

    std::cout<<"~matching_graph_time_per_rule:"<<avg_time<<" ms"<<std::endl;
    std::cout<<"rule_count:"<<rule_count<<std::endl;
}

void Scalability_prop_opt_global(const std::string &choice, const std::string &topr, const std::string &method, const std::string &len){
    HeterGraph g(choice);

    std::string rules_path = (choice + "/"+choice+"-cod-global-rules.dat"+len);
    std::ifstream rules_in;
    rules_in.open(rules_path);
    std::string rules_line;

    auto visited = new std::vector<std::vector<bool>*>();
    auto back_visited = new std::vector<std::vector<bool>*>();
    auto qp = new Pattern();

    unsigned int rule_count = 0;

    getline(rules_in, rules_line);
    rules_in.close();

    double avg_time = 0.0;

    int state = -1;
    std::string::size_type temp_pos;
    int size = rules_line.size();

    for(unsigned int i = 0; i<size;i++) {
        temp_pos = rules_line.find(' ', i);
        if (temp_pos < size) {
            int sub = stoi(rules_line.substr(i, temp_pos - i));

            if (sub == -1) state = 0; // next int indicates a variable rule
            else if (sub == -2) qp->EDirect.push_back(1); // ->
            else if (sub == -3) qp->EDirect.push_back(-1); // <-
            else if (sub == -4) { // pop
                qp->EDirect.pop_back();
                qp->ETypes.pop_back();
                qp->NTypes.pop_back();
            } else if (sub == -5) state = 1; // next int indicates a instance rule
            else {
                if (state == 1) { // instance rule
                    qp->instance = sub;
                    while(visited->size()< qp->ETypes.size()+1){
                        visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                    }

                    bool found = effectiveness::COD_prop_global_scale(qp, &g, std::stod(topr), visited, back_visited, method, avg_time);
                    if(found) rule_count++;
                    if(rule_count >= PATHCOUNT) break;

                    qp->instance = -1;
                    state = -1;
                } else if (state == 0) { // variable rule
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);
                    while(visited->size()< qp->ETypes.size()+1){
                        visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                    }
                    bool found =  effectiveness::COD_prop_global_scale(qp, &g, std::stod(topr), visited, back_visited, method, avg_time);
                    if(found) rule_count++;
                    if(rule_count >= PATHCOUNT) break;
                    state = -1;
                } else {
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);
                }
            }
            i = temp_pos;
        }
    }
    qp->clear();

    for (auto &it : *visited) delete it;
    for (auto &it : *back_visited) delete it;
    delete visited;
    delete back_visited;

    avg_time /= rule_count;
    avg_time /= 1000000000;

    std::cout<<"~time_per_rule:"<<avg_time<<" s"<<std::endl;
    std::cout<<"rule_count:"<<rule_count<<std::endl;
}

void Effective_prop_opt_global_cross(const std::string &choice, const std::string &topr, const std::string &method){
    std::ifstream hg_dom_greater_in, hg_dom_in;
    if(method == "df1") {
        hg_dom_greater_in.open("global_res/" + choice + "/df1/hg_global_greater_r"+topr+".res");
        hg_dom_in.open("global_res/"+choice+"/df1/hg_global_r"+topr+".res");
    }
    else {
        hg_dom_greater_in.open("global_res/" + choice + "/hf1/hg_global_greater_r"+topr+".res");
        hg_dom_in.open("global_res/"+choice+"/hf1/hg_global_r"+topr+".res");
    }
    std::string hg_dom_line;

    auto hg_doms_greater = new std::vector<std::vector<unsigned int>*>();
    auto hg_isdoms_greater = new std::vector<bool>();

    unsigned int absolute_rule_idx = 0;
    while(getline(hg_dom_greater_in, hg_dom_line)){
        if(hg_dom_line[0] == ')' || hg_dom_line[0] == '('){
            auto temp_doms = split(hg_dom_line, " ", false);
            hg_isdoms_greater->push_back(hg_dom_line[0] == '(');
            hg_doms_greater->push_back(new std::vector<unsigned int>());
            for(unsigned int i=1;i<temp_doms.size();i++){
                hg_doms_greater->at(hg_doms_greater->size()-1)->push_back((unsigned int)stoi(temp_doms[i]));
            }
        }
    }
    hg_dom_greater_in.close();

    auto hg_doms = new std::vector<std::vector<unsigned int>*>();
    auto hg_isdoms = new std::vector<bool>();

    while(getline(hg_dom_in, hg_dom_line)){
        if(hg_dom_line[0] == ')' || hg_dom_line[0] == '('){
            auto temp_doms = split(hg_dom_line, " ", false);
            hg_isdoms->push_back(hg_dom_line[0] == '(');
            hg_doms->push_back(new std::vector<unsigned int>());
            for(unsigned int i=1;i<temp_doms.size();i++){
                hg_doms->at(hg_doms->size()-1)->push_back((unsigned int)stoi(temp_doms[i]));
            }
        }
    }
    hg_dom_in.close();

    HeterGraph g(choice);

    std::string rules_path = (choice + "/"+choice+"-cod-global-rules.dat");
    std::ifstream rules_in;
    rules_in.open(rules_path);
    std::string rules_line;

    auto visited = new std::vector<std::vector<bool>*>();
    auto back_visited = new std::vector<std::vector<bool>*>();
    auto qp = new Pattern();

    unsigned int rule_count = 0;
    unsigned int dom_count = 0;
    double avg_relative_error = 0;

    getline(rules_in, rules_line);
    rules_in.close();

    double avg_time = 0.0;

    int state = -1;
    // state == 0: next int represents a variable rule;
    // state == 1: next int represents a instance rule;

    std::string::size_type temp_pos;
    int size = rules_line.size();

    for(unsigned int i = 0; i<size;i++) {
        temp_pos = rules_line.find(' ', i);
        if (temp_pos < size) {
            int sub = stoi(rules_line.substr(i, temp_pos - i));

            if (sub == -1) state = 0; // next int indicates a variable rule
            else if (sub == -2) qp->EDirect.push_back(1); // ->
            else if (sub == -3) qp->EDirect.push_back(-1); // <-
            else if (sub == -4) { // pop
                    qp->EDirect.pop_back();
                    qp->ETypes.pop_back();
                    qp->NTypes.pop_back();
            } else if (sub == -5) state = 1; // next int indicates a instance rule
            else {
                if (state == 1) { // instance rule
                    qp->instance = sub;
                    while(visited->size()< qp->ETypes.size()+1){
                        visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                    }

                    unsigned int active_edges = 0;
                    double pre_time = avg_time;
                    double relative_error = effectiveness::COD_prop_global_cross_f1(qp, &g, std::stod(topr), visited, back_visited, method, hg_doms, hg_isdoms, hg_doms_greater, hg_isdoms_greater, dom_count, avg_time, active_edges);
                    double rule_time = avg_time - pre_time;
                    std::cout << "SCATTER_DATA: " << absolute_rule_idx++ << "," << active_edges << "," << (rule_time/1000000000.0) << std::endl;
                    if(relative_error >= -0.1){
                        rule_count++;
                        avg_relative_error += relative_error;
                    }

                    qp->instance = -1;
                    state = -1;
                } else if (state == 0) { // variable rule
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);
                    while(visited->size()< qp->ETypes.size()+1){
                        visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                    }

                    unsigned int active_edges = 0;
                    double pre_time = avg_time;
                    double relative_error = effectiveness::COD_prop_global_cross_f1(qp, &g, std::stod(topr), visited, back_visited, method, hg_doms, hg_isdoms, hg_doms_greater, hg_isdoms_greater, dom_count, avg_time, active_edges);
                    double rule_time = avg_time - pre_time;
                    std::cout << "SCATTER_DATA: " << rule_count << "," << active_edges << "," << (rule_time/1000000000.0) << std::endl;
                    if(relative_error > -0.1){
                        rule_count++;
                        avg_relative_error += relative_error;
                    }
                    state = -1;
                } else {
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);
                }
            }
            i = temp_pos;
        }
    }
    qp->clear();

    for (auto &it : *visited) delete it;
    for (auto &it : *back_visited) delete it;
    delete visited;
    delete back_visited;

    avg_relative_error /= rule_count;
    avg_time /= rule_count;
    avg_time /= 1000000000;

    std::cout<<"~goodness:"<<avg_relative_error<<std::endl;
    std::cout<<"~time_per_rule:"<<avg_time<<" s"<<std::endl;
    std::cout<<"rule_count:"<<rule_count<<std::endl;
}

void Scalability_prop_opt_personalized(const std::string &choice, const std::string &topr, double beta, const std::string &method, const std::string &len){
    HeterGraph g(choice);

    std::string rules_path = (choice + "/"+choice+"-cod-global-rules.dat"+len);
    std::ifstream rules_in;
    rules_in.open(rules_path);
    std::string rules_line;

    auto visited = new std::vector<std::vector<bool>*>();
    auto back_visited = new std::vector<std::vector<bool>*>();
    auto qp = new Pattern();

    unsigned int rule_count = 0;

    getline(rules_in, rules_line);
    rules_in.close();

    double avg_time = 0.0;

    int state = -1;
    // state == 0: next int represents a variable rule;
    // state == 1: next int represents a instance rule;

    std::string::size_type temp_pos;
    int size = rules_line.size();

    for(unsigned int i = 0; i<size;i++) {
        temp_pos = rules_line.find(' ', i);
        if (temp_pos < size) {
            int sub = stoi(rules_line.substr(i, temp_pos - i));

            if (sub == -1) state = 0; // next int indicates a variable rule
            else if (sub == -2) qp->EDirect.push_back(1); // ->
            else if (sub == -3) qp->EDirect.push_back(-1); // <-
            else if (sub == -4) { // pop
                qp->EDirect.pop_back();
                qp->ETypes.pop_back();
                qp->NTypes.pop_back();
            } else if (sub == -5) state = 1; // next int indicates a instance rule
            else {
                if (state == 1) { // instance rule
                    qp->instance = sub;
                    while(visited->size()< qp->ETypes.size()+1){
                        visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                    }

                    bool found = effectiveness::COD_prop_personalized_scale(qp, &g, std::stod(topr), beta, visited, back_visited, method, avg_time);
                    if(found) rule_count++;
                    if(rule_count >= PATHCOUNT) break;

                    qp->instance = -1;
                    state = -1;
                } else if (state == 0) { // variable rule
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);
                    while(visited->size()< qp->ETypes.size()+1){
                        visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                    }
                    bool found = effectiveness::COD_prop_personalized_scale(qp, &g, std::stod(topr), beta, visited, back_visited, method, avg_time);
                    if(found) rule_count++;
                    if(rule_count >= PATHCOUNT) break;

                    state = -1;
                } else {
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);
                }
            }
            i = temp_pos;
        }
    }
    qp->clear();

    for (auto &it : *visited) delete it;
    for (auto &it : *back_visited) delete it;
    delete visited;
    delete back_visited;

    avg_time /= rule_count;
    avg_time /= 1000000000;

    std::cout<<"~time_per_rule:"<<avg_time<<" s"<<std::endl;
    std::cout<<"rule_count:"<<rule_count<<std::endl;
}

void Effective_prop_opt_personalized_cross(const std::string &choice, const std::string &topr, double beta, const std::string &method){
    std::ifstream hg_dom_greater_in, hg_dom_in;
    if(method == "df1" || method == "dp" || method == "d") {
        hg_dom_greater_in.open("global_res/" + choice + "/df1/hg_global_greater_r"+topr+".res");
        hg_dom_in.open("global_res/"+choice+"/df1/hg_global_r"+topr+".res");
    }
    else {
        hg_dom_greater_in.open("global_res/" + choice + "/hf1/hg_global_greater_r"+topr+".res");
        hg_dom_in.open("global_res/"+choice+"/hf1/hg_global_r"+topr+".res");
    }
    std::string hg_dom_line;

    auto hg_doms_greater = new std::vector<std::vector<unsigned int>*>();
    auto hg_isdoms_greater = new std::vector<bool>();

    unsigned int absolute_rule_idx = 0;
    while(getline(hg_dom_greater_in, hg_dom_line)){
        if(hg_dom_line[0] == ')' || hg_dom_line[0] == '('){
            auto temp_doms = split(hg_dom_line, " ", false);
            hg_isdoms_greater->push_back(hg_dom_line[0] == '(');
            hg_doms_greater->push_back(new std::vector<unsigned int>());
            for(unsigned int i=1;i<temp_doms.size();i++){
                hg_doms_greater->at(hg_doms_greater->size()-1)->push_back((unsigned int)stoi(temp_doms[i]));
            }
        }
    }
    hg_dom_greater_in.close();

    auto hg_doms = new std::vector<std::vector<unsigned int>*>();
    auto hg_isdoms = new std::vector<bool>();

    while(getline(hg_dom_in, hg_dom_line)){
        if(hg_dom_line[0] == ')' || hg_dom_line[0] == '('){
            auto temp_doms = split(hg_dom_line, " ", false);
            hg_isdoms->push_back(hg_dom_line[0] == '(');
            hg_doms->push_back(new std::vector<unsigned int>());
            for(unsigned int i=1;i<temp_doms.size();i++){
                hg_doms->at(hg_doms->size()-1)->push_back((unsigned int)stoi(temp_doms[i]));
            }
        }
    }
    hg_dom_in.close();

    HeterGraph g(choice);

    std::string rules_path = (choice + "/"+choice+"-cod-global-rules.dat");
    std::ifstream rules_in;
    rules_in.open(rules_path);
    std::string rules_line;

    auto visited = new std::vector<std::vector<bool>*>();
    auto back_visited = new std::vector<std::vector<bool>*>();
    auto qp = new Pattern();

    unsigned int rule_count = 0;
    unsigned int dom_count = 0;
    double avg_relative_error = 0;

    getline(rules_in, rules_line);
    rules_in.close();

    double avg_time = 0.0;

    int state = -1;
    // state == 0: next int represents a variable rule;
    // state == 1: next int represents a instance rule;

    std::string::size_type temp_pos;
    int size = rules_line.size();

    for(unsigned int i = 0; i<size;i++) {
        temp_pos = rules_line.find(' ', i);
        if (temp_pos < size) {
            int sub = stoi(rules_line.substr(i, temp_pos - i));

            if (sub == -1) state = 0; // next int indicates a variable rule
            else if (sub == -2) qp->EDirect.push_back(1); // ->
            else if (sub == -3) qp->EDirect.push_back(-1); // <-
            else if (sub == -4) { // pop
                qp->EDirect.pop_back();
                qp->ETypes.pop_back();
                qp->NTypes.pop_back();
            } else if (sub == -5) state = 1; // next int indicates a instance rule
            else {
                if (state == 1) { // instance rule
                    qp->instance = sub;
                    while(visited->size()< qp->ETypes.size()+1){
                        visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                    }

                    unsigned int active_edges = 0;
                    double pre_time = avg_time;
                    double relative_error = effectiveness::COD_prop_global_cross_f1(qp, &g, std::stod(topr), visited, back_visited, method, hg_doms, hg_isdoms, hg_doms_greater, hg_isdoms_greater, dom_count, avg_time, active_edges);
                    double rule_time = avg_time - pre_time;
                    std::cout << "SCATTER_DATA: " << absolute_rule_idx++ << "," << active_edges << "," << (rule_time/1000000000.0) << std::endl;
                    if(relative_error >= -0.1){
                        rule_count++;
                        avg_relative_error += relative_error;
                    }

                    qp->instance = -1;
                    state = -1;
                } else if (state == 0) { // variable rule
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);
                    while(visited->size()< qp->ETypes.size()+1){
                        visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                    }
                    unsigned int active_edges = 0;
                    double pre_time = avg_time;
                    double relative_error = effectiveness::COD_prop_personalized_cross_precision(qp, &g, std::stod(topr), beta, visited, back_visited, method, hg_doms, hg_isdoms, hg_doms_greater, hg_isdoms_greater, dom_count, avg_time, active_edges);
                    double rule_time = avg_time - pre_time;
                    std::cout << "SCATTER_DATA: " << rule_count << "," << active_edges << "," << (rule_time/1000000000.0) << std::endl;
                    if(relative_error > -0.1){
                        rule_count++;
                        avg_relative_error += relative_error;
                    }
                    state = -1;
                } else {
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);
                }
            }
            i = temp_pos;
        }
    }
    qp->clear();

    for (auto &it : *visited) delete it;
    for (auto &it : *back_visited) delete it;
    delete visited;
    delete back_visited;

    avg_relative_error /= rule_count;
    avg_time /= rule_count;
    avg_time /= 1000000000;

    std::cout<<"~goodness:"<<avg_relative_error<<std::endl;
    std::cout<<"~time_per_rule:"<<avg_time<<" s"<<std::endl;
    std::cout<<"rule_count:"<<rule_count<<std::endl;
}

void Scalability_hg_greater_f1(const std::string &choice, double topr, const std::string &method, const std::string &len){
    HeterGraph g(choice);

    std::string rules_path = (choice + "/"+choice+"-cod-global-rules.dat"+len);
    std::ifstream rules_in;
    rules_in.open(rules_path);
    std::string rules_line;

    auto visited = new std::vector<std::vector<bool>*>();
    auto back_visited = new std::vector<std::vector<bool>*>();
    auto qp = new Pattern();

    getline(rules_in, rules_line);
    rules_in.close();

    unsigned int rule_count = 0;
    double avg_time = 0.0;

    int state = -1;
    // state == 0: next int represents a variable rule;
    // state == 1: next int represents a instance rule;

    std::string::size_type temp_pos;
    int size = rules_line.size();

    for(unsigned int i = 0; i<size;i++) {
        temp_pos = rules_line.find(' ', i);
        if (temp_pos < size) {
            int sub = stoi(rules_line.substr(i, temp_pos - i));

            if (sub == -1) state = 0; // next int indicates a variable rule
            else if (sub == -2) qp->EDirect.push_back(1); // ->
            else if (sub == -3) qp->EDirect.push_back(-1); // <-
            else if (sub == -4) { // pop
                qp->EDirect.pop_back();
                qp->ETypes.pop_back();
                qp->NTypes.pop_back();
            } else if (sub == -5) state = 1; // next int indicates a instance rule
            else {
                if (state == 1) { // instance rule
                    qp->instance = sub;
                    while(visited->size()< qp->ETypes.size()+1){
                        visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                    }
                    bool peer_found = effectiveness::COD_hg_global_greater_f1(qp, &g, topr, visited, back_visited, method, avg_time);
                    if(peer_found) rule_count++;
                    if(rule_count >= PATHCOUNT) break;

                    qp->instance = -1;
                    state = -1;
                } else if (state == 0) { // variable rule
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);
                    while(visited->size()< qp->ETypes.size()+1){
                        visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                    }
                    bool peer_found = effectiveness::COD_hg_global_greater_f1(qp, &g, topr, visited, back_visited, method, avg_time);
                    if(peer_found) rule_count++;
                    if(rule_count >= PATHCOUNT) break;

                    state = -1;
                } else {
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);
                }
            }
            i = temp_pos;
        }
    }
    qp->clear();

    for (auto &it : *visited) delete it;
    for (auto &it : *back_visited) delete it;
    delete visited;
    delete back_visited;

    avg_time /= rule_count;
    avg_time /= 1000000000;

    std::cout<<"~time_per_rule:"<<avg_time<<" s"<<std::endl;
    std::cout<<"rule_count:"<<rule_count<<std::endl;
}

void Effective_hg_global_greater_f1(const std::string &choice, double topr, const std::string &method){
    HeterGraph g(choice);

    std::string rules_path = (choice + "/"+choice+"-cod-global-rules.dat");
    std::ifstream rules_in;
    rules_in.open(rules_path);
    std::string rules_line;

    auto visited = new std::vector<std::vector<bool>*>();
    auto back_visited = new std::vector<std::vector<bool>*>();
    auto qp = new Pattern();

    getline(rules_in, rules_line);
    rules_in.close();

    unsigned int rule_count = 0;
    double avg_time = 0.0;

    int state = -1;
    // state == 0: next int represents a variable rule;
    // state == 1: next int represents a instance rule;

    std::string::size_type temp_pos;
    int size = rules_line.size();

    for(unsigned int i = 0; i<size;i++) {
        temp_pos = rules_line.find(' ', i);
        if (temp_pos < size) {
                int sub = stoi(rules_line.substr(i, temp_pos - i));

                if (sub == -1) state = 0; // next int indicates a variable rule
                else if (sub == -2) qp->EDirect.push_back(1); // ->
                else if (sub == -3) qp->EDirect.push_back(-1); // <-
                else if (sub == -4) { // pop
                    qp->EDirect.pop_back();
                    qp->ETypes.pop_back();
                    qp->NTypes.pop_back();
                } else if (sub == -5) state = 1; // next int indicates a instance rule
                else {
                    if (state == 1) { // instance rule
                        qp->instance = sub;
                        while(visited->size()< qp->ETypes.size()+1){
                            visited->push_back(new std::vector<bool>(g.NT.size(), false));
                            back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        }
                        bool peer_found = effectiveness::COD_hg_global_greater_f1(qp, &g, topr, visited, back_visited, method, avg_time);
                        if(peer_found) rule_count++;

                        qp->instance = -1;
                        state = -1;
                    } else if (state == 0) { // variable rule
                        qp->ETypes.push_back(sub);
                        qp->NTypes.push_back(-1);
                        while(visited->size()< qp->ETypes.size()+1){
                            visited->push_back(new std::vector<bool>(g.NT.size(), false));
                            back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        }
                        bool peer_found = effectiveness::COD_hg_global_greater_f1(qp, &g, topr, visited, back_visited, method, avg_time);
                        if(peer_found) rule_count++;

                        state = -1;
                    } else {
                        qp->ETypes.push_back(sub);
                        qp->NTypes.push_back(-1);
                    }
                }
                i = temp_pos;
        }
    }
    qp->clear();

    for (auto &it : *visited) delete it;
    for (auto &it : *back_visited) delete it;
    delete visited;
    delete back_visited;

    avg_time /= rule_count;
    avg_time /= 1000000000;

    std::cout<<"~time_per_rule:"<<avg_time<<" s"<<std::endl;
    std::cout<<"rule_count:"<<rule_count<<std::endl;
}

void Effective_hg_global_f1_by_union(const std::string &choice, double topr, const std::string &method){
    HeterGraph g(choice);

    std::string rules_path = (choice + "/"+choice+"-cod-global-rules.dat");
    std::ifstream rules_in;
    rules_in.open(rules_path);
    std::string rules_line;

    auto visited = new std::vector<std::vector<bool>*>();
    auto back_visited = new std::vector<std::vector<bool>*>();
    auto qp = new Pattern();

    getline(rules_in, rules_line);
    rules_in.close();

    unsigned int rule_count = 0;
    double avg_time = 0.0;

    int state = -1;
    // state == 0: next int represents a variable rule;
    // state == 1: next int represents a instance rule;

    std::string::size_type temp_pos;
    int size = rules_line.size();

    for(unsigned int i = 0; i<size;i++) {
        temp_pos = rules_line.find(' ', i);
        if (temp_pos < size) {
            int sub = stoi(rules_line.substr(i, temp_pos - i));

            if (sub == -1) state = 0; // next int indicates a variable rule
            else if (sub == -2) qp->EDirect.push_back(1); // ->
            else if (sub == -3) qp->EDirect.push_back(-1); // <-
            else if (sub == -4) { // pop
                qp->EDirect.pop_back();
                qp->ETypes.pop_back();
                qp->NTypes.pop_back();
            } else if (sub == -5) state = 1; // next int indicates a instance rule
            else {
                if (state == 1) { // instance rule
                    qp->instance = sub;
                    while(visited->size()< qp->ETypes.size()+1){
                        visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                    }
                    bool peer_found = effectiveness::COD_hg_global_f1_by_union(qp, &g, topr, visited, back_visited, method, avg_time);
                    if(peer_found) rule_count++;

                    qp->instance = -1;
                    state = -1;
                } else if (state == 0) { // variable rule
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);
                    while(visited->size()< qp->ETypes.size()+1){
                        visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                    }
                    bool peer_found = effectiveness::COD_hg_global_f1_by_union(qp, &g, topr, visited, back_visited, method, avg_time);
                    if(peer_found) rule_count++;

                    state = -1;
                } else {
                    qp->ETypes.push_back(sub);
                    qp->NTypes.push_back(-1);
                }
            }
            i = temp_pos;
        }
    }

    qp->clear();
    for (auto &it : *visited) delete it;
    for (auto &it : *back_visited) delete it;
    delete visited;
    delete back_visited;

    avg_time /= rule_count;
    avg_time /= 1000000000;

    std::cout<<"~time_per_rule:"<<avg_time<<" s"<<std::endl;
    std::cout<<"rule_count:"<<rule_count<<std::endl;
}

void Effective_hg_global_f1(const std::string &choice, double topr, const std::string &method){
    HeterGraph g(choice);

    std::string rules_path = (choice + "/"+choice+"-cod-global-rules.dat");
    std::ifstream rules_in;
    rules_in.open(rules_path);
    std::string rules_line;

    auto visited = new std::vector<std::vector<bool>*>();
    auto back_visited = new std::vector<std::vector<bool>*>();
    auto qp = new Pattern();

    getline(rules_in, rules_line);
    rules_in.close();

    unsigned int rule_count = 0;
    double avg_time = 0.0;

    int state = -1;
    // state == 0: next int represents a variable rule;
    // state == 1: next int represents a instance rule;

    std::string::size_type temp_pos;
    int size = rules_line.size();

    for(unsigned int i = 0; i<size;i++) {
        temp_pos = rules_line.find(' ', i);
        if (temp_pos < size) {
                int sub = stoi(rules_line.substr(i, temp_pos - i));

                if (sub == -1) state = 0; // next int indicates a variable rule
                else if (sub == -2) qp->EDirect.push_back(1); // ->
                else if (sub == -3) qp->EDirect.push_back(-1); // <-
                else if (sub == -4) { // pop
                    qp->EDirect.pop_back();
                    qp->ETypes.pop_back();
                    qp->NTypes.pop_back();
                } else if (sub == -5) state = 1; // next int indicates a instance rule
                else {
                    if (state == 1) { // instance rule
                        qp->instance = sub;
                        while(visited->size()< qp->ETypes.size()+1){
                            visited->push_back(new std::vector<bool>(g.NT.size(), false));
                            back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        }
                        bool peer_found = effectiveness::COD_hg_global_f1(qp, &g, topr, visited, back_visited, method, avg_time);
                        if(peer_found) rule_count++;

                        qp->instance = -1;
                        state = -1;
                    } else if (state == 0) { // variable rule
                        qp->ETypes.push_back(sub);
                        qp->NTypes.push_back(-1);
                        while(visited->size()< qp->ETypes.size()+1){
                            visited->push_back(new std::vector<bool>(g.NT.size(), false));
                            back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        }
                        bool peer_found = effectiveness::COD_hg_global_f1(qp, &g, topr, visited, back_visited, method, avg_time);
                        if(peer_found) rule_count++;

                        state = -1;
                    } else {
                        qp->ETypes.push_back(sub);
                        qp->NTypes.push_back(-1);
                    }
                }
                i = temp_pos;
        }
    }

    qp->clear();
    for (auto &it : *visited) delete it;
    for (auto &it : *back_visited) delete it;
    delete visited;
    delete back_visited;

    avg_time /= rule_count;
    avg_time /= 1000000000;

    std::cout<<"~time_per_rule:"<<avg_time<<" s"<<std::endl;
    std::cout<<"rule_count:"<<rule_count<<std::endl;
}

void Effective_epsilon(const std::string &choice, const std::string &topr){
    HeterGraph g(choice);

    std::string rules_path = (choice + "/cod-rules_"+choice+".limit");
    std::ifstream rules_in;
    rules_in.open(rules_path);
    std::string rules_line;
    getline(rules_in, rules_line);
    rules_in.close();

    auto visited = new std::vector<std::vector<bool>*>();
    auto back_visited = new std::vector<std::vector<bool>*>();
    auto qp = new Pattern();

    int state = -1;
    std::string::size_type temp_pos;
    int size = rules_line.size();

    unsigned int rule_count = 0;

    // For each centrality: degree and h-index
    for (const auto &centrality : {"df1", "hf1"}) {
        double total_eps = 0;
        unsigned int n_rules = 0;

        // Parse rule line (same parser as other functions)
        qp->clear();
        for (auto &it : *visited) delete it;
        for (auto &it : *back_visited) delete it;
        visited->clear();
        back_visited->clear();
        state = -1;

        for(unsigned int i = 0; i < (unsigned int)size; i++) {
            temp_pos = rules_line.find(' ', i);
            if (temp_pos < (unsigned int)size) {
                int sub = stoi(rules_line.substr(i, temp_pos - i));

                if (sub == -1) state = 0;
                else if (sub == -2) qp->EDirect.push_back(1);
                else if (sub == -3) qp->EDirect.push_back(-1);
                else if (sub == -4) {
                    qp->EDirect.pop_back();
                    qp->ETypes.pop_back();
                    qp->NTypes.pop_back();
                } else if (sub == -5) state = 1;
                else {
                    if (state == 1) {
                        qp->instance = sub;
                        while(visited->size() < qp->ETypes.size()+1){
                            visited->push_back(new std::vector<bool>(g.NT.size(), false));
                            back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        }
                        double eps = effectiveness::COD_epsilon(qp, &g, std::stod(topr), visited, back_visited, centrality);
                        if (eps >= 0) { total_eps += eps; n_rules++; }
                        qp->instance = -1;
                        state = -1;
                    } else if (state == 0) {
                        qp->ETypes.push_back(sub);
                        qp->NTypes.push_back(-1);
                        while(visited->size() < qp->ETypes.size()+1){
                            visited->push_back(new std::vector<bool>(g.NT.size(), false));
                            back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        }
                        double eps = effectiveness::COD_epsilon(qp, &g, std::stod(topr), visited, back_visited, centrality);
                        if (eps >= 0) { total_eps += eps; n_rules++; }
                        state = -1;
                    } else {
                        qp->ETypes.push_back(sub);
                        qp->NTypes.push_back(-1);
                    }
                }
                i = temp_pos;
            }
        }
        qp->clear();

        double avg_eps = n_rules > 0 ? total_eps / n_rules : -1;
        std::cout << "EPSILON_" << centrality << ":" << avg_eps << std::endl;
        std::cout << "EPSILON_" << centrality << "_rules:" << n_rules << std::endl;
    }

    for (auto &it : *visited) delete it;
    for (auto &it : *back_visited) delete it;
    delete visited;
    delete back_visited;
    delete qp;
}

void Effective_hg_stats(const std::string &choice){
    HeterGraph g(choice);
    std::string qnodes_path = (choice + "/qnodes_"+choice+".dat");
    std::ifstream qnodes_in;
    qnodes_in.open(qnodes_path);
    std::vector<unsigned int> qnodes;
    std::string qnode_line;
    while(getline(qnodes_in, qnode_line)){
        auto qnode = (unsigned int)(stoi(qnode_line));
        qnodes.push_back(qnode);
    }
    qnodes_in.close();

    std::string rules_path = (choice + "/cod-rules_"+choice+".limit");
    std::ifstream rules_in;
    rules_in.open(rules_path);
    std::string rules_line;

    auto visited = new std::vector<std::vector<bool>*>();
    auto back_visited = new std::vector<std::vector<bool>*>();
    auto qp = new Pattern();

    int query_count = -1;
    std::vector<double> res;
    for(unsigned int i=0;i < 4;i++) res.push_back(0.0);

    while(getline(rules_in, rules_line)){
        query_count++;
        unsigned int qnode = qnodes[query_count];
        int state = -1;
        // state == 0: next int represents a variable rule;
        // state == 1: next int represents a instance rule;

        std::string::size_type temp_pos;
        int size = rules_line.size();

        std::cout<<"@ qn:"<<qnode<<std::endl;

        std::vector<double> qn_res;
        for(unsigned int i=0;i<res.size();i++) qn_res.push_back(0.0);
        int qn_rule_count = 0;

        for(unsigned int i = 0; i<size;i++) {
            temp_pos = rules_line.find(' ', i);
            if (temp_pos < size) {
                int sub = stoi(rules_line.substr(i, temp_pos - i));

                if (sub == -1) state = 0; // next int indicates a variable rule
                else if (sub == -2) qp->EDirect.push_back(1); // ->
                else if (sub == -3) qp->EDirect.push_back(-1); // <-
                else if (sub == -4) { // pop
                    qp->EDirect.pop_back();
                    qp->ETypes.pop_back();
                    qp->NTypes.pop_back();
                } else if (sub == -5) state = 1; // next int indicates a instance rule
                else {
                    if (state == 1) { // instance rule
                        qp->instance = sub;
                        while(visited->size()< qp->ETypes.size()+1){
                            visited->push_back(new std::vector<bool>(g.NT.size(), false));
                            back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        }
                        auto temp_res = effectiveness::COD_hg_statistics(qp, &g, qnode, visited, back_visited);
                        if(temp_res != nullptr) {
                            for (unsigned int z = 0; z < res.size(); z++) qn_res[z] += temp_res->at(z);
                            qn_rule_count++;
                            delete temp_res;
                        }
                        qp->instance = -1;
                        state = -1;
                    } else if (state == 0) { // variable rule
                        qp->ETypes.push_back(sub);
                        qp->NTypes.push_back(-1);
                        while(visited->size()< qp->ETypes.size()+1){
                            visited->push_back(new std::vector<bool>(g.NT.size(), false));
                            back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
                        }
                        auto temp_res = effectiveness::COD_hg_statistics(qp, &g, qnode, visited, back_visited);
                        if(temp_res != nullptr) {
                            for (unsigned int z = 0; z < res.size(); z++) qn_res[z] += temp_res->at(z);
                            qn_rule_count++;
                            delete temp_res;
                        }
                        state = -1;
                    } else {
                        qp->ETypes.push_back(sub);
                        qp->NTypes.push_back(-1);
                    }
                }
                i = temp_pos;
            }
        }
        qp->clear();
        for(unsigned int i=0;i<res.size();i++) {
            qn_res[i] /= qn_rule_count;
            res[i] += qn_res[i];
        }
        std::cout<<"q_dens:"<<qn_res[0]<<std::endl;
        std::cout<<"q_d_same:"<<qn_res[1]<<std::endl;
        std::cout<<"q_h_same:"<<qn_res[2]<<std::endl;
        std::cout<<"q_|peer|:"<<qn_res[3]<<std::endl;
    }
    rules_in.close();

    for (auto &it : *visited) delete it;
    for (auto &it : *back_visited) delete it;
    delete visited;
    delete back_visited;

    for(unsigned int i=0;i<res.size();i++) res[i] /= qnodes.size();
    std::cout<<"~dens:"<<res[0]<<std::endl;
    std::cout<<"~d_same:"<<res[1]<<std::endl;
    std::cout<<"~h_same:"<<res[2]<<std::endl;
    std::cout<<"~|peer|:"<<res[3]<<std::endl;
}



// --- HELPER FUNCTION: To parse the rule file ---
Pattern* parse_first_rule_from_file(const std::string& rule_file) {
    std::ifstream rules_in(rule_file);
    if (!rules_in.is_open()) {
        throw std::runtime_error("Could not open rule file: " + rule_file);
    }
    
    std::string rules_line;
    getline(rules_in, rules_line);
    rules_in.close();

    auto qp = new Pattern();
    int state = -1;
    std::string::size_type temp_pos;
    int size = rules_line.size();

    for (unsigned int i = 0; i < size; i++) {
        temp_pos = rules_line.find(' ', i);
        if (temp_pos >= std::string::npos) break; // Use std::string::npos

        int sub = stoi(rules_line.substr(i, temp_pos - i));
        
        if (sub == -1) state = 0; // variable rule
        else if (sub == -2) qp->EDirect.push_back(1); // ->
        else if (sub == -3) qp->EDirect.push_back(-1); // <-
        else if (sub == -4) { // pop
            qp->EDirect.pop_back();
            qp->ETypes.pop_back();
            qp->NTypes.pop_back();
        } else if (sub == -5) state = 1; // instance rule
        else {
            if (state == 1) { // instance
                qp->instance = sub;
                // Found the end of the rule, break
                break; 
            } else if (state == 0) { // variable
                qp->ETypes.push_back(sub);
                qp->NTypes.push_back(-1);
                // Found the end of the rule, break
                break; 
            } else {
                qp->ETypes.push_back(sub);
                qp->NTypes.push_back(-1);
            }
        }
        i = temp_pos;
    }
    
    if (qp->ETypes.empty()) {
        delete qp;
        throw std::runtime_error("Failed to parse any rule from file.");
    }

    qp->print(); // Print the pattern to confirm
    return qp;
}


// --- METHOD 1: MATERIALIZATION ---
void run_materialization(const std::string& dataset, const std::string& rule_file, const std::string& output_file) {
    
    // 1. Setup: Load graph and parse rule
    HeterGraph g(dataset);
    Pattern* qp = parse_first_rule_from_file(rule_file);

    // 2. Setup: Create 'visited' vectors (same as original code)
    auto visited = new std::vector<std::vector<bool>*>();
    auto back_visited = new std::vector<std::vector<bool>*>();
    for(unsigned int i=0; i <= qp->ETypes.size(); ++i) {
        visited->push_back(new std::vector<bool>(g.NT.size(), false));
        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
    }

    // 3. Setup: Find peers and active nodes (same as original code)
    auto t_start = std::chrono::steady_clock::now();

    auto frontiers = new std::vector<unsigned int>();
    auto peers = new std::vector<unsigned int>();
    Peers(qp, &g, frontiers, peers, visited);

    std::vector<std::set<unsigned int>*>* active = ActiveMidNodes(peers, frontiers, qp, &g, visited, back_visited);
    auto ractive = new std::vector<std::vector<unsigned int>*>();
    for (unsigned int l = 0; l <= qp->ETypes.size(); l++) ractive->push_back(new std::vector<unsigned int>(g.NT.size(), 0));
    for (unsigned int n = 0; n < g.NT.size(); n++) for (unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;

    // 4. *** RUN THE EXTRACTED LOGIC ***
    unsigned int meta_layer = qp->ETypes.size();
    if(qp->instance != -1) meta_layer = qp->ETypes.size() - 1;

    auto hidden_graph = hidden_graph_construction(meta_layer, qp, &g, visited, ractive);

    auto t_end = std::chrono::steady_clock::now();
    double algo_seconds = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "time:" << algo_seconds << std::endl;

    // 5. Write to output file
    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        throw std::runtime_error("Could not open output file: " + output_file);
    }
    
    for (unsigned int node_id = 0; node_id < hidden_graph->size(); ++node_id) {
        if (!ractive->at(0)->at(node_id)) continue; // Only write peer nodes
        if (hidden_graph->at(node_id)->empty()) continue;

        out_file << node_id; // Source node
        for (unsigned int neighbor : *hidden_graph->at(node_id)) {
            out_file << " " << neighbor; // Neighbors
        }
        out_file << "\n";
    }
    out_file.close();

    // 6. Cleanup (very important!)
    delete qp;
    delete frontiers;
    delete peers;
    for (auto &it : *visited) delete it;
    for (auto &it : *back_visited) delete it;
    delete visited;
    delete back_visited;
    for (auto &it : *active) delete it;
    delete active;
    for (auto &it : *ractive) delete it;
    delete ractive;
    for (auto &it : *hidden_graph) delete it;
    delete hidden_graph;
}


// --- METHOD 2: SKETCH PROPAGATION & SAMPLING ---
void run_sketch_sampling(const std::string& dataset, const std::string& rule_file, const std::string& output_file) {
    
    // 1. Setup: Load graph and parse rule
    HeterGraph g(dataset);
    Pattern* qp = parse_first_rule_from_file(rule_file);

    // 2. Setup: 'visited' vectors
    auto visited = new std::vector<std::vector<bool>*>();
    auto back_visited = new std::vector<std::vector<bool>*>();
    for(unsigned int i=0; i <= qp->ETypes.size(); ++i) {
        visited->push_back(new std::vector<bool>(g.NT.size(), false));
        back_visited->push_back(new std::vector<bool>(g.NT.size(), false));
    }

    // 3. Setup: Find peers and active nodes (same as original code)
    auto t_start = std::chrono::steady_clock::now();

    auto frontiers = new std::vector<unsigned int>();
    auto peers = new std::vector<unsigned int>();
    Peers(qp, &g, frontiers, peers, visited);

    std::vector<std::set<unsigned int>*>* active = ActiveMidNodes(peers, frontiers, qp, &g, visited, back_visited);
    auto ractive = new std::vector<std::vector<unsigned int>*>();
    for (unsigned int l = 0; l <= qp->ETypes.size(); l++) ractive->push_back(new std::vector<unsigned int>(g.NT.size(), 0));
    for (unsigned int n = 0; n < g.NT.size(); n++) for (unsigned int l : *active->at(n)) ractive->at(l)->at(n) = 1;

    auto t_peers = std::chrono::steady_clock::now();
    std::cout << "[sketch-timing] peers+active: "
              << std::chrono::duration<double>(t_peers - t_start).count() << "s" << std::endl;

    // 4. Setup: Build hidden edges and synopses cache (same as prop::COD)
    auto synopses_cache = new std::vector<std::vector<jsy::Synopse>*>();
    for(unsigned int l = 0; l <= qp->ETypes.size(); l++){
        synopses_cache->push_back(new std::vector<jsy::Synopse>());
        for(unsigned int p=0; p < g.NT.size(); p++) {
            if(ractive->at(l)->at(p) > 0){
                jsy::Synopse s; jsy::init_synopse(&s, p);
                synopses_cache->at(l)->push_back(s);
                ractive->at(l)->at(p) = synopses_cache->at(l)->size();
            }
        }
    }

    auto t_synalloc = std::chrono::steady_clock::now();
    std::cout << "[sketch-timing] synopse alloc: "
              << std::chrono::duration<double>(t_synalloc - t_peers).count() << "s" << std::endl;
    for(unsigned int l = 0; l <= qp->ETypes.size(); l++)
        std::cout << "[sketch-timing]   layer " << l << " synopses: " << synopses_cache->at(l)->size() << std::endl;

    // Build hidden_edges with edge-type filtering.
    // g.EL[p] contains ALL edge types. g.ET[p] is parallel — ET[p][i] holds
    // the type IDs for the i-th edge in EL[p]. We check that the edge type
    // matches the metapath's expected type at each layer.
    auto hidden_edges = new std::vector<HiddenEdge>();
    for(unsigned int l=0; l < qp->ETypes.size(); l++){
        unsigned int etype = qp->ETypes[l];
        for(unsigned int p=0; p < g.NT.size(); p++){
            if(ractive->at(l)->at(p) > 0){
                int s_idx = ractive->at(l)->at(p) - 1;
                if (qp->EDirect[l] == 1) {
                    for (unsigned int i = 0; i < g.EL[p]->size(); i++) {
                        unsigned int nbr = g.EL[p]->at(i);
                        // Check edge type matches metapath
                        bool type_ok = false;
                        for (unsigned int t : *g.ET[p]->at(i)) {
                            if (t == etype) { type_ok = true; break; }
                        }
                        if (type_ok && ractive->at(l + 1)->at(nbr) > 0) {
                            HiddenEdge he{.s=(unsigned int)s_idx, .t=(ractive->at(l+1)->at(nbr) - 1), .l=l};
                            hidden_edges->push_back(he);
                        }
                    }
                }
                else {
                    for (unsigned int i = 0; i < g.rEL[p]->size(); i++) {
                        unsigned int nbr = g.rEL[p]->at(i);
                        bool type_ok = false;
                        for (unsigned int t : *g.rET[p]->at(i)) {
                            if (t == etype) { type_ok = true; break; }
                        }
                        if (type_ok && ractive->at(l + 1)->at(nbr) > 0) {
                            HiddenEdge he{.s=(unsigned int)s_idx, .t=(ractive->at(l+1)->at(nbr) - 1), .l=l};
                            hidden_edges->push_back(he);
                        }
                    }
                }
            }
        }
    }

    auto t_hedges = std::chrono::steady_clock::now();
    std::cout << "[sketch-timing] hidden_edges: "
              << std::chrono::duration<double>(t_hedges - t_synalloc).count() << "s  count="
              << hidden_edges->size() << std::endl;

    // 5. *** RUN THE EXTRACTED LOGIC ***
    unsigned int meta_layer = qp->ETypes.size();
    if(qp->instance != -1) meta_layer = qp->ETypes.size() - 1;

    std::cout << "[sketch-timing] K=" << K << " L=" << L << " meta_layer=" << meta_layer << std::endl;

    // Propagate sketches and get the random-to-peer mappings
    auto rand2ps = jsy::gnn_synopses(meta_layer, hidden_edges, synopses_cache);

    auto t_prop = std::chrono::steady_clock::now();
    std::cout << "[sketch-timing] gnn_synopses: "
              << std::chrono::duration<double>(t_prop - t_hedges).count() << "s" << std::endl;

    auto t_end = std::chrono::steady_clock::now();
    double algo_seconds = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "time:" << algo_seconds << std::endl;

    // 6. Sample from final sketches and write to file
    std::ofstream out_file(output_file);
    if (!out_file.is_open()) {
        throw std::runtime_error("Could not open output file: " + output_file);
    }

    // 6. Sample from ALL L sketches and write to SEPARATE files
    std::mt19937 generator(SEED);
    
    // Get the base filename (e.g., "graph.txt") to create "graph_0.txt", "graph_1.txt"
    // Helper lambda to insert index before extension
    auto make_filename = [&](const std::string& base, int idx) {
        size_t dot_pos = base.find_last_of(".");
        if (dot_pos == std::string::npos) return base + "_" + std::to_string(idx);
        return base.substr(0, dot_pos) + "_" + std::to_string(idx) + base.substr(dot_pos);
    };

    // Read from the LAST layer of the forward pass (meta_layer), NOT layer 0.
    // Layer 0 after backward pass contains a superset (2*meta_layer hops instead
    // of meta_layer hops) because the backward pass propagates accumulated hashes
    // through shared intermediate nodes, creating false positives.
    auto final_synopses = synopses_cache->at(meta_layer);

    // --- LOOP THROUGH ALL LAYERS (0 to L-1) ---
    for (unsigned int l = 0; l < L; l++) {
        
        string current_outfile = make_filename(output_file, l);
        std::ofstream out_file(current_outfile);
        
        if (!out_file.is_open()) {
            cerr << "Warning: Could not open " << current_outfile << endl;
            continue;
        }

        cout << "Materializing Sketch Layer " << l << " to " << current_outfile << "..." << endl;

        for (auto &synopse : *final_synopses) {
            unsigned int peer_node = synopse.p;
            
            // *** CRITICAL CHANGE: Pass 'l' instead of '0' ***
            auto neighbors = jsy::sample(&synopse, K, generator, l, rand2ps);

            if (neighbors->empty()) {
                delete neighbors;
                continue;
            }
            out_file << peer_node;
            for (unsigned int neighbor : *neighbors) {
                out_file << " " << neighbor;
            }
            out_file << "\n";
            delete neighbors;
        }
        out_file.close();
    }

    auto t_write = std::chrono::steady_clock::now();
    std::cout << "[sketch-timing] sampling+write: "
              << std::chrono::duration<double>(t_write - t_prop).count() << "s" << std::endl;
    std::cout << "[sketch-timing] TOTAL: "
              << std::chrono::duration<double>(t_write - t_start).count() << "s" << std::endl;

    // 7. Cleanup
    delete qp;
    delete frontiers;
    delete peers;
    for (auto &it : *visited) delete it;
    for (auto &it : *back_visited) delete it;
    delete visited;
    delete back_visited;
    for (auto &it : *active) delete it;
    delete active;
    for (auto &it : *ractive) delete it;
    delete ractive;
    for (auto &it : *synopses_cache) delete it;
    delete synopses_cache;
    delete hidden_edges;
    for (auto &it : *rand2ps) delete it;
    delete rand2ps;
}