#ifndef HIN
#define HIN

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <queue>
#include <fstream>
#include <set>
#include <math.h>
#include <sstream>

struct Guard{
unsigned int etype;
unsigned int begin;
unsigned int end;
};

class HeterGraph{
public:
    std::vector<std::vector<unsigned int>* > NT;  // node types
    std::vector<std::vector<unsigned int>* > EL;  // edge lists
    std::vector<std::vector<std::vector<unsigned int>* >* > ET;  // edge types
    std::vector<std::vector<unsigned int>* > rEL;  // reverse edge lists
    std::vector<std::vector<std::vector<unsigned int>* >* > rET;  // reverse edge types

    std::vector<std::vector<std::pair<unsigned int, unsigned int>>*> ETLG; // Edge Type List node-Guards
    std::vector<std::vector<unsigned int>* > ETL;  // Edge Type Lists
    std::vector<std::vector<std::pair<unsigned int, unsigned int>>*> rETLG;
    std::vector<std::vector<unsigned int>* > rETL;

    std::vector<std::vector<Guard>*> NGET; // Node-Guard Edge-Type
    std::vector<std::vector<Guard>*> rNGET; // reverse Node-Guard Edge-Type

    explicit HeterGraph(const std::string &choice){
        std::string meta_file = choice + "/meta.dat";
        std::ifstream meta_in;
        meta_in.open(meta_file);
        std::string meta_line;
        getline(meta_in, meta_line);
        meta_in.close();
        unsigned int vnum = (unsigned int)stoi(meta_line.substr(17, meta_line.length()-17));

        std::string node_file = choice + "/node.dat";
        std::ifstream node_in;
        node_in.open(node_file);
        std::string node_line;

        int NodeId = 0;
        this->NT.reserve(vnum);
        while(getline(node_in, node_line)){
            std::stringstream ss0(node_line);
            std::string types_str;
            int temp_count = 0;
            while(std::getline(ss0, types_str, '\t')){
                temp_count++;
                if(temp_count==3) break;
            }
            std::stringstream ss1(types_str);
            std::string type;
            this->NT.push_back(new std::vector<unsigned int>());
            while(std::getline(ss1, type, ',')) this->NT[NodeId]->push_back((unsigned int)stoi(type));
            NodeId++;
        }
        node_in.close();
        // read node.dat to get node types

        std::string link_file = choice + "/link.dat";
        std::ifstream link_in;
        link_in.open(link_file);
        std::string link_line;

        for(int i=0;i<vnum;i++){
            this->EL.push_back(new std::vector<unsigned int>());
            this->ET.push_back(new std::vector<std::vector<unsigned int>* >());
            this->rEL.push_back(new std::vector<unsigned int>());
            this->rET.push_back(new std::vector<std::vector<unsigned int>*>());
        }
        while(getline(link_in, link_line)){
            std::stringstream ss0(link_line);
            std::string linktype_str;
            int temp_count = 0;

            unsigned int s, t;
            while(std::getline(ss0, linktype_str, '\t')){
                temp_count++;
                if(temp_count == 1) s = (unsigned int)stoi(linktype_str);
                else if(temp_count == 2) t = (unsigned int)stoi(linktype_str);
                else if(temp_count == 3) break;
            }
            this->EL[s]->push_back(t);
            this->rEL[t]->push_back(s);
            this->ET[s]->push_back(new std::vector<unsigned int>());
            this->rET[t]->push_back(new std::vector<unsigned int>());

            std::stringstream ss1(linktype_str);
            std::string type;
            while(std::getline(ss1, type, ',')) {
                this->ET[s]->at(this->ET[s]->size() -1)->push_back((unsigned int) stoi(type));
                this->rET[t]->at(this->rET[t]->size()-1)->push_back((unsigned int) stoi(type));
            }}
        link_in.close();

        for(unsigned int n=0;n<this->EL.size();n++){
            for(unsigned i=0;i<this->EL[n]->size();i++){
                unsigned int nbr = EL[n]->at(i);
                for(unsigned int j=0;j<this->ET[n]->at(i)->size();j++){
                    unsigned int et = this->ET[n]->at(i)->at(j);
                    while(this->ETL.size() <= et){
                        this->ETL.push_back(new std::vector<unsigned int>());
                        this->ETLG.push_back(new std::vector<std::pair<unsigned int, unsigned int>>());
                    }
                    if(this->ETLG[et]->empty() || this->ETLG[et]->at(this->ETLG[et]->size()-1).first!=n)
                        this->ETLG[et]->push_back(std::pair<unsigned int, unsigned int>(n, this->ETL[et]->size()));
                    this->ETL[et]->push_back(nbr);
                }
            }
        }

        for(unsigned int n=0;n<this->rEL.size();n++){
            for(unsigned i=0;i<this->rEL[n]->size();i++){
                unsigned int nbr = rEL[n]->at(i);
                for(unsigned int j=0;j<this->rET[n]->at(i)->size();j++){
                    unsigned int et = this->rET[n]->at(i)->at(j);
                    while(this->rETL.size() <= et){
                        this->rETL.push_back(new std::vector<unsigned int>());
                        this->rETLG.push_back(new std::vector<std::pair<unsigned int, unsigned int>>());
                    }
                    if(this->rETLG[et]->empty() || this->rETLG[et]->at(this->rETLG[et]->size()-1).first!=n)
                        this->rETLG[et]->push_back(std::pair<unsigned int, unsigned int>(n, this->rETL[et]->size()));
                    this->rETL[et]->push_back(nbr);
                }
            }
        }

        for(unsigned int n=0;n<this->EL.size();n++){
            NGET.push_back(new std::vector<Guard>());
            rNGET.push_back(new std::vector<Guard>());
        }
        for(unsigned int et=0;et<this->ETLG.size();et++){
            for(unsigned n=0;n<this->ETLG[et]->size();n++){
                Guard et_g;
                et_g.etype = et;
                et_g.begin = this->ETLG[et]->at(n).second;
                if(n < this->ETLG[et]->size()-1) et_g.end = this->ETLG[et]->at(n+1).second;
                else et_g.end = this->ETL[et]->size();
                NGET[this->ETLG[et]->at(n).first]->push_back(et_g);
            }
        }

        for(unsigned int et=0;et<this->rETLG.size();et++){
            for(unsigned n=0;n<this->rETLG[et]->size();n++){
                Guard et_g;
                et_g.etype = et;
                et_g.begin = this->rETLG[et]->at(n).second;
                if(n < this->rETLG[et]->size()-1) et_g.end = this->rETLG[et]->at(n+1).second;
                else et_g.end = this->rETL[et]->size();
                rNGET[this->rETLG[et]->at(n).first]->push_back(et_g);
            }
        }
    }
};

#endif