#include "ReadAnyBURLRules.cpp" 
#include "param.h"
#include <string>
#include <iostream>
#include <vector>
#include <sys/stat.h> // For mkdir

using namespace std;

// ==========================================
// FORWARD DECLARATIONS & HELPER FUNCTIONS
// ==========================================

// Forward declarations for existing preprocessing functions
void run_materialization(const string& dataset, const string& rule_file, const string& output_file);
void run_sketch_sampling(const string& dataset, const string& rule_file, const string& output_file);
void Effective_epsilon(const string& dataset, const string& topr);

// // Helper to create directories (Linux/Mac compatible)
// void ensure_directory(const string& path) {
//     string cmd = "mkdir -p " + path;
//     system(cmd.c_str());
// }

// Debug helper to parse rules
void debug_rule_parsing(const string& rule_file) {
    try {
        Pattern* qp = parse_first_rule_from_file(rule_file);
        cout << "DEBUG_PARSED_SEQUENCE:";
        for (int et : qp->ETypes) cout << " " << et;
        cout << endl;
        
        cout << "DEBUG_DIRECTION_SEQUENCE:";
        for (int d : qp->EDirect) cout << " " << d;
        cout << endl;

        delete qp;
    } catch (const exception& e) {
        cerr << "PARSER_ERROR: " << e.what() << endl;
        exit(1);
    }
}

// Dump graph structure to file
void dump_graph(const string& dataset, const string& output_file) {
    HeterGraph g(dataset);
    std::ofstream out(output_file);
    if (!out.is_open()) {
        std::cerr << "Failed to open dump file." << std::endl;
        exit(1);
    }

    for (unsigned int u = 0; u < g.NT.size(); u++) {
        if (g.EL[u]->empty()) continue;
        for (size_t i = 0; i < g.EL[u]->size(); i++) {
            unsigned int v = g.EL[u]->at(i);
            if (i < g.ET[u]->size()) {
                for (unsigned int type_id : *g.ET[u]->at(i)) {
                    out << u << "\t" << v << "\t" << type_id << "\n";
                }
            }
        }
    }
    out.close();
    std::cout << "Dumped graph structure to " << output_file << std::endl;
}

// ==========================================
// MAIN ENTRY POINT
// ==========================================

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: ./graph_prep <task> [args...]" << endl;
        cerr << "Tasks: materialize, sketch, ExactD, ExactD+, GloD, PerD, scale..." << endl;
        return 1;
    }

    string task = argv[1];

    try {
        // ---------------------------------------------------------
        // PART 1: Data Generation & Preprocessing (New Logic)
        // ---------------------------------------------------------
        if (task == "materialize") {
            if (argc < 5) { cerr << "Usage: materialize <dataset> <rule_file> <output>" << endl; return 1; }
            run_materialization(argv[2], argv[3], argv[4]);
        } 
        else if (task == "sketch") {
            if (argc < 6) { cerr << "Usage: sketch <dataset> <rule_file> <output> <K> [L]" << endl; return 1; }
            K = std::stoul(argv[5]);
            if (argc >= 7) L = std::stoul(argv[6]);
            if (argc >= 8) SEED = std::stoul(argv[7]); // Allow seed input
            run_sketch_sampling(argv[2], argv[3], argv[4]);
        }
        else if (task == "debug_rule") {
            if (argc < 3) { cerr << "Usage: debug_rule <rule_file>" << endl; return 1; }
            debug_rule_parsing(argv[2]);
        }
        else if (task == "dump") {
            if (argc < 4) { cerr << "Usage: dump <dataset_dir> <output_file>" << endl; return 1; }
            dump_graph(argv[2], argv[3]);
        }

        // ---------------------------------------------------------
        // PART 2: Research & Benchmarking (Original Logic + Restored)
        // ---------------------------------------------------------
        else {
            if (argc < 3) { cerr << "Error: Experiments require <dataset>." << endl; return 1; }
            string dataset = argv[2];
            string topr = (argc > 3) ? argv[3] : "0.1";
            string beta = (argc > 4) ? argv[4] : "0.1"; 

            if (argc > 5) SEED = std::stoul(argv[5]);

            // Create output directories required for ExactD+/ExactH+ output redirection
            // Note: The C++ code doesn't write these files itself, it prints to stdout.
            // These dirs are needed if you pipe output like: > global_res/dataset/df1/file.res
            // ensure_directory("global_res/" + dataset + "/df1");
            // ensure_directory("global_res/" + dataset + "/hf1");

            cout << ">>> Running Experiment: " << task << " on " << dataset << endl;

            // --- A. Exact Algorithms (Ground Truth Generation) ---
            if (task == "ExactD") {
                // Generates strictly greater ground truth
                Effective_hg_global_greater_f1(dataset, stod(topr), "df1");
            }
            else if (task == "ExactD+") { // RESTORED
                // Generates inclusive ground truth (Required for GloD)
                Effective_hg_global_f1(dataset, stod(topr), "df1");
            }
            else if (task == "ExactH") {
                Effective_hg_global_greater_f1(dataset, stod(topr), "hf1");
            }
            else if (task == "ExactH+") { // RESTORED
                // Generates inclusive ground truth (Required for GloH)
                Effective_hg_global_f1(dataset, stod(topr), "hf1");
            }
            
            else if (task == "GloD") { 
                // Parse K if provided (Argument index 5 corresponds to 'str(k_val)' from Python)
                if (argc > 5) {
                    K = std::stoul(argv[5]);
                }
                std::cout << ">>> Running GloD with K=" << K << std::endl; // Debug print
                Effective_prop_opt_global_cross(dataset, topr, "df1");
            }
            else if (task == "GloH") { 
                if (argc > 5) K = std::stoul(argv[5]); 
                Effective_prop_opt_global_cross(dataset, topr, "hf1");
            }

            // --- C. Approximate Algorithms (Personalized) ---
            else if (task == "PerD") {
                if (argc > 5) K = std::stoul(argv[5]);
                Effective_prop_opt_personalized_cross(dataset, topr, stod(beta), "d");
            }
            else if (task == "PerH") { 
                if (argc > 5) K = std::stoul(argv[5]);
                Effective_prop_opt_personalized_cross(dataset, topr, stod(beta), "h");
            }
            else if (task == "PerD+") { 
                if (argc > 5) K = std::stoul(argv[5]);
                Effective_prop_opt_personalized_cross(dataset, topr, stod(beta), "dp");
            }
            else if (task == "PerH+") { 
                if (argc > 5) K = std::stoul(argv[5]);
                Effective_prop_opt_personalized_cross(dataset, topr, stod(beta), "hp");
            }
            // --- D2. Epsilon (approximation ratio) ---
            else if (task == "Epsilon") {
                // Computes epsilon for degree and h-index centrality
                // Usage: Epsilon <dataset> <topr> <K> [SEED]
                if (argc > 5) K = std::stoul(argv[5]);
                if (argc > 6) SEED = std::stoul(argv[6]);
                Effective_epsilon(dataset, topr);
            }

            // --- D. Utility & Stats ---
            else if (task == "mpcount") { 
                // Counts meta-paths in the rule file (1 = limit file, 0 = regular dat)
                MetaPathCount(dataset, 1); 
            }
            else if (task == "lensplit") { // RESTORED
                // Splits rules into different files based on length
                MetaPathLenSplit(dataset);
            }
            else if (task == "hg_stats") { // RESTORED
                // Calculates graph statistics (density, overlap)
                Effective_hg_stats(dataset);
            }
            else if (task == "matching_graph_time") { // RESTORED
                MatchingGraphTime(dataset);
            }

            // --- E. Scalability & Alternative Implementations ---
            else if (task == "scale" || task == "global_scale") {
                // Default scalability test (Global Degree)
                Scalability_prop_opt_global(dataset, topr, "d", ""); 
            }
            else if (task == "personalized_scale") { // RESTORED
                Scalability_prop_opt_personalized(dataset, topr, stod(beta), "d", "");
            }
            else if (task == "hg_scale") { // RESTORED
                Scalability_hg_greater_f1(dataset, stod(topr), "df1", "");
            }
            else if (task == "union") { // RESTORED
                // Uses the "Union" method for Exact calculation
                Effective_hg_global_f1_by_union(dataset, stod(topr), "df1");
            }
            else {
                cerr << "ERROR: Unknown task '" << task << "'" << endl;
                return 1;
            }
        }
    } catch (const exception& e) {
        cerr << "CRITICAL ERROR: " << e.what() << endl;
        return 1;
    }

    return 0;
}