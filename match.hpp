#include <string>
#include <list>
#include <vector>
#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include "nfa.hpp"

using namespace std;

struct match_t {
    string txt; 
    int idx; 
    void print()
    {
        printf("idx: %d\t match: %s\n",idx,txt.c_str());
    }; 
};


void match_this_regex(string regex, string line);
std::list<string> traverse_node_with_end(vector<state*> state_vec, string line,int start_idx, int end_idx);