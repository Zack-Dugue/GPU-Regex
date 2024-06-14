
#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <cstdio>
#include <ctime>
using namespace std;



struct transition
{
    int next_state_idx;
    string txt;
    bool consume; 
};


struct state 
{
    vector<transition> transitions;
    state() : transitions() {};
    void add_transition(int next_state_idx, string txt, bool consume)
    {
        transition tr = {next_state_idx, txt, consume};
        transitions.push_back(tr);
    }
    void print_self()
    {
        printf("\tprinting state at %p:\n",this );
        int i = 0;
        for(transition tr: transitions)
        {
            printf("\t\ttransition %d: (next_idx=%d, txt = %s) \n", i, tr.next_state_idx, tr.txt.c_str());
            i++;
        }
    }

};

list<state*> parse_regex(string reg,state* main_node);

vector<state*> convert_state_list(list<state*> st_list);

void generate_nfa_diagram(const std::vector<state*>& states,std::string base_filename);

void pruneNFA(std::vector<state*>& nfa);

