#include "nfa.hpp"
#include <algorithm>
#include <map>

using namespace std;

const int MAX_CHUNK_LENGTH = 64;

string fix_this(std::string s) {
    std::replace( s.begin(), s.end(), '|', '\0'); // replace all 'x' to 'y'
    return s;
}


void print_nfa(vector<state*> nfa)
{
        int i = 0;
    for(state* st : nfa)
    {
        printf("state %d:\n",i);
        st->print_self();
        i++;

    }
}

void checkNFA(const std::vector<state*>& states) {
    if (states.empty()) {
        throw std::runtime_error("The NFA is empty.");
    }

    int rootCount = 0;
    int leafCount = 0;
    std::vector<int> incomingCounts(states.size(), 0);

    // Calculate incoming counts for each state
    int i = 0;

    for (const state* st : states) {
        for (const transition& t : st->transitions) {
            if (t.next_state_idx < 0 || t.next_state_idx >= states.size()) {
                printf("state %d has a transition points to %d which is not between 0 and %d",i, t.next_state_idx, states.size());
                print_nfa(states);
                generate_nfa_diagram(states, "error_diagram\\ERROR");
                throw std::runtime_error("Transition points to an invalid state index.");
            }
            incomingCounts[t.next_state_idx]++;
        }
        i++;
    }

    // Determine the number of root and leaf nodes
    for (size_t i = 0; i < states.size(); ++i) {
        if (incomingCounts[i] == 0) {
            rootCount++;
        }
        if (states[i]->transitions.empty()) {
            leafCount++;
        }
    }

    // Prepare to accumulate error messages if any conditions are violated
    std::stringstream errors;

    if (rootCount != 1) {
        cout << "Error: There should be exactly one root node, found " << rootCount << ".\n";
    }

    if (leafCount != 1) {
        cout << "Error: There should be exactly one leaf node, found " << leafCount << ".\n";
    }

    // If there are any error messages, throw an exception
    if (!errors.str().empty()) {
        print_nfa(states);
        generate_nfa_diagram(states, "error_diagram\\ERROR");
        throw std::runtime_error(errors.str());
    }

    // std::cout << "The NFA configuration is correct." << std::endl;
}



// Function to generate a filename with a timestamp
std::string get_unique_filename(const std::string& base_name, const std::string& extension) {
    // Get current time
    std::time_t now = std::time(nullptr);  // Current time as time_t
    std::tm* now_tm = std::localtime(&now); // Convert time_t to tm as local time

    // Buffer to hold the formatted date and time string
    char buffer[80];

    // Format the time using strftime
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", now_tm);

    // Create a stringstream to format the filename
    std::stringstream ss;
    ss << base_name << "_" << buffer << "." << extension;

    return ss.str();
}

void generate_nfa_diagram(const std::vector<state*>& states,std::string base_filename) {
    return;
    std::string dot_filename = get_unique_filename(base_filename, "dot");
    std::string png_filename = get_unique_filename(base_filename, "png");

    // Write to dot file
    std::ofstream dot_file(dot_filename);
    if (!dot_file) {
        std::cerr << "Failed to open file for writing: " << dot_filename << std::endl;
        return;
    }

    dot_file << "digraph NFA {" << std::endl;
    dot_file << "    rankdir=LR;" << std::endl;
    dot_file << "    node [shape = circle];" << std::endl;
    
    for (size_t i = 0; i < states.size(); ++i) {
        for (const auto& trans : states[i]->transitions) {
            dot_file << "    " << i << " -> " << trans.next_state_idx
                     << " [label=\"" << trans.txt << "\"];" << std::endl;
        }
    }

    dot_file << "    " << 0 << " [shape=doublecircle, label=\"Start\"];" << std::endl;
    if (states.size() > 1) {
        dot_file << "    " << 1 << " [shape=doublecircle, label=\"End\"];" << std::endl;
    }

    dot_file << "}" << std::endl;
    dot_file.close();

    // Generate PNG using Graphviz
    std::string command = "dot -Tpng " + dot_filename + " -o " + png_filename;
    std::system(command.c_str());
}

std::string pre_process_regex(std::string reg)
{
    int j =0 ;
    bool skip_next = false;
    for(int i = 0; i < ((int)reg.length()-1); i++)
    {
        // printf("pre_processessing iter %d",i);
        if(skip_next)
        {
            skip_next = false;
            continue;
        }
        if((reg[i] == '+' || reg[i] == '*'|| reg[i] == '?'|| reg[i] == ')') && (reg[i+1] != '@'&& reg[i+1] != '+' &&  reg[i+1] != '*' && reg[i+1] != '?' && reg[i+1] != ')' ))
        {
            reg.insert(i+1,1, '@');
            i++;
        } 
        if(reg[i] != '+' && reg[i] != '*' && reg[i] != '?' && reg[i] != ')' && reg[i] != '(' && reg[i] != '@')
        {
            j++;
        }
        else
        {
            j=0;
        }
        if(j >= MAX_CHUNK_LENGTH)
        {
            // printf("MAX CHUNK LENGTH EXCEEDED\n");
            reg.insert(i, 1, '@');
            j=0;
            // skip_next=true;
        }
    }
    return reg;
}

//What is my problem rn? 
// My problem is that I can't really define an "end" state, properly. 
// {}

list<state*> append_state_list(list<state*> l1, list<state*> l2, int end_state_idx)
{
    // printf("\t\t beginning state list append\n");
    // printf("\t\t l2 is length %d\n", l2.size());

    int old_size = l1.size();
    for(state* st : l2)
    {
        // printf("\t\t iterating on for loop\n");
        // printf("\t\t st->transitions.size() = %d\n", st->transitions.size());
        for(transition& t: st->transitions)
        {
            t.next_state_idx = t.next_state_idx + old_size;
        }


        if(st->transitions.size() == 0)
        {
            // printf("\t\t adding end state transition\n");
            st->add_transition(end_state_idx,string(""),true);
            // printf("\t\t done adding end state transition\n");

        }


        l1.push_back(st);

    }
    return l1;
}

//TODO: Rework to use the character first method. 
vector<state*> convert_state_list(list<state*> st_list){
    vector<state*> st_vec =  vector<state*>();
    for(state* st: st_list)
    {
        st_vec.push_back(st);
    }
    return st_vec;
}



// The different Regex
list<state*> parse_regex(string reg,state* main_node)
{
    //works left to right, starts deep.
    // First handle the "or" situation
    printf("Starting parse\n");
    reg = pre_process_regex(reg);
    printf("Regex after pre-processing %s\n",reg.c_str());
    list<state*> state_list;
    state_list.push_front(main_node);

    int depth = 0;
    for (int i = 0;  i <reg.length(); i ++){
        if(reg[i] == '(') depth += 1;
        else if(reg[i] == ')') depth -= 1;
        if(reg[i] == '|' && depth == 0)
        {
            state* end_node = new state();
            state_list.push_back(end_node);
            printf("before anything nfa\n ");
            print_nfa(convert_state_list(state_list));
            state* left_node = new state();
            if(i == 0)
            {   
                main_node->add_transition(1,string(""), true);
            }
            else
            {
                list<state*> l_state_list= parse_regex(reg.substr(0, i), left_node);
                printf("L STATE LIST\n");
                print_nfa(convert_state_list(l_state_list));
                main_node->add_transition((int)state_list.size(),string(""), true);
                state_list = append_state_list(state_list, l_state_list,1);
                     printf("COMBINED STATE LIST\n ");
                print_nfa(convert_state_list(state_list));
            }
            state* right_node = new state();

            if(i== (int)(reg.length()-1)){
                main_node->add_transition(1,string(""), true);
            }
            else
            {
                list<state*> r_state_list= parse_regex(reg.substr(i+1, reg.length()-i), right_node);
                                printf("R STATE LIST\n");
                print_nfa(convert_state_list(r_state_list));
                main_node->add_transition((int)state_list.size(),string(""), true);
                state_list = append_state_list(state_list, r_state_list,1);
                printf("COMBINED STATE LIST\n ");
                print_nfa(convert_state_list(state_list));
            }
            checkNFA(convert_state_list(state_list));
            generate_nfa_diagram(convert_state_list(state_list),(string("progressive_diagram\\") + fix_this(reg)).c_str());
            printf("returning NFA for reg %s\n", reg.c_str());
            return state_list;
        }
     }
     
    // Next we handle the Concatenation Situation (this needs to be put somehwere in 
    // like a string pre-processing thing).
     for (int i = 0;  i < (int)(reg.length()); i ++){
        if(reg[i] == '(') depth += 1;
        else if(reg[i] == ')') depth -= 1;
        if(reg[i] == '@' && depth == 0)
        {
            state* end_node = new state();
            state_list.push_back(end_node);

            state* left_node = new state();
            list<state*> l_state_list = parse_regex(reg.substr(0, i), left_node);
            main_node->add_transition((int) (state_list.size()),string(""), true);
            state_list = append_state_list(state_list, l_state_list,state_list.size() + l_state_list.size());

            state* right_node = new state();
            list<state*> r_state_list= parse_regex(reg.substr(i+1,reg.size() -i), right_node);
            state_list = append_state_list(state_list, r_state_list,1);
            checkNFA(convert_state_list(state_list));
            generate_nfa_diagram(convert_state_list(state_list),(string("progressive_diagram\\") + fix_this(reg)).c_str());
            printf("returning NFA for reg %s\n",reg.c_str());
            
            return state_list;            
        }
     }

    switch(reg[reg.length()-1])
    {
    case '*':
    {  
        state* end_node = new state();
        state_list.push_back(end_node);
        state* left_node = new state();
        list<state*> l_state_list = parse_regex(reg.substr(0, reg.length()-1), left_node);
        main_node->add_transition(1,string(""), true);
        main_node->add_transition(2,string(""), true);
        printf("before star append size of  list %d\n", state_list.size());

        state_list = append_state_list(state_list, l_state_list,0);
        printf("after star append size of new list %d\n", state_list.size());
        checkNFA(convert_state_list(state_list));
        printf("returning NFA for reg %s\n", reg.c_str());
        return state_list;
    }
    case '+':
    {
        std::string new_string;
        new_string.append(reg.substr(0,reg.length()-1)).append("@").append(reg.substr(0,reg.length()-1)).append("*");
        // printf("\t plus new string %s\n", new_string.c_str());
        printf("returning NFA for reg %s\n", reg.c_str());
        return parse_regex(new_string,main_node);
    }
    case '?':
    {
        // printf("We are in the ? case\n");
        std::string new_string = std::string("");
        // printf("begin appending\n");
        new_string.append("(|");
        // printf("append the actual text\n");
        new_string.append(reg.substr(0,reg.length()-1));
        // printf("append the ) \n");
        new_string.append(")");
        // printf("now print the string\n");
        // printf("new_string %s\n", new_string.c_str());
        return parse_regex(new_string,main_node);
    }
    }

    if(reg[0] == '(' && reg[reg.length()-1] == ')')
    {
        printf("returning NFA for reg %s\n", reg.c_str());
        return parse_regex(reg.substr(1,reg.length()-2), main_node);
    }


    
    state* end_node = new state();
    state_list.push_back(end_node);
    main_node->add_transition(1,reg, true);
    //TODO replace this with a string copy


    return state_list;

}

