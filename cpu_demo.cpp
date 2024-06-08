#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <queue>
#include <stack>
//Many different types of parallelsim are achievable here.
// Firstly, we have to consider parallelism that we need to consider accepting or rejecting the string at every start point. 
// Next there is the parallelism involved in the NFA interpretation of a model.

//How to implement look ahead?


struct node_t { //Make a data structure for the nodes of our NFA
  node_t** next_nodes;
  std::string *transition_array;
  bool eat_str;
  int num_transitions;
}; 

node_t * init_node(node_t* node,node_t** next_nodes, std::string *transition_array, int num_transitions, bool eat_str)
{
    node->next_nodes = next_nodes;
    node-> eat_str = eat_str;
    node-> transition_array = transition_array;
    node-> num_transitions = num_transitions;
    return node;
}




// The different Regex
node_t* parse_regex(std::string reg, node_t* end_node)
{
    //works left to right, starts deep.
    // First handle the "or" situation
    int depth = 0;
     for (int i = 0;  i <reg.length(); i ++){
        if(reg[i] == '(') depth += 1;
        else if(reg[i] == ')') depth -= 1;
        if(reg[i] == '|' && depth == 0)
        {
            node_t* main_node = (node_t*) malloc(sizeof(node_t));
            node_t* left_node = parse_regex(reg.substr(0, i-1), end_node);
            node_t* right_node = parse_regex(reg.substr(i,reg.length() - i),end_node);
            node_t** next_nodes = (node_t**) malloc(2 * sizeof(node_t*));
            next_nodes[0] = left_node;
            next_nodes[1] = right_node;
            std::string *transition_array = (std::string *) malloc(sizeof(std::string) * 2);
            transition_array[0] = "";
            transition_array[1] = "";
            int num_transitions = 2;
            return init_node(main_node, next_nodes,transition_array, num_transitions,false);
        }
     }
     
    // Next we handle the Concatenation Situation (this needs to be put somehwere in 
    // like a string pre-processing thing).
     for (int i = 0;  i <reg.length(); i ++){
        if(reg[i] == '(') depth += 1;
        else if(reg[i] == ')') depth -= 1;
        if(reg[i] == '@' && depth == 0)
        {
            node_t* main_node = (node_t*) malloc(sizeof(node_t));
            node_t* right_node = parse_regex(reg.substr(i,reg.length() - i),end_node);
            node_t* left_node = parse_regex(reg.substr(0, i-1),right_node);

            node_t** next_nodes = (node_t**) malloc(2 * sizeof(node_t*));
            next_nodes[0] = left_node;
            std::string *transition_array = (std::string *) malloc(sizeof(std::string));
            transition_array[0] = "";
            int num_transitions = 1;
            return init_node(main_node, next_nodes,transition_array, num_transitions,false);
        }
     }

    switch(reg[reg.length()-2])
    {
    case '*':
    {  
        node_t* split_node = (node_t*)  malloc(sizeof(node_t));

        node_t* main_node = parse_regex(reg.substr(0,reg.length()-2), split_node);
        node_t** next_nodes = (node_t**) malloc(2 * sizeof(node_t*));
        std::string *transition_array = (std::string *) malloc(sizeof(std::string) * 2);
        transition_array[0] = "";
        transition_array[1] = "";
        next_nodes[0] = main_node;
        next_nodes[1] = end_node;

        return init_node(split_node, next_nodes, transition_array, 2, false);
    }
    case '+':
    {
        std::string new_string;
        new_string.append(reg.substr(0,reg.length()-2)).append("@").append(reg.substr(0,reg.length()-2)).append("*");
        return parse_regex(new_string,end_node);
    }
    case '?':
    {
        std::string new_string;
        new_string.append("").append("|").append(reg.substr(0,reg.length()-2));
        return parse_regex(new_string,end_node);
    }
    }
    if(reg[0] == '(' && reg[reg.length()-2] == ')')
    {
        parse_regex(reg.substr(1,reg.length()-2), end_node);
    }


    
    node_t* main_node = (node_t*) malloc(sizeof(node_t));
    node_t** next_nodes = (node_t**) malloc(sizeof(node_t*));
    next_nodes[0] = end_node;
    //TODO replace this with a string copy
    std::string *transition_array = &reg;
    return init_node(main_node, next_nodes, transition_array, 1,false);

}


int main()
{   
    node_t* end_node = (node_t*) malloc(sizeof(node_t));
    node_t* ouput_node = parse_regex("hello", end_node); 
}