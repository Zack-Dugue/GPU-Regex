#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <queue>
#include <stack>
#include <tuple>
#include <list>
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
            node_t* left_node = parse_regex(reg.substr(0, i), end_node);
            node_t* right_node = parse_regex(reg.substr(i+1),end_node);
            node_t** next_nodes = (node_t**) malloc(2 * sizeof(node_t*));
            next_nodes[0] = left_node;
            next_nodes[1] = right_node;
            std::string* transition_array = new std::string[2];
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
            node_t* right_node = parse_regex(reg.substr(i+1,reg.length() - i),end_node);
            node_t* left_node = parse_regex(reg.substr(0, i),right_node);

            node_t** next_nodes = (node_t**) malloc(sizeof(node_t*));
            next_nodes[0] = left_node;
            std::string* transition_array = new std::string[1];
            transition_array[0] = "";
            int num_transitions = 1;
            return init_node(main_node, next_nodes,transition_array, num_transitions,false);
        }
     }

    switch(reg[reg.length()-1])
    {
    case '*':
    {  
        node_t* split_node = (node_t*)  malloc(sizeof(node_t));

        node_t* main_node = parse_regex(reg.substr(0,reg.length()-1), split_node);
        node_t** next_nodes = (node_t**) malloc(2 * sizeof(node_t*));
        std::string* transition_array = new std::string[2];
        transition_array[0] = std::string("");
        transition_array[1] = std::string("");
        next_nodes[0] = main_node;
        next_nodes[1] = end_node;

        return init_node(split_node, next_nodes, transition_array, 2, false);
    }
    case '+':
    {
        std::string new_string;
        new_string.append(reg.substr(0,reg.length()-1)).append("@").append(reg.substr(0,reg.length()-1)).append("*");
        printf("\t plus new string %s\n", new_string.c_str());
        return parse_regex(new_string,end_node);
    }
    case '?':
    {
        std::string new_string;
        new_string.append("").append("|").append(reg.substr(0,reg.length()-2));
        return parse_regex(new_string,end_node);
    }
    }

    if(reg[0] == '(' && reg[reg.length()-1] == ')')
    {
        return parse_regex(reg.substr(1,reg.length()-2), end_node);
    }


    
    node_t* main_node = (node_t*) malloc(sizeof(node_t));
    node_t** next_nodes = (node_t**) malloc(sizeof(node_t*));
    next_nodes[0] = end_node;
    //TODO replace this with a string copy
    std::string *transition_array = new std::string[1];
    transition_array[0] = reg;

    return init_node(main_node, next_nodes, transition_array, 1,false);

}

struct trav_return_t
{
    node_t* ending_node_list;
    std::string* string_list;
};


//TODO make this return a list of possible strings rather then a complete list of strings. 
//
std::tuple<node_t*, std::string> traverse_node(node_t* node, std::string line)
{
    for(int i = 0; i < node->num_transitions; i++)
    {
        printf("traversing on transition %d\n", i);
        std::string transition = (node->transition_array)[i];
        printf("\t transition\t%s\n", transition.c_str());
        printf("\t line      \t%s\n", line.substr(0,transition.length()).c_str());
        if(transition.length() == 0 ||transition.compare(line.substr(0,transition.length())) == 0)
        {
            printf("\t they are equal\n");
            auto [new_node, new_line] = traverse_node(node->next_nodes[i], line.substr(transition.length()));
            return {new_node , line.substr(0,transition.length()).append(new_line)};
        }
    }
    return {node, ""};
}


//TODO Rewrite later:

std::list<std::string> prepend_string_to_list(std::list<std::string> old_list, std::string line) {
    // Iterate through each element in the list
    for (std::string& item : old_list) {
        // Prepend 'line' to the current item
        item = line + item;
    }
    // Return the modified list
    return old_list;
}

std::list<std::string> traverse_node_with_end(node_t* node, node_t* end_node, std::string line)
{
    std::list<std::string> full_match_list;
    for(int i = 0; i < node->num_transitions; i++)
    {
        printf("traversing on transition %d\n", i);
        std::string transition = (node->transition_array)[i];
        printf("\t transition\t%s\n", transition.c_str());
        printf("\t line      \t%s\n", line.substr(0,transition.length()).c_str());
        std::list<std::string> partial_match_list;
        if(transition.length() == 0 ||transition.compare(line.substr(0,transition.length())) == 0)
        {
            printf("\t transition was real \n");
            partial_match_list = traverse_node_with_end(node->next_nodes[i], end_node, line.substr(transition.length()));
            printf("\t match length was: %d\n", partial_match_list.size());
            if(partial_match_list.size() > 0)
            {
                full_match_list.splice(full_match_list.end(), prepend_string_to_list(partial_match_list,line.substr(0,transition.length())));
            }
        }
    }
    if(node == end_node)
    {
        full_match_list.push_front(std::string(""));
    }
    return full_match_list;
}



// Returns the many regex match it finds. 
std::string* find_match(std::string line, std::string regex)
{
    node_t* end_node = (node_t*) malloc(sizeof(node_t));
    node_t* start_node = parse_regex(regex,end_node);
    std::string* match_arr = new std::string[line.length()];
    int j = 0;
    for(int i =0 ; i < line.length(); i++)
    {
        auto [out_node, match_str] = traverse_node(start_node,line.substr(i));
        if(out_node == end_node)
        {
            match_arr[j] = match_str;
            j++;
        }

    }
    return match_arr;
}


// Returns the many regex match it finds. 
std::list<std::string> find_match_better(std::string line, std::string regex)
{
    node_t* end_node = (node_t*) malloc(sizeof(node_t));
    node_t* start_node = parse_regex(regex,end_node);
    std::list<std::string> match_list;
    int j = 0;
    for(int i =0 ; i < line.length(); i++)
    {
            std::list<std::string> partial_match_list = traverse_node_with_end(start_node,end_node,line.substr(i, line.length()-i));
            for(std::string match : partial_match_list)
            {
                match_list.emplace_front(match);
            }
    }
    printf("\n\nwe found %d matches\n\n", match_list.size());
    return match_list;
}

int demo()
{
    printf("Beginning Demo\n\n");
    printf("This is a limited form of regex matching. ");
    printf("Currently unsupported regex features includes: look around, indeterminate character matching (ie using '.'), and group based matching. ");
    printf("Further more, implicit concatenation (the concatenation operator is '@') is only assumed between non operator characters. For example 'a(b|c)' is an invalid regex. Instead, use 'a@(b|c)'. ");
    printf("I hope to fix these issues in the future somewhat but for now I wanted to get a demo to you guys that has all the basic functionalities. \n");
    while(true){
    char regex[1000];
    char line[1000];
    char cont[3];
    printf("\n match a regex: [Y] or [N]\t");
    fgets(cont,3,stdin);
    if(cont == "N\n"){break;}
    printf("\n what is your regex?\t");
    fgets(regex,1000,stdin);
    printf("\n what is your line?\t");
    fgets(line,1000,stdin);
    std::string str_regex = std::string(regex);
    str_regex = str_regex.substr(0, str_regex.length()-1);
    std::string str_line = std::string(line);
    str_line = str_line.substr(0, str_line.length()-1);
    printf("\n Begin Matching\n");
    std::list<std::string> matches = find_match_better(str_line,str_regex);
    printf("\n\n Matches found:\n");
    for(std::string match: matches)
    {
        printf("%s\n",match.c_str());
    }
    }
}


int main()
{   
    // printf("starting program\n");
    // printf("Does it run without failing on hello\n");
    // node_t* end_node = (node_t*) malloc(sizeof(node_t));
    // node_t* ouput_node = parse_regex("hello", end_node); 
    // printf("Does it run without failing on hello@world\n");
    // ouput_node = parse_regex("hello@world", end_node); 
    // printf("Does it run without failing on hello|world\n");
    // ouput_node = parse_regex("hello|world\n", end_node); 
    // printf("Does it run without failing on hello*\n");
    // ouput_node = parse_regex("hello*\n", end_node);
    // printf("Now we try to traverse the regex 'hello'\n");
    // ouput_node = parse_regex("hello", end_node); 
    // traverse_node(ouput_node, "hello world");
    // printf("Now we try to traverse the regex 'hello*'\n");
    // ouput_node = parse_regex("hello*", end_node); 
    // traverse_node(ouput_node, "hello world");
    // printf("Now we try to traverse the regex 'hello@world'\n");
    // ouput_node = parse_regex("hello@_world", end_node); 
    // traverse_node(ouput_node, "hello_world");
    //  printf("Now we try to traverse the regex 'hello|world'\n");
    // ouput_node = parse_regex("hello|world", end_node); 
    // traverse_node(ouput_node, "world");
    // if(ouput_node != end_node)
    // {
    //     printf("MASSIVE L\n");
    // }
    // printf("Now testing actually looking for matches\n");
    // std::string line = std::string("I'm a boss ass bitch bitch bitch bitch");
    // std::string regex = std::string("boss|bi@tch");
    // std::list<std::string> matches = find_match_better(line,regex);
    // for(std::string match: matches)
    // {
    //     printf("%s\n",match.c_str());
    // }
    // printf("Now testing a second time\n");
    //  line = std::string("abcdeeefg");
    //  regex = std::string("e*@f");
    // matches = find_match_better(line,regex);
    // for(std::string match: matches)
    // {
    //     printf("%s\n",match.c_str());
    // }
    // printf("Now testing a third time\n");
    //  line = std::string("abcdddaccddc");
    //  regex = std::string("(a|b)@c+");

    // matches = find_match_better(line,regex);
    // for(std::string match: matches)
    // {
    //     printf("%s\n",match.c_str());
    // }

    // printf("program concluded\n");
    demo();

}