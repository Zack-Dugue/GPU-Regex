    #include <string>
#include <vector>
#include <list>

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

};


std::string pre_process_regex(std::string reg)
{
    for(int i = 0; i < reg.length()-1; i++)
    {
        if((reg[i] == '+' || reg[i] == '*'|| reg[i] == '?'|| reg[i] == ')') && (reg[i+1] != '@'&& reg[i] != '+' &&  reg[i] != '*' && reg[i] != '?' && reg[i] != ')' ))
        {
            reg.insert(i+1,1, '@');
            i++;
        }    
    }
    return reg;
}

//What is my problem rn? 
// My problem is that I can't really define an "end" state, properly. 
// {}

list<state*> append_state_list(list<state*> l1, list<state*> l2, int end_state_idx)
{
    printf("\t\t beginning state list append\n");
    printf("\t\t l2 is length %d\n", l2.size());

    for(state* st : l2)
    {
        printf("\t\t iterating on for loop\n");
        printf("\t\t st->transitions.size() = %d\n", st->transitions.size());
        if(st->transitions.size() == 0)
        {
            printf("\t\t adding end state transition\n");
            st->add_transition(end_state_idx,string(""),true);
            printf("\t\t done adding end state transition\n");

        }
        else{
        for(transition t: st->transitions)
        {
            printf("\t\t altering transition\n");
            t.next_state_idx = t.next_state_idx + l1.size();
                        printf("\t\t done altering transition\n");

        }

        }
        printf("\t\t adding guy to new list\n");

        l1.push_back(st);
        printf("\t\t done adding guy to new list\n");

    }
    return l1;
}

//TODO: Rework to use the character first method. 


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
            state* left_node = new state();
            if(i == 0)
            {   
                main_node->add_transition(1,string(""), true);
            }
            else
            {
                list<state*> l_state_list= parse_regex(reg.substr(0, i), left_node);
                main_node->add_transition((int)state_list.size(),string(""), true);
                state_list = append_state_list(state_list, l_state_list,1);
            }
            state* right_node = new state();

            if(i== reg.length()-1){
                main_node->add_transition(1,string(""), true);
            }
            else
            {
                list<state*> r_state_list= parse_regex(reg.substr(i+1, reg.length()-i), right_node);
                main_node->add_transition((int)state_list.size(),string(""), true);
                state_list = append_state_list(state_list, r_state_list,1);
            }
            return state_list;
        }
     }
     
    // Next we handle the Concatenation Situation (this needs to be put somehwere in 
    // like a string pre-processing thing).
     for (int i = 0;  i <reg.length(); i ++){
        if(reg[i] == '(') depth += 1;
        else if(reg[i] == ')') depth -= 1;
        if(reg[i] == '@' && depth == 0)
        {
            state* end_node = new state();
            state_list.push_back(end_node);
            printf("\t state list old  size %d\n", state_list.size());
            state* left_node = new state();
            list<state*> l_state_list = parse_regex(reg.substr(0, i), left_node);
            printf("\t adding a transition for concatenation\n");
            main_node->add_transition((int) (state_list.size()),string(""), true);
            printf("\t transition added\n");
            printf("\t appending state list\n");
            state_list = append_state_list(state_list, l_state_list,state_list.size() + l_state_list.size());
            printf("\t state list new size %d\n", state_list.size());
            printf("\t moving on to right node\n");
            state* right_node = new state();
            list<state*> r_state_list= parse_regex(reg.substr(i+1,reg.size() -i), right_node);
            printf("\t parse concluded\n");
            state_list = append_state_list(state_list, r_state_list,1);
            printf("\t state list new size %d\n", state_list.size());
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

        return state_list;
    }
    case '+':
    {
        std::string new_string;
        new_string.append(reg.substr(0,reg.length()-1)).append("@").append(reg.substr(0,reg.length()-1)).append("*");
        // printf("\t plus new string %s\n", new_string.c_str());
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
        return parse_regex(reg.substr(1,reg.length()-2), main_node);
    }


    
    state* end_node = new state();
    state_list.push_back(end_node);
    main_node->add_transition(1,reg, true);
    //TODO replace this with a string copy


    return state_list;

}

vector<state*> convert_state_list(list<state*> st_list){
    vector<state*> st_vec =  vector<state*>();
    for(state* st: st_list)
    {
        st_vec.push_back(st);
    }
    return st_vec;
}



bool regex_string_comp(std::string transition, std::string line)
{
    for(int i = 0; i < transition.length(); i++)
    {
        if(transition[i] != line[i] && transition[i] != '.')
        {
            // printf("%s and %s are not the same \n", transition.c_str(), line.c_str());
            return false;
        }

    }
    return true;
}

std::list<string> traverse_node_with_end(vector<state*> state_vec, string line,int start_idx, int end_idx)
{
    printf("begin traverse node\n");
    std::list<std::string> full_match_list;
    printf("state vec size is %d\n",state_vec.size());

    printf("\t initializing active states\n");
    int active_states[state_vec.size()];
    printf("\t done initializing active states\n");

    printf("\t setting active states to 0\n");
    for(int i = 0; i < state_vec.size(); i++)
    {
        active_states[i] = 0;
    }
     printf("\t done setting active states to 0\n");
    
    printf("\t setting start idx to 1\n");
    active_states[start_idx] = 1;
        
    printf("\t iterating through state vector \n");
    bool states_are_active = true;
    int j = 0;
    while(states_are_active){
    printf("\t iteration %d\n",j);
    j++;

    states_are_active = false;
    for(int i = 0; i < state_vec.size(); i++)
    {
    printf("\t\t checking state %d \n", i);
    // printf("\t\t can we acces node 0? stave_vec[0] size is %d\n",state_vec[0]->transitions.size());
    // printf("\t\t can we acces node 1? stave_vec[1] size is %d\n",state_vec[1]->transitions.size());
    if(active_states[i])
    {
        printf("\t\t this state is active \n");
        states_are_active = true;

        int old_val = active_states[i];
        active_states[i] = 0;
        printf("\t\t iterating through transitions \n");


        for(transition tr : state_vec[i]->transitions)
        {
            printf("\t\t\t tranisition iteration\n");
            printf("\t\t\t tr.next_state_idx %d\n", tr.next_state_idx);
            printf("\t\t\t tr.txt  %s\n", tr.txt.c_str());

            if(regex_string_comp(tr.txt, line.substr(old_val-1,tr.txt.length()))){
            printf("\t\t\t it's a match\n");
            if(active_states[tr.next_state_idx] != 0)
            {
                printf("HOLY CRAP IT HAPPENED\n");
                throw 505;
            }
            active_states[tr.next_state_idx] = old_val + tr.txt.length();
            }
        }
        if(i == end_idx)
        {
            printf("\t\t\t This is an accept state\n");
            if(old_val != 1){
            full_match_list.push_back(line.substr(0,old_val-1));
            }
            else
            {
                printf("\t\t\tbut this string is empty\n");
            }
        }
    }
    }
}
    return full_match_list;

}

std::list<string> match_regex(string regex, string line)
{
  printf("\n\n beginning regex match of %s on %s\n", regex.c_str(), line.c_str());
  state* start_state = new state();
  list<state*> state_list = parse_regex(regex, start_state);
  vector<state*> state_vec = convert_state_list(state_list);
  printf("num states %d\n", state_vec.size());
  list<string> full_match_list; 
  for(int i = 0; i < line.length(); i++){
    printf("checking string %s\n",line.substr(i,line.length()-i).c_str());
  list<string> partial_match_list = traverse_node_with_end(state_vec,line.substr(i,line.length()-i), 0, 1);
  full_match_list.merge(partial_match_list);
  }
  return full_match_list;
}


int main()
{
        printf("starting program\n");
    printf("Does it run without failing on hello\n");
    state* start_node = new state();
    list<state*> nfa = parse_regex("hello", start_node); 

    printf("Does it run without failing on hello@world\n");
    start_node->transitions.clear();
    nfa = parse_regex("hello@world", start_node); 

    start_node->transitions.clear();

    printf("Does it run without failing on hello|world\n");
    nfa = parse_regex("hello|world", start_node); 

    printf("Does it run without failing on hello*\n");
    start_node->transitions.clear();
    nfa = parse_regex("(hello)*", start_node); 
    

    printf("Now we try to traverse the regex 'hello'\n");
    start_node->transitions.clear();
    nfa = parse_regex("hello", start_node); 
    vector<state*>state_vec = convert_state_list(nfa);
    printf("state_vec size %d\n", state_vec.size());
    traverse_node_with_end(state_vec, "hello world",0,1);


    printf("Now we try to traverse the regex 'hello*'\n");
    start_node->transitions.clear();
    nfa = parse_regex("(hello)*", start_node); 
    traverse_node_with_end(convert_state_list(nfa), "hello world",0,1);

    printf("Now we try to traverse the regex 'hello@world'\n");
    start_node->transitions.clear();
    nfa = parse_regex("hello@world", start_node); 
    traverse_node_with_end(convert_state_list(nfa), "hello world",0,1);

     printf("Now we try to traverse the regex 'hello|world'\n");
     start_node->transitions.clear();
    nfa = parse_regex("hello|world", start_node); 
    traverse_node_with_end(convert_state_list(nfa), "hello world",0,1);

    printf("Now testing actually looking for matches\n");
    std::string line = std::string("hello world");
    std::string regex = std::string("world");
    std::list<std::string> matches = match_regex(regex,line);
    for(std::string match: matches)
    {
        printf("%s\n",match.c_str());
    }

    line = std::string("I'm a boss ass bitch bitch bitch");
    regex = std::string("bi@tch");
     matches = match_regex(regex,line);
    for(std::string match: matches)
    {
        printf("%s\n",match.c_str());
    }

        line = std::string("I'm a boss ass bitch bitch bitch");
    regex = std::string("so it's like I was saying dave, I mean am I really crazy for thinking that's a messed up thing to do, I feel like it was so obviously uncalled for. Idk many things are wacky these days.");
     matches = match_regex(regex,line);
    for(std::string match: matches)
    {
        printf("%s\n",match.c_str());
    }
    // printf("Now testing a second time\n");
    //  line = std::string("abcdeeefg");
    //  regex = std::string("e*@f");
    // matches = match_regex(line,regex);
    // for(std::string match: matches)
    // {
    //     printf("%s\n",match.c_str());
    // }
    // printf("Now testing a third time\n");
    //  line = std::string("abcdddaccddc");
    //  regex = std::string("(a|b)@c+");

    // matches = match_regex(line,regex);
    // for(std::string match: matches)
    // {
    //     printf("%s\n",match.c_str());
    // }

    printf("program concluded\n");
}
