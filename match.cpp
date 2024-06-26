
#include "match.hpp"




bool regex_string_comp(std::string transition, std::string line)
{
    if(transition.length() != line.length())
    {
        // printf("%s and %s are not the same length \n", transition.c_str(), line.c_str());
        return false;
    }

    for(int i = 0; i < line.length(); i++)
    {
        if(transition[i] == '.')
        {
            if(i==0)
            continue;
            else if(transition[i-1] == '\\')
            continue; 
        }
        else
        {
            if(transition[i] != line[i])
            {
                // printf("%s and %s are not the same \n", transition.c_str(), line.c_str());
                return false;
            }
        }

    }
    return true;
}

std::list<string> traverse_node_with_end(vector<state*> state_vec, string line,int start_idx, int end_idx)
{
    //printf("begin traverse node\n");
    std::list<std::string> full_match_list;
    //printf("state vec size is %d\n",state_vec.size());

    //printf("\t initializing active states\n");
    int active_states[state_vec.size()];
    //printf("\t done initializing active states\n");

    //printf("\t setting active states to 0\n");
    for(int i = 0; i < state_vec.size(); i++)
    {
        active_states[i] = 0;
    }
     //printf("\t done setting active states to 0\n");
    
    //printf("\t setting start idx to 1\n");
    active_states[start_idx] = 1;
        
    //printf("\t iterating through state vector \n");
    bool states_are_active = true;
    int j = 0;
    while(states_are_active){
    //printf("\t iteration %d\n",j);
    j++;

    states_are_active = false;
    for(int i = 0; i < state_vec.size(); i++)
    {
    //printf("\t\t checking state %d \n", i);
    // //printf("\t\t can we acces node 0? stave_vec[0] size is %d\n",state_vec[0]->transitions.size());
    // //printf("\t\t can we acces node 1? stave_vec[1] size is %d\n",state_vec[1]->transitions.size());
    if(active_states[i])
    {
        //printf("\t\t this state is active \n");
        states_are_active = true;

        int old_val = active_states[i];
        active_states[i] = 0;
        //printf("\t\t iterating through transitions \n");


        for(transition tr : state_vec[i]->transitions)
        {
            //printf("\t\t\t tranisition iteration\n");
            //printf("\t\t\t tr.next_state_idx %d\n", tr.next_state_idx);
            //printf("\t\t\t tr.txt  %s\n", tr.txt.c_str());

            if(regex_string_comp(tr.txt, line.substr(old_val-1,tr.txt.length()))){
            //printf("\t\t\t it's a match\n");
            // if(active_states[tr.next_state_idx] != 0)
            // {
            //     printf("HOLY CRAP IT HAPPENED\n");
            //     throw 505;
            // }
            active_states[tr.next_state_idx] = old_val + tr.txt.length();
            }
        }
        if(i == end_idx)
        {
            //printf("\t\t\t This is an accept state\n");
            if(old_val != 1){
            full_match_list.push_back(line.substr(0,old_val-1));
            }
            else
            {
                //printf("\t\t\tbut this string is empty\n");
            }
        }
    }
    }
}
    return full_match_list;

}

void match_this_regex(string regex, string line)
{
  //printf("\n\n beginning regex match of %s on %s\n", regex.c_str(), line.c_str());
  state* start_state = new state();
  list<state*> state_list = parse_regex(regex, start_state);
  vector<state*> state_vec = convert_state_list(state_list);
  //printf("num states %d\n", state_vec.size());
  list<match_t> full_match_list; 
  for(int i = 0; i < line.length(); i++){
    //printf("checking string %s\n",line.substr(i,line.length()-i).c_str());
    list<string> partial_match_list = traverse_node_with_end(state_vec,line.substr(i,line.length()-i), 0, 1);
    for(string match : partial_match_list)
    {
        full_match_list.push_back((match_t){match, i});
    }
  }
  printf("printing %d matches with cpu algo:\n",full_match_list.size());
//   for(match_t match : full_match_list)
//   {
//     printf("\tmatch at index %d : \t %s\n", match.idx,match.txt.c_str());
//   }
}