#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <queue>
#include <stack>


enum token_type {str, op, l_paren, r_paren, special};


enum op_type {cup,cat,kstar,plus, q_mark, look_ahead};


// +---+----------------------------------------------------------+
// |   |             ERE Precedence (from high to low)            |
// +---+----------------------------------------------------------+
// | 1 | Collation-related bracket symbols | [==] [::] [..]       |
// | 2 | Escaped characters                | \<special character> |
// | 3 | Bracket expression                | []                   |
// | 4 | Grouping                          | ()                   |
// | 5 | Single-character-ERE duplication  | * + ? {m,n}          |
// | 6 | Concatenation                     |                      |
// | 7 | Anchoring                         | ^ $                  |
// | 8 | Alternation                       | |                    |
// +---+-----------------------------------+----------------------+

class Token {       // The class
  public:
    token_type type;            // Access specifier
    void* contents;
};

token_type get_token_type(char symb)
{
    switch(symb)
    {
        case symb
    }
}


std::string regex_to_rpn(std::string reg)
{
    int i = 0;
    int j = 0;
    std::queue<Token> output_queue;
    std::stack<Token> operator_stack;
    std::string running_str;
    while(i < reg.length())
    {   
        std::string symb = reg.substr(i,i+1);
        if()
        {
            running_str.append(symb);
        }
        else
        {
            if(running_str.length() != 0)
            {
                operator_stack.push(running_str);
                running_str.clear();
            }


        }
        i += 1
    }

}