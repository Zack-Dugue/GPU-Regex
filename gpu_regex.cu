    #include <string>
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include "nfa.hpp"
#include <cassert>


struct match_t {
    string txt; 
    int idx; 
    void print()
    {
        printf("idx: %d\t match: %s\n",idx,txt.c_str());
    }; 
    };


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

__global__
void optimalTransposeKernel(const float *input, float *output, int n) {
    // TODO: This should be based off of your shmemTransposeKernel.
    // Use any optimization tricks discussed so far to improve performance.
    // Consider ILP and loop unrolling.
        //padding for BANK CONFLICTS. 
    __shared__ float data[64][65];
    
    //load into shared memory

    const int t_row = threadIdx.x;
    int t_col= 4 * threadIdx.y;

    //Use Instruction Level Parallelism! All these newly set variables
    // should just be registers. So their order isn't important. 
    float a,b,c,d ;
    a = input[(t_row+64 * blockIdx.x) + (t_col+64 * blockIdx.y)*n];
    b = input[(t_row+64 * blockIdx.x) + (t_col+1+64 * blockIdx.y)*n];
    c = input[(t_row+64 * blockIdx.x) + (t_col+2+64 * blockIdx.y)*n];
    d = input[(t_row+64 * blockIdx.x) + (t_col+3+64 * blockIdx.y)*n];

    data[t_row][t_col] = a;
    data[t_row][t_col+1] = b;
    data[t_row][t_col+2] = c;
    data[t_row][t_col+3] = d;
    

    __syncthreads();
}