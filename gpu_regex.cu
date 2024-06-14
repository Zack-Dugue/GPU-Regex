#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include "nfa.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <math.h>
#include "gpu_regex.cuh"


#define MAX_NUMBER_OF_MATCHES 256
const int MAX_CHUNK_LENGTH = 64;

#define CUDA_CALL(status) do {                                                \
    std::stringstream _error;                                                \
    if ((status) != 0) {                                                    \
      _error << "Cuda failure: " << _cudaGetErrorEnum(status);                \
      FatalError(_error.str());                                                \
    }                                                                        \
} while(0)

struct d_transition
{
    int next_state_idx;
    char txt[MAX_CHUNK_LENGTH];
    int len; 
};


struct d_state 
{
    int start_tr;
    int num_tr;
};

struct nfa_t 
{
    vector<d_state> states;
    vector<d_transition> transitions;
    int n_states;
    int n_transitions;
};
//Rewrite this later:
nfa_t convert_state_vec_d(vector<state*> st_vec)
{
    vector<d_state> d_state_vec;
    vector<d_transition> d_tr_vec;

    int offset = 0;
    for(int i =0; i<st_vec.size();i++ )
    {
        state* st = st_vec[i];
        d_state_vec.push_back((d_state){offset,st->transitions.size()});
        for(transition tr : st->transitions)
        {
            d_tr_vec.push_back((d_transition) {tr.next_state_idx, *tr.txt.c_str(), tr.txt.length()});
        }
        offset += st->transitions.size();
    }
    return (nfa_t) {d_state_vec, d_tr_vec, st_vec.size(), offset};
} 




__device__ bool regex_string_comp(char* transition, const char* line, int len)
{
 
    for(int i = 0; i < len; i++)
    {
    if(transition[i] != line[i] && transition[i] != '.')
    {
        return false;
    }
    }
    return true;
}



 //todo replace all integers in here with unsigned integers.
__global__ void RMatchKernel(d_state* nfa_st,d_transition* nfa_tr, const char* text, const int n, const int L, int* out_vec){
    int blocksInGrid = gridDim.x;  // Total number of blocks in the grid along the x-axis
    int threadsInBlock = blockDim.x;  // Number of threads per block along the x-axis

    // You can also access the block index and thread index
    int blockIndex = blockIdx.x;  // Index of the current block within the grid
    int threadIndex = threadIdx.x;
    int warp_idx = threadIndex/32;


    // TODO handle chunking up the data so we don't have such a huge blob in shared memory.
    // There are state number of rows and L number of columns in our big state vector.
    // Consider transposing this!
    int **state_vec;
    cudaMalloc((void***) &state_vec, sizeof(int)*n*L);
    //We need an object to actually be able to write out to.
    int out_counter;
    //Now we set everything to 0 except for the initial states. 
    for(int j = blockIndex*threadsInBlock + threadIndex ; j < n*L; j = j + blocksInGrid * threadsInBlock)
    {
        state_vec[j/n][j%n] = 0 + (j % n == 0);
    }



    __syncthreads();
    bool any_active_state = true;
    while(any_active_state){
    __syncthreads();
    unsigned int bitmask =0;
    for(int j = blockIndex*threadsInBlock + threadIndex ; j < n*L; j = j + blocksInGrid * threadsInBlock)
    {
        if(state_vec[j/n][j%n] != 0 )
        {
        __syncwarp(bitmask);
        int old_val = state_vec[j/n][j%n];
        state_vec[j/n][j%n] = 0;
        
        for(int i = nfa_st[j%n].start_tr; i < nfa_st[j%n].start_tr + nfa_st[j%n].num_tr; i++ )
        {
            d_transition tr = nfa_tr[i];
            //Consider race condition here. Maybe want to account for that? 
            state_vec[j/n][tr.next_state_idx] = old_val + tr.len*regex_string_comp(tr.txt,text,L-(i/n));
        }
        if(j%n == 1)
        {
            int current_counter = atomicAdd(&out_counter, 1);
            // out_vec[current_counter] = j/n;
            // out_vec[current_counter+1] = old_val;
        }
        any_active_state = true;
        }
        }
    
    bitmask |= (1 << warp_idx);
    }
}


__host__
void GPU_Match_It(string regex, string text,int gridSize, int blockSize)
{
    printf("Beginning GPU matching of regex %s with string %s\n", regex.c_str(), text.c_str());
    state* start_state = new state();
    printf("Parsing Regex\n");
    list<state*> state_list = parse_regex(regex,start_state);
    printf("Converting NFA into GPU friendly format.\n");
    vector<state*> nfa_vec = convert_state_list(state_list);
    nfa_t nfa_struct = convert_state_vec_d(nfa_vec);

    printf("Initializing out_vec in cuda memory\n");
    int* out_vec;
    CUDA_CALL(cudaMalloc((void**) &out_vec, 2*MAX_NUMBER_OF_MATCHES*sizeof(int)));
    CUDA_CALL(cudaMemset((void*) out_vec,-1,  2*MAX_NUMBER_OF_MATCHES*sizeof(int)));


    const int n = nfa_struct.n_states;
    // state** nfa;

    // TODO work on a shared memory implementation.
    printf("Initializing nfa_st in cuda memory\n");
 
    d_state* nfa_st;
    CUDA_CALL(cudaMalloc((void**) &nfa_st, sizeof(d_state)*n));
    CUDA_CALL(cudaMemcpy(nfa_st, (&nfa_struct.states[0]), sizeof(d_state)*n, cudaMemcpyHostToDevice));
    
    printf("Initializing nfa_tr in cuda memory\n");

    d_transition* nfa_tr;
    CUDA_CALL(cudaMalloc((void**) &nfa_tr, sizeof(d_transition)*nfa_struct.n_transitions));
    CUDA_CALL(cudaMemcpy(nfa_tr, (&nfa_struct.transitions[0]), sizeof(d_transition)*nfa_struct.n_transitions, cudaMemcpyHostToDevice));
    
    printf("Initializing d_text in cuda memory\n");
    char* d_text;
    CUDA_CALL(cudaMalloc((void**) &d_text, sizeof(d_transition)*nfa_struct.n_transitions));
    CUDA_CALL(cudaMemcpy(d_text, text.c_str(), text.length(), cudaMemcpyHostToDevice));

    printf("Calling the kernel\n");
    RMatchKernel<<<gridSize, blockSize>>>(nfa_st, nfa_tr, d_text,nfa_struct.n_states,text.length(), out_vec);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    printf("kernel complete\n");

    printf("Copying memory off the device back onto the host.\n");
    int *h_out_vec = new int[2*MAX_NUMBER_OF_MATCHES];
    printf("h_out_vec pointer %p, out_vec points %p\n ", h_out_vec, out_vec);
    CUDA_CALL(cudaMemcpy((void*) h_out_vec, (void*) out_vec, 2*MAX_NUMBER_OF_MATCHES*sizeof(int), cudaMemcpyDeviceToHost));

    printf("print matches\n");
    for(int i =0; i < MAX_NUMBER_OF_MATCHES; i++)
    {
        if(h_out_vec[2*i] >0)
        {
            printf("match found at idx %d: %s",h_out_vec[2*i], text.substr(h_out_vec[2*i], h_out_vec[2*i+1]));
        }
    }
}
