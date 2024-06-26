#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include "nfa.hpp"
#include <cassert>
#include <assert.h>
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
        d_state_vec.push_back((d_state) {offset,st->transitions.size()});
        for(transition tr : st->transitions)
        {
            d_transition new_tr;
            new_tr.next_state_idx = tr.next_state_idx;
            memset(new_tr.txt, 0, MAX_CHUNK_LENGTH);
            strncpy(new_tr.txt,tr.txt.c_str(),tr.txt.length());
            new_tr.len = tr.txt.length();
            d_tr_vec.push_back(new_tr);
        }
        offset += st->transitions.size();
    }
    return (nfa_t) {d_state_vec, d_tr_vec, st_vec.size(), offset};
} 


// Function to print a human-readable description of the NFA
void describeNFA(const nfa_t& nfa) {
    std::cout << "NFA Description:" << std::endl;
    printf("Num States = %d , Num Transitions = %d\n", nfa.n_states, nfa.n_transitions);

    for (int i = 0; i < nfa.states.size(); i++) {
        const d_state& state = nfa.states[i];
        std::cout << "State " << i << " with " << state.num_tr << " transitions:" << std::endl;
        for (int j = state.start_tr; j < state.start_tr + state.num_tr; j++) {
            const d_transition& tr = nfa.transitions[j];
            std::cout << "  Transition to state " << tr.next_state_idx 
                      << " on symbol '" << std::string(tr.txt, tr.len) << "'" << std::endl;
        }
    }
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
__global__ void RMatchKernel(d_state* nfa_st,d_transition* nfa_tr, const char* text, const int n, const int L_prime, int* out_vec, int* out_counter){
    int blocksInGrid = gridDim.x ;  // Total number of blocks in the grid along the x-axis
    int threadsInBlock = blockDim.x;  // Number of threads per block along the x-axis

    // You can also access the block index and thread index
    int blockIndex = blockIdx.x;  // Index of the current block within the grid
    int threadIndex = threadIdx.x;
    int warp_idx = threadIndex % 32;


    // TODO handle chunking up the data so we don't have such a huge blob in shared memory.
    // There are state number of rows and L number of columns in our big state vector.
    // Consider transposing this!
    // the state_vec can be assumed to take the shape of
    // n*(L/blocks
    extern __shared__ int state_vec[];

    // We need to make it impossible for two blocks to have threads
    // working on the same suffix.
    // We're just going to pretend that L is smaller then it actually is,
    // And then shift t when accessing strings, based on the block index and the.
    // number of blocks. 
    // int baseL = L_prime / blocksInGrid;
    // int remainder = L_prime % blocksInGrid;
    // int L = baseL + (blockIndex < remainder ? 1 : 0);

    int L ;
    if(blockIndex == blocksInGrid-1)
    {
        L = L_prime % (blocksInGrid-1);
    }
    else
    {
        L = L_prime / (blocksInGrid-1);
    }

    bool* any_active_state = (bool*) &state_vec[n*L];




    for(int j = threadIndex ; j < n*L ; j = j + threadsInBlock)
    {
        state_vec[j] = 0 + (j / L == 0);
    }

    
    __syncthreads();
    *any_active_state = true;
    if(blockIndex == 0)printf("starting while loop\n");

    int r = 0;
    while(*any_active_state){
        __syncthreads();
        *any_active_state = false;
        if(blockIndex == 0)printf("While loop iteration %d\n",r); 
        r++;
        unsigned int bitmask =0xffffffff;
        for(int i = 0; threadIndex + threadsInBlock*i < n*L; i++)
        {
            int state_vec_idx = threadIndex + i * threadsInBlock;
            // int j = (blockIndex*threadsInBlock + threadIndex) + (blocksInGrid * threadsInBlock)*i;
            int old_val = state_vec[state_vec_idx];
            int t = L*blockIndex + state_vec_idx % L;
            int s = state_vec_idx / L ;
            if (blockIndex == 0 && threadIdx.x == 0) {
                printf("Calculated t: %d, Base L: %d, BlockIndex: %d, ThreadIdx: %d, StateVecIdx: %d\n", t, L, blockIndex, threadIdx.x, state_vec_idx);
            }
            if(state_vec[state_vec_idx] != 0 )
            {



            state_vec[state_vec_idx] = 0;
            __syncwarp(bitmask);
            for(int k = nfa_st[s].start_tr; k < nfa_st[s].start_tr + nfa_st[s].num_tr; k++ )
            {
                d_transition tr = nfa_tr[k];

                //Consider race condition here. Maybe want to account for that? 
                
                if(regex_string_comp(tr.txt,&text[t+old_val-1],tr.len) && (state_vec[L*tr.next_state_idx + (t- L*blockIndex)] < (old_val + tr.len) ))
                {   

                    
                    // if(blockIndex == 0)printf("\t for loop iteration %d: \t state_vec_idx=%d \t character_idx=%d \t state_idx=%d\n",i,state_vec_idx,t,s);
                    // if(blockIndex == 0)printf("\t gpu_info: \t thread_idx=%d \t block_idx=%d \n",threadIndex,blockIndex);
                    // // if(blockIndex == 0)printf("\t current read_string is %s\n",&text[t]);
                    // if(blockIndex == 0)printf("\t\t transition %d to state %d transition string %s\n",k,tr.next_state_idx,tr.txt);

                    //  if(blockIndex == 0)printf("\t\t the transition matches:\n");
                    // if(blockIndex == 0)printf("\t\t writing %d to state_vec idx %d \n",(old_val + tr.len),L*tr.next_state_idx + (t- L*blockIndex));
                    if(state_vec[L*tr.next_state_idx + (t- L*blockIndex)] != 0)
                    {
                        assert(false);
                        printf("HOLY CRAP IT HAPPENED\n");
                    }
                    state_vec[L*tr.next_state_idx + (t- L*blockIndex)] = (old_val + tr.len);}
            }
            if(s == 1 && old_val > 1)
            {
                if(blockIndex == 0)printf("\t this is an end state\n");
                int current_counter = atomicAdd(out_counter, 1);
                if(blockIndex == 0)printf("\t\t end state atomic add complete\n");
                out_vec[2*current_counter] = t;
                out_vec[2*current_counter+1] = old_val-1;
                if(blockIndex == 0)printf("\t\t %d end state data wrote %d, %d\n",current_counter, t, old_val);

            }
            *any_active_state = true;
            }
            }
        
        bitmask |= ~(1 << warp_idx);
        }
if(blockIndex == 0)printf("KERNEL COMPLETE\n");
}


__host__
void GPU_Match_It(string regex, string text,int gridSize, int blockSize)
{
    // printf("Beginning GPU matching of regex %s with string %s\n", regex.c_str(), text.c_str());
    state* start_state = new state();
    printf("Parsing Regex\n");
    list<state*> state_list = parse_regex(regex,start_state);
    printf("Converting NFA into GPU friendly format.\n");
    vector<state*> nfa_vec = convert_state_list(state_list);
    // print_nfa(nfa_vec);
    // checkNFA(nfa_vec);
    nfa_t nfa_struct = convert_state_vec_d(nfa_vec);
    // describeNFA(nfa_struct);

    printf("Initializing out_vec in cuda memory\n");
    int* out_vec;
    CUDA_CALL(cudaMalloc((void**) &out_vec, 2*MAX_NUMBER_OF_MATCHES*sizeof(int)));
    CUDA_CALL(cudaMemset((void*) out_vec,-1,  2*MAX_NUMBER_OF_MATCHES*sizeof(int)));
    printf("\tInitializing out counter as well\n");
    int* out_counter = 0;
    cudaMalloc((void**)&out_counter,sizeof(int));
    cudaMemset((void*) out_counter,0,sizeof(int) );

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
    CUDA_CALL(cudaMemcpy(nfa_tr, &(nfa_struct.transitions[0]), sizeof(d_transition)*nfa_struct.n_transitions, cudaMemcpyHostToDevice));
    
    printf("Initializing d_text in cuda memory\n");
    char* d_text;
    CUDA_CALL(cudaMalloc((void**) &d_text, sizeof(char)*text.length()) );
    CUDA_CALL(cudaMemcpy(d_text, text.c_str(), text.length(), cudaMemcpyHostToDevice));


    // printf("The d_string: %s", d_text);
    size_t extern_size = n*(text.length()/(gridSize-1)) * sizeof(int) + sizeof(bool);
    if(extern_size >= 48000)
    {
        char error_message[200];
        sprintf(error_message, "You are requesting %zu bytes of data however there is a hard limit on shared memory per state machine of 48000 bytes. Please assign more blocks.\n",extern_size);
        throw invalid_argument(error_message);
    }
    printf("The amount of shared memory we're requesting is %zu\n\n", extern_size);
    cudaDeviceSynchronize();
    RMatchKernel<<<gridSize, blockSize,extern_size>>>(nfa_st, nfa_tr, d_text,nfa_struct.n_states,text.length(), out_vec,out_counter);
    cudaError_t execErr = cudaGetLastError();
    if (execErr != cudaSuccess) {
            printf("Execution Error: %s\n", cudaGetErrorString(execErr));
    }

    printf("Copying memory off the device back onto the host.\n");
    int *h_out_vec = new int[2*MAX_NUMBER_OF_MATCHES];
    printf("h_out_vec pointer %p, out_vec points %p\n ", h_out_vec, out_vec);
    cudaMemcpy((void*) h_out_vec, (void*) out_vec, 2*MAX_NUMBER_OF_MATCHES*sizeof(int), cudaMemcpyDeviceToHost);
    int *h_out_counter = new int;
    cudaMemcpy((void*) h_out_counter, (void*) out_counter,sizeof(int), cudaMemcpyDeviceToHost);
    printf("print %d matches\n",*h_out_counter);
    // for(int i =0; i < MAX_NUMBER_OF_MATCHES; i++)
    // {
    //     if(h_out_vec[2*i+1] > 0)
    //     {
    //         printf("match found at idx %d of length %d: %s\n",h_out_vec[2*i], h_out_vec[2*i+1],text.substr(h_out_vec[2*i], h_out_vec[2*i+1]).c_str());
    //     }
    // }
}
