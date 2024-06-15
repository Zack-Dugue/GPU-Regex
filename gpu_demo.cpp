#include "gpu_regex.cuh"
#include <string>
using namespace std;

int main()
{
    // printf("TRY TO MATCH REG: 'world' with STRING 'hello world'\n\n");
    // GPU_Match_It(string("world"), string("hello world"), 1,1);
    // printf("TRY TO MATCH REG: 'world|hello' with STRING 'hello world'\n\n");
    // GPU_Match_It(string("world|hello"), string("hello world"), 1,1);
    // printf("TRY TO MATCH REG: 'wor@ld' with STRING 'hello world'\n\n");
    // GPU_Match_It(string("wor@ld"), string("hello world"), 1,1);
    printf("TRY TO MATCH REG: 'l*' with STRING 'hello world'\n\n");
    GPU_Match_It(string("l*"), string("hello world"), 1,1);
}