#include "gpu_regex.cuh"
#include <string>
using namespace std;

int main()
{
    GPU_Match_It(string("hello"), string("hello world"), 1,1);
}