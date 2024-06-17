#include "gpu_regex.cuh"
#include "match.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <tuple>
#include <chrono>
using namespace std;

tuple<string,string> get_file(string filePath)
{
    

    // Open the file using ifstream
    ifstream file(filePath);

    // confirm file opening
    if (!file.is_open()) {
        // print error message and return
        cerr << "Failed to open file: " << filePath << endl;
        throw invalid_argument("We failed to open this file gang");
    }

    // Read the file line by line into a string
    string regex;
    getline(file, regex);

    // Read the rest of the file line by line into a string called line
    string line, content;
    while (getline(file, line)) {
        content += line + "\n";  // Appends each line to content with a newline
    }
    // Close the file
    file.close();

    return {content, regex };
}


void gpu_experiment(string path,int grid_size, int num_threads)
{
    auto[text,regex] = get_file(path);
    auto start = chrono::high_resolution_clock::now();
   
    GPU_Match_It(regex,text,grid_size,num_threads);

    auto finish = std::chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> elapsed = finish - start;
    cout << "Elapsed time running GPU REGEX: " << elapsed.count() << " ms\n";

}


void cpu_experiment(string path)
{
    auto[text,regex] = get_file(path);
    auto start = chrono::high_resolution_clock::now();
   
    match_this_regex(regex,text);

    auto finish = std::chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> elapsed = finish - start;
    cout << "Elapsed time running CPU REGEX: " << elapsed.count() << " ms\n";
    // printf("\n\n\n TRY WITH PARTIAL TEXT\n\n\n");
    // GPU_Match_It(regex,text.substr(0,128),grid_size, num_threads);

}

int main()
{
    string path;
    int num_blocks;
    int num_threads;

    // while(true)
    // {
        // Prompt for file path
        // printf("Please give the path of a file. That file should have the desired regex as its first line.\n");
        // printf("To exit, type 'exit'.\n> ");
        // getline(cin, path);

        // Check if the user wants to exit
        path = "/home/zdugue/cs179/project/GPU-Regex/data/sonnets_hard.txt";
        // if(path == "exit") {
        //     break;
        // }

        // Run the CPU regex matcher
        // printf("\nRunning CPU regex matcher\n");

        // Get the number of blocks
        // printf("Enter the number of blocks:\n> ");
        // scanf("%d", &num_blocks);
        num_blocks=256;

        // Clean up the newline character left by scanf
        // cin.ignore(numeric_limits<streamsize>::max(), '\n');

        // Get the number of threads
        printf("\nEnter the number of threads:\n> ");
        scanf("%d", &num_threads);

        // Clean up the newline character left by scanf
        cin.ignore(numeric_limits<streamsize>::max(), '\n');

        // Placeholder: Output the entered values
        printf("You entered %d blocks and %d threads.\n", num_blocks, num_threads);
        printf("Running the gpu regex engine:\n");
        gpu_experiment(path, num_blocks,num_threads);
        printf("Running the CPU regex enginge");
        cpu_experiment(path);
    // }

    return 0;
}


    // printf("starting GPU experiment\n");
    // gpu_experiment("/home/zdugue/cs179/project/GPU-Regex/data/sonnets_hard.txt",2048,32);

    // printf("starting CPU experiment\n");
    // cpu_experiment("/home/zdugue/cs179/project/GPU-Regex/data/sonnets_hard.txt");
    // printf("starting GPU experiment\n");
    // gpu_experiment("/home/zdugue/cs179/project/GPU-Regex/data/sonnets_hard.txt",512,256);

    // printf("TRY TO MATCH REG: 'world' with STRING 'hello world'\n\n");
    // GPU_Match_It(string("world"), string("hello_world"),3,3);
    //     printf("TRY TO MATCH REG: 'hello|world' with STRING 'hello world'\n\n");
    // GPU_Match_It(string("hello|world"), string("hello world"), 3,3);
    // printf("TRY TO MATCH REG: 'world|hello' with STRING 'hello world'\n\n");
    // GPU_Match_It(string("world|hello"), string("hello world"), 4,2);
    // printf("TRY TO MATCH REG: 'wor@ld' with STRING 'hello world'\n\n");
    // GPU_Match_It(string("wor@ld"), string("hello world"), 3,1);
    // printf("TRY TO MATCH REG: 'l*' with STRING 'hello world'\n\n");
    // GPU_Match_It(string("l*"), string("hello world"), 2,2);
    // printf("TRY TO MATCH REG: 'hello' with STRING 'hello hello hello bruh bruh hello hello'\n\n");
    // GPU_Match_It(string("hello"), string("hello hello hello bruh bruh hello hello"), 4,4);
    
    // printf("TRY TO MATCH REG: '.*' with STRING 'hello hello hello bruh bruh hello hello'\n\n");
    // GPU_Match_It(string(".*"), string("hello hello hello bruh bruh hello hello"), 4,4);

