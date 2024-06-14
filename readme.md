How to run the demo:

Simply run demo.exe and then input a regex you'd like to search. For the regex demo string lengths and regex lengths are limited to 1000 characters. There are also a few limitations from standard regex, but these are stated in the demo itself. 

Note that this does not depend on the C++ regex engine at all. The engine is entirely my own. This is because there really aren't a ton of great engines available that would work for the purposes of parallelization. This is because I need explicit access to a representation of the non deterministic finite automota in order to traverse it in parallel. 