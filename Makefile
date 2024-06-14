# Compiler to use
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -g

# Build target
TARGET = cpu_demo_2

# Object files directory
OBJDIR = executables

# Object files
OBJS = $(OBJDIR)/cpu_demo_2.o $(OBJDIR)/nfa.o

# Linking phase specification
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Compilation phase specification for each object file
$(OBJDIR)/cpu_demo_2.o: cpu_demo_2.cpp nfa.hpp
	$(CXX) $(CXXFLAGS) -c cpu_demo_2.cpp -o $@

$(OBJDIR)/nfa.o: nfa.cpp nfa.hpp
	$(CXX) $(CXXFLAGS) -c nfa.cpp -o $@

# Clean up
clean:
	rm -f $(TARGET)
	rm -rf $(OBJDIR)

# Make clean and then make target
rebuild: clean
