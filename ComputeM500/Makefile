# Compiler and flags
CXX = mpicxx
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra -I.  # -I. to find headers in current dir

# Source files
SRCS = main.cpp IO.cpp Particles.cpp
OBJS = $(SRCS:.cpp=.o)

# Target executable
TARGET = compute_m500

# Default rule
all: $(TARGET)

# Link step
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile step with header dependencies
%.o: %.cpp IO.H Particles.H
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)

