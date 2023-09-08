# Variables to control Makefile operation
 
CXX = g++
#CXXFLAGS = -Wall -g
CXXFLAGS = -Wall -O3 -fomit-frame-pointer
#CXXFLAGS = -Wall -O3 -fomit-frame-pointer -march=native
 
mixing: mixing.o wcsp.o wcspfun.o wcspvar.o solution.o
	$(CXX) $(CXXFLAGS) -o mixing mixing.o wcsp.o wcspfun.o wcspvar.o solution.o
 
mixing.o: mixing.cpp wcsp.h wcspreader.hh wcspfun.h wcspvar.h
	$(CXX) $(CXXFLAGS) -c mixing.cpp
 
wcsp.o: wcsp.cpp wcsp.h wcspfun.h wcspvar.h
	$(CXX) $(CXXFLAGS) -c wcsp.cpp

wcspfun.o: wcspfun.cpp wcspfun.h wcspvar.h
	$(CXX) $(CXXFLAGS) -c wcspfun.cpp

wcspvar.o: wcspvar.cpp wcspvar.h
	$(CXX) $(CXXFLAGS) -c wcspvar.cpp

solution.o: solution.cpp
	$(CXX) $(CXXFLAGS) -c solution.cpp

clean:
	rm -rf *.o

mrproper:
	rm -rf mixing
