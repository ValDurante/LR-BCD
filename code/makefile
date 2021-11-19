# Variables to control Makefile operation
 
CXX = g++
CXXFLAGS = -Wall -g
LIB = -I/EigenPath
 
mixing: mixing.o wcsp.o wcspfun.o wcspvar.o
	$(CXX) $(CXXFLAGS) $(LIB) -o mixing mixing.o wcsp.o wcspfun.o wcspvar.o
 
mixing.o: mixing.cpp wcsp.h wcspreader.hh wcspfun.h wcspvar.h
	$(CXX) $(CXXFLAGS) $(LIB) -c mixing.cpp
 
wcsp.o: wcsp.cpp wcsp.h wcspfun.h wcspvar.h
	$(CXX) $(CXXFLAGS) $(LIB) -c wcsp.cpp

wcspfun.o: wcspfun.cpp wcspfun.h wcspvar.h
	$(CXX) $(CXXFLAGS) -c wcspfun.cpp

wcspvar.o: wcspvar.cpp wcspvar.h
	$(CXX) $(CXXFLAGS) -c wcspvar.cpp

clean:
	rm -rf *.o

mrproper:
	rm -rf mixing
