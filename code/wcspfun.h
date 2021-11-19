//wcspfun.h
#include <vector>
#include "wcspvar.h"

#ifndef WCSPFUN_H
#define WCSPFUN_H

using namespace std;

class wcspfun

{
private:
    vector<wcspvar*> _scope;
    vector<double> _costs;
    vector<int> _indices;

public:
    wcspfun(vector<wcspvar*>,vector<double>,vector<int>);
    wcspfun();
    ~wcspfun(void);

    const vector<wcspvar*>& getScope();
    const vector<double>& getCosts();
    const vector<int>& getIndices();
    double getUnaryValue(int);
    double getBinaryValue(int,int);
    double getMin();
    double getMax();
    double getMaxInit();
    double getAssignment();
    vector<int> validValues();

};

#endif