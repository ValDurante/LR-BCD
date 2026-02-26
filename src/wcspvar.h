//wcspvar.h
#ifndef WCSPVAR_H
#define WCSPVAR_H

#include <vector>

using namespace std;
class wcspfun;
class wcspvar

{
private:
    vector<int> _domain;
    int _value;
    bool _assigned;

public:
    vector<wcspfun *> functions;

    wcspvar(size_t);
    wcspvar();
    ~wcspvar(void);

    void addFunction(wcspfun *f) { functions.push_back(f); };

    void setValue(int, bool);
    void setDomain(vector<int>, bool);
    int getValue();
    const vector<int> &getDomain();
    bool isAssigned();
    void removeDomain();
    size_t domainSize();
    size_t relatDomainSize();
    void resetVariable();
};

#endif