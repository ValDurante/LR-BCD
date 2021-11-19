//wcspvar.h
#include <vector>

#ifndef WCSPVAR_H
#define WCSPVAR_H

using namespace std;

class wcspvar

{
private:
    vector<int> _domain;
    int _value;
    bool _assigned;

public:
    wcspvar(size_t);
    wcspvar();
    ~wcspvar(void);

    void setValue(int);
    void setDomain(vector<int>);
    int getValue();
    const vector<int>& getDomain();
    bool isAssigned();
    void removeDomain();
    size_t domainSize();
    size_t relatDomainSize();
    void resetVariable();

};

#endif