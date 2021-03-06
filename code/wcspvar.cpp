//wcspvar.cpp
#include "wcspvar.h"
#include <iostream>
#include <algorithm>

using namespace std;

/*
How do we deal with the varaible domain :
domain[i] = 1 if the variable takes value i
domain[i] = -1 if the value i is removed from the domain variable
domain[i] = 0 otherwise
*/

wcspvar::wcspvar(size_t domSize)
{
    _domain = vector<int>(domSize);
    _value = 0;
    _assigned = false;
}

wcspvar::wcspvar()
{
    _domain = vector<int>();
    _value = 0;
    _assigned = false;
}

wcspvar::~wcspvar(void)
{   
}

//Tag usefull for preprocessing and multiple rounding;
//find another solution to make it more robust
void wcspvar::setValue(int value, bool tag)
{
    _value = value;
    //Change it for the B&B solver
    if (tag)
    {
        _assigned = true;
    }
    else
    {
        _assigned = false; 
    }
    
    _domain[value] = 1;
}

void wcspvar::setDomain(vector<int> dom, bool tag)
{
    _domain = dom;
    int domSize = this->domainSize();
    
    //find another solution to make it more robust
    if (!_assigned)
    {
        for (int i = 0; i<domSize ; i++)
        {
            if (_domain[i] == 1)
            {
                this->setValue(i, tag);
            }   
        }

        //Other case that has to be considered
        int nbRm = count(dom.begin(),dom.end(),-1);
        if (nbRm == domSize - 1)
        {
            this->setValue(domSize-1, tag);
        }
    }
}

int wcspvar::getValue()
{
    return _value;
}

const vector<int>& wcspvar::getDomain()
{
    return _domain;
}

bool wcspvar::isAssigned()
{
    return _assigned;
}

void wcspvar::removeDomain()
{
    if (_domain.empty() != true)
    {
        _domain.pop_back();
    }
    else 
    {
        cout << "le domaine est vide";
    }
}

size_t wcspvar::domainSize()
{
    return _domain.size();
}


/*
If the variable is assigned then we set the relative
domain size to zero, else, the relative domain size is
equal to the number of zeros in the _domain vector.
*/
size_t wcspvar::relatDomainSize()
{
    size_t relatDomainSize = 0;

    if(!this->isAssigned())
    {
        int nbZeros = count(_domain.begin(),_domain.end(),0);
        relatDomainSize = nbZeros;
    }

    return relatDomainSize;
}

void wcspvar::resetVariable()
{
    size_t domSize = _domain.size();
    for(size_t i = 0; i != domSize; i++)
    {
        _domain[i] = 0;
    }
    _value = 0;
    _assigned = false;
}