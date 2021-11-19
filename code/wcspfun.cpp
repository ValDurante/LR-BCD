//wcspfun.cpp
#include "wcspfun.h"
#include "wcspvar.h"
#include <iostream>
#include <algorithm>

using namespace std;

wcspfun::wcspfun(vector<wcspvar*> scope,vector<double> costs, vector<int> indices)
{
    _scope = scope;
    _costs = costs;
    _indices = indices;
}

wcspfun::wcspfun()
{
    _scope = vector<wcspvar*>();
    _costs = vector<double>();
    _indices = vector<int>();
}

wcspfun::~wcspfun()
{

}

const vector<wcspvar*>& wcspfun::getScope()
{
    return _scope;
}

const vector<double>& wcspfun::getCosts()
{
    return _costs;
}

const vector<int>& wcspfun::getIndices()
{
    return _indices;
}

double wcspfun::getUnaryValue(int i)
{
    return _costs[i];
}

double wcspfun::getBinaryValue(int i, int j)
{
    size_t domY = _scope[1]->domainSize();
    return _costs[i*domY + j];
}

double wcspfun::getMin()
{ 
    vector<int> validValues = this->validValues();
    size_t size = validValues.size();
    double min = _costs[validValues[0]];
    
    if(size > 1)
    {
        for(size_t i = 1; i != validValues.size(); i ++)
        {   
            double currentValue = _costs[validValues[i]];
            if(currentValue < min)
            {
                min = currentValue;
            }
        }
    }
    
    return min;
}

double wcspfun::getMax()
{
    vector<int> validValues = this->validValues();
    size_t size = validValues.size();
    double max = _costs[validValues[0]];

    if(size > 1)
    {
        for(size_t i = 1; i != validValues.size(); i ++)
        {   
            double currentValue = _costs[validValues[i]];
            if(currentValue > max)
            {
                max = currentValue;
            }
        }
    }
    
    return max;
}

double wcspfun::getMaxInit()
{
    double max = *max_element(_costs.begin(),_costs.end());

    return max;
}

double wcspfun::getAssignment()
{
    double value;
    if (_scope.size()==2)
    {
        size_t domY = _scope[1]->domainSize();
        int i = _scope[0]->getValue();
        int j = _scope[1]->getValue();
        value = _costs[i*domY + j]; 
    }
    else
    {
        int i = _scope[0]->getValue();
        value = _costs[i];
    }
    
    return value;
}

/*
Returns a vector of indices which correspond to values that are
not removed from the variable domain. 
*/
vector<int> wcspfun::validValues()
{
    vector<int> validValues;
    if (_scope.size()==2)
    {
        bool assignedX = _scope[0]->isAssigned();
        bool assignedY = _scope[1]->isAssigned();
        const vector<int>& pDomX = _scope[0]->getDomain();
        const vector<int>& pDomY = _scope[1]->getDomain();
        size_t domX = _scope[0]->domainSize();
        size_t domY = _scope[1]->domainSize();

        if (assignedX && assignedY)
        {
            int i = _scope[0]->getValue();
            int j = _scope[1]->getValue();
            validValues.push_back(i*domY + j); 
        }

        else if (assignedX && !assignedY)
        {
            int i = _scope[0]->getValue();
            for(size_t j = 0; j!=domY; j++)
            {
                if(pDomY[j] != -1)
                {
                    validValues.push_back(i*domY + j);
                }
            }
        }

        else if (!assignedX && assignedY)
        {
            int j = _scope[1]->getValue();
            for(size_t i = 0; i!= domX; i++)
            {
                if(pDomX[i] != -1)
                {
                    validValues.push_back(i*domY + j);
                }
            }
        }

        else
        {
            for(size_t i = 0; i != domX; i++)
            {
                for (size_t j = 0; j!=domY; j++)
                {
                    if (pDomX[i] != -1 && pDomY[j] != -1)
                    {
                        validValues.push_back(i*domY + j);
                    }
                }
            }
        }

    }
    else 
    {
        bool assignedX = _scope[0]->isAssigned();
        const vector<int>& pDomX = _scope[0]->getDomain();
        size_t domX = _scope[0]->domainSize();

        if(assignedX)
        {   
            int i = _scope[0]->getValue();
            validValues.push_back(i);
        }
        else
        {
            for(size_t i = 0; i != domX; i++)
            {
                if(pDomX[i] != -1)
                {
                    validValues.push_back(i);
                }
            }
        }
    }

    return validValues;
}
