#ifndef __WCSPREADER_HH
#define __WCSPREADER_HH

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <tuple>
#include <numeric>
#include <cstdlib>

#include "wcspvar.h"
#include "wcspfun.h"
#include "wcsp.h"

using namespace std;

typedef double Cost;

template<typename T>
vector<T> read_vec(istream& is)
{
    vector<T> r;
    T s;
    is >> s;
    while(is) {
        r.push_back(s);
        is >> s;
    }
    return r;
}

template<typename T>
vector<T> read_vec(string const& line)
{
    istringstream iss(line);
    return read_vec<T>(iss);
}

tuple<string, size_t, size_t, size_t, Cost> read_header(string const& line)
{
    istringstream iss(line); 

    string name;
    size_t nvars;
    size_t domsize;
    size_t nfun;
    Cost ub;

    iss >> name >> nvars >> domsize >> nfun >> ub;
    return make_tuple(name, nvars, domsize, nfun, ub);
}

vector<wcspvar*> read_variables(istream& is)
{
    string line;
    getline(is, line);
    vector<size_t> domains = read_vec<size_t>(line);
    vector<wcspvar*> variables;

    size_t nvars = domains.size();

    for(size_t i = 0; i != nvars; ++i){
        size_t dom = domains[i];
        wcspvar* var = new wcspvar(dom);
        variables.push_back(var);
    } 

    return variables;
}

wcspfun* read_fun(istream& is, vector<wcspvar*> const& variables, double& lb)
{
    string line;
    getline(is, line);
    vector<Cost> hd = read_vec<Cost>(line);

    size_t arity = hd[0];
    size_t nspec = hd[hd.size()-1];
    Cost defcost = hd[hd.size()-2];

    vector<wcspvar*> scope;
    vector<int> indices;
    wcspfun *pfun = NULL;

    if (arity == 0)
    {
        lb = hd[1];
    }

    else
    {
        for(size_t i = 1; i != 1 + arity; i++){
            int var = hd[i];
            scope.push_back(variables[var]);
            indices.push_back(var);
        }

        size_t size;
        if(arity == 1){
            size = scope[0]->domainSize();   
        }
        else
        {
            size = scope[0]->domainSize()*scope[1]->domainSize();
        }

        vector<Cost>  costs(size,defcost);

        if(nspec == size)
        {
            for(size_t i = 0; i != nspec; ++i) {
                getline(is, line);
                vector<Cost> v = read_vec<Cost>(line);
                costs[i] = v[v.size()-1];
            }
        }
        else
        {
            if (arity == 1)
            {
                for(size_t i = 0; i != nspec; ++i) {
                    getline(is, line);
                    vector<Cost> v = read_vec<Cost>(line);
                    costs[v[0]] = v[v.size()-1];     
                }
            }
            else
            {
                size_t domY = scope[1]->domainSize();

                for(size_t i = 0; i != nspec; ++i) {
                    getline(is, line);
                    vector<Cost> v = read_vec<Cost>(line);
                    costs[v[0]*domY + v[1]] = v[v.size()-1];  
                    
                }

            }
            
        }

        pfun = new wcspfun(scope, costs, indices);
    }
    return pfun;
}

wcsp readwcsp(istream& is)
{

    string name;
    size_t nvars;
    size_t domsize;
    size_t nfun;
    Cost ub;
    Cost lb = 0;

    string line;

    getline(is, line);
    tie(name, nvars, domsize, nfun, ub) = read_header(line);
  
    vector<wcspvar*> variables = read_variables(is);
    vector<wcspfun*> functions;

    for(size_t i = 0; i != nfun; ++i)
    {
        functions.push_back(read_fun(is,variables,lb));
    }

    if(!functions[nfun-1]) //remove the constant fun
    {
        functions.pop_back();
    }
  
    wcsp w = wcsp(ub,lb,variables,functions);

    return w;
}

#endif
