//wcsp.cpp
#include <iostream>
#include <math.h>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include "wcspfun.h"
#include "wcspvar.h"
#include "wcsp.h"

using namespace std;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::VectorXd DnVec;
typedef Eigen::MatrixXd DnMat;

wcsp::wcsp(double ub, double lb, const vector<wcspvar *> &variables, const vector<wcspfun *> &functions)
{
    _ub = ub;
    _lb = lb;
    _variables = variables;
    _functions = functions;

    size_t fun_size = _functions.size();
    for (size_t i = 0; i < fun_size; i++)
    {
        if (_functions[i]->getIndices().size() == 2)
        {
            _variables[_functions[i]->getIndices()[0]]->addFunction(_functions[i]);
            _variables[_functions[i]->getIndices()[1]]->addFunction(_functions[i]);
        }
        else
        {
            _variables[_functions[i]->getIndices()[0]]->addFunction(_functions[i]);
        }
    }
}

wcsp::wcsp()
{
    _ub = 0;
    _lb = 0;
    _variables = vector<wcspvar *>();
    _functions = vector<wcspfun *>();
}

wcsp::~wcsp()
{
}

double wcsp::getUpperBound()
{
    return _ub;
}

double wcsp::getLowerBound()
{
    return _lb;
}

void wcsp::setUpperBound(double lb)
{
    _lb = lb;
}

const vector<wcspvar *> &wcsp::getVariables()
{
    return _variables;
}

const vector<wcspfun *> &wcsp::getFunctions()
{
    return _functions;
}

size_t wcsp::getSDPSize()
{
    size_t d = 0;
    size_t nbVar = _variables.size();

    for (size_t i = 0; i != nbVar; ++i)
    {
        d = d + _variables[i]->relatDomainSize(); //relatDomainSize() for the nodes or domainSize() for the root
    }

    return d;
}

size_t wcsp::getRank(int m)
{
    size_t n = this->getSDPSize() + 1;

    // add the N exactly one constraints for the BCD method
    if (m == 2)
    {
        n = n + _variables.size();
    }

    size_t k = ceil(sqrt(2 * n));

    return k;
}

void wcsp::assignmentUpdate(const vector<vector<int>> &assignment, bool tag)
{
    size_t nbVar = assignment.size();

    for (size_t i = 0; i < nbVar; i++)
    {
        _variables[i]->setDomain(assignment[i], tag);
    }
}

void wcsp::resetWcsp()
{
    size_t nbVar = _variables.size();
    for (size_t i = 0; i != nbVar; i++)
    {
        _variables[i]->resetVariable();
    }
}

/*
domains[i] = \sum_j^i d_j
Will be usefull for the cost matrix
*/
vector<int> wcsp::domains()
{
    int acc = 0;
    vector<int> dom;
    dom.push_back(acc);
    const vector<wcspvar *> &pVar = this->getVariables();

    for (size_t i = 0; i != (pVar.size()); i++)
    {
        acc = acc + pVar[i]->domainSize();
        dom.push_back(acc);
    }

    return dom;
}

vector<int> wcsp::relatDomains()
{
    int acc = 0;
    vector<int> dom;
    dom.push_back(acc);
    const vector<wcspvar *> &pVar = this->getVariables();

    for (size_t i = 0; i != (pVar.size()); i++)
    {
        if (!pVar[i]->isAssigned())
        {
            acc = acc + pVar[i]->relatDomainSize();
            dom.push_back(acc);
        }
    }

    return dom;
}

/*
Build the SDP cost matrix
*/

DnVec wcsp::unaryCostVector()
{
    size_t d = this->getSDPSize();
    vector<int> domains = this->domains();

    DnVec U(d);

    //U.reserve(d);

    for (size_t i = 0; i != _functions.size(); i++)
    {
        const vector<int> &pInd = _functions[i]->getIndices();

        //check if the function is a unary cost function
        if (pInd.size() == 1)
        {
            int ind = domains[pInd[0]];
            vector<double> costs = _functions[i]->getCosts();

            for (size_t j = 0; j != costs.size(); j++)
            {
                if (costs[j] != 0)
                {
                    //U.insert(ind + j) = costs[j];
                    U(ind + j) = costs[j];
                }
            }
        }
    }

    return U;
}

DnMat wcsp::binaryCostMatrix()
{
    size_t d = this->getSDPSize();
    vector<int> domains = this->domains();

    DnMat C(d, d);

    /*
    size_t domSize = _variables[0]->domainSize();
    size_t nbFun = _functions.size();
    int estimation_of_entries = round(domSize * domSize * nbFun);
    C.reserve(estimation_of_entries);
    */

    for (size_t i = 0; i != _functions.size(); i++)
    {
        const vector<int> &pInd = _functions[i]->getIndices();

        //check if the function is a binary cost function
        if (pInd.size() == 2)
        {
            const vector<wcspvar *> &scope = _functions[i]->getScope();
            size_t domY = scope[1]->domainSize();
            int indX = domains[pInd[0]];
            int indY = domains[pInd[1]];
            vector<double> costs = _functions[i]->getCosts();

            for (size_t j = 0; j != costs.size(); j++)
            {
                if (costs[j] != 0)
                {
                    int q = j / domY;
                    int r = j % domY;
                    //C.insert(indX + q, indY + r) = costs[j];
                    C(indX + q, indY + r) = costs[j];
                }
            }
        }
    }

    return C;
}

/*
Build the SDP cost matrix from the unary cost vector
and the binary cost matrix.
For the MAXCUT reformulation :
C -> (1/4)*C
U -> (U + C*e)/2
*/

DnMat wcsp::SDPmat()
{
    DnVec U = this->unaryCostVector();
    DnMat C = this->binaryCostMatrix();
    size_t d = this->getSDPSize();
    DnMat Q(d + 1, d + 1);

    /*
    size_t domSize = _variables[0]->domainSize();
    size_t nbFun = _functions.size();
    int estimation_of_entries = round(domSize * domSize * nbFun);
    Q.reserve(estimation_of_entries);
    */

    C = 0.5 * (DnMat(C.transpose()) + C);
    DnMat C_m = 0.25 * C;
    DnVec e = DnVec::Ones(d);
    DnVec U_m = 0.5 * (U + C * e);
    U_m = 0.5 * U_m;

    /*
    for (int k = 0; k < C_m.outerSize(); ++k)
    {
        for (SpMat::InnerIterator it(C_m, k); it; ++it)
        {
            Q.insert(it.row(), it.col()) = it.value();
        }
    }

    for (int i = 0; i != U_m.size(); i++)
    {
        if (U_m(i) != 0)
        {
            Q.insert(i, d) = U_m(i);
            Q.insert(d, i) = U_m(i);
        }
    }
    */

    Q.block(0, 0, d, d) = C_m;
    Q.block(0, d, d, 1) = U_m;
    Q.block(d, 0, 1, d) = U_m.transpose();

    return Q;
}

SpMat wcsp::constMat()
{
    int N = _variables.size();
    int d = this->getSDPSize();
    vector<int> domains = this->domains();

    SpMat A(N, d);
    A.reserve(d);

    for (size_t i = 0; i != domains.size() - 1; i++)
    {
        for (int j = domains[i]; j != domains[i + 1]; j++)
            A.insert(i, j) = 1;
    }

    return A;
}

DnVec wcsp::rhs()
{
    int N = _variables.size();
    DnVec b = DnVec::Ones(N);

    return b;
}

// build gangster operator constraint matrix
DnMat wcsp::gOperator()
{
    int d = this->getSDPSize();
    SpMat A = this->constMat();
    DnMat M(d + 1, d + 1);

    DnMat AA = DnMat(A.transpose()) * A;
    DnVec e = DnVec::Ones(d);

    DnVec diag = AA.diagonal();
    DnMat AAdiag = diag.asDiagonal();
    DnMat H = AA - AAdiag;

    DnMat G = 0.25 * H;
    DnVec q = 0.5 * H * e;
    double r = 0.25 * e.transpose() * H * e;

    M.block(0, 0, d, d) = G;
    M.block(0, d, d, 1) = 0.5 * q;
    M.block(d, 0, 1, d) = 0.5 * q.transpose();
    M(d, d) = r;

    return M;
}

// build dual mat
DnMat wcsp::dualMat()
{
    int d = this->getSDPSize();
    double pc = this->penaltyCoeff();

    DnVec b = this->rhs();
    SpMat A = this->constMat();
    DnVec U = this->unaryCostVector();
    DnMat C = this->binaryCostMatrix();
    DnMat Q(d + 1, d + 1);

    DnVec e = DnVec::Ones(d);
    DnVec temp = 0.5 * A * e;
    //DnVec temp = A*e;
    b = b - temp;
    A = 0.5 * A;

    C = 0.5 * (DnMat(C.transpose()) + C);
    DnMat C_m = 0.25 * C;
    DnVec U_m = 0.5 * (U + C * e);
    //U_m = 0.5*U_m;

    DnMat AA = DnMat(A.transpose()) * A;
    DnMat F = (2 * pc + 1) * AA + C_m;
    double bb = b.dot(b);
    DnVec bA = 2 * (2 * pc + 1) * (b.transpose() * A).transpose();
    DnVec c = 0.5 * (U_m - bA);

    Q.block(0, 0, d, d) = F;
    Q.block(0, d, d, 1) = c;
    Q.block(d, 0, 1, d) = c.transpose();
    Q(d, d) = (2 * pc + 1) * bb;

    return Q;
}

// Build the symmetric matrix for the exactly one constraint
DnMat wcsp::exactlyOne()
{
    int d = this->getSDPSize();
    DnMat F(d + 1, d + 1);
    DnVec e = DnVec::Ones(d);

    F.block(0, d, d, 1) = e;
    F.block(d, 0, 1, d) = e.transpose();

    return F;
}

//Return the penalty coeff for constraint dualization
double wcsp::penaltyCoeff()
{
    double penaltyCoeff = 0;
    const vector<wcspfun *> &pfun = this->getFunctions();
    size_t nbFun = pfun.size();

    for (size_t i = 0; i < nbFun; i++)
    {
        penaltyCoeff = penaltyCoeff + pfun[i]->getMaxInit(); //pfun[i]->getMax();
    }

    return penaltyCoeff;
}

vector<int> wcsp::validRows()
{
    const vector<wcspvar *> &pVar = this->getVariables();
    vector<int> domains = this->domains();
    vector<int> vRows;

    for (size_t i = 0; i != pVar.size(); i++)
    {
        int curr_dom = domains[i];
        if (pVar[i]->isAssigned())
        {
            continue;
        }
        else
        {
            vector<int> dom = pVar[i]->getDomain();

            for (size_t j = 0; j != dom.size(); j++)
            {
                if (dom[j] == 0)
                {
                    vRows.push_back(curr_dom + j);
                }
            }
        }
    }

    return vRows;
}