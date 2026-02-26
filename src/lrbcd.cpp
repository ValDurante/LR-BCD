#ifndef __MIXING_HH
#define __MIXING_HH

#define _USE_MATH_DEFINES

#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <float.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <chrono>

#include "wcsp.h"
#include "wcspreader.hh"
#include "solution.cpp"

using namespace std;

const bool debug = false;

typedef Eigen::VectorXd DnVec;
typedef Eigen::MatrixXd DnMat;

const double MEPS = 1e-24;

static inline double cpuTime(void)
{
        struct rusage ru;
        getrusage(RUSAGE_SELF, &ru);
        return (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1000000;
}

DnMat mixingInit(wcsp &wcsp, int k)
{
        default_random_engine generator;
        generator.seed(time(0));
        // normal_distribution<double> distribution(0, 1); // normal distribution
        uniform_real_distribution<double> distribution(-1.0, 1.0); // uniform distribution

        size_t n = wcsp.getSDPSize() + 1;
        DnMat V = DnMat::Zero(k, n);

        for (size_t j = 0; j < n; j++)
        {
                for (int i = 0; i < k; i++)
                {
                        V(i, j) = distribution(generator);
                }
                V.col(j) = V.col(j) / (V.col(j).norm());
        }

        return V;
}

/*
Bias introduced by the change of variable {0,1} -> {-1,1}
*/
double bias(wcsp &wcsp)
{
        double unaryBias = 0;
        double binaryBias = 0;
        double lb = wcsp.getLowerBound();
        const vector<wcspfun *> &pfun = wcsp.getFunctions();

        for (size_t i = 0; i < pfun.size(); i++)
        {
                const vector<wcspvar *> &pVar = pfun[i]->getScope();
                const vector<double> &pCost = pfun[i]->getCosts();
                vector<int> validValues = pfun[i]->validValues();

                if (pVar.size() == 2)
                {
                        if (pVar[0]->isAssigned() && pVar[1]->isAssigned())
                        {
                                size_t domY = pVar[1]->domainSize();
                                int i = pVar[0]->getValue();
                                int j = pVar[1]->getValue();
                                binaryBias += pCost[i * domY + j];
                        }
                        else
                        {
                                double acc = 0;
                                for (size_t j = 0; j < validValues.size(); j++)
                                {
                                        acc += pCost[validValues[j]];
                                }
                                binaryBias += 0.25 * acc;
                        }
                }
                else
                {
                        if (pVar[0]->isAssigned())
                        {
                                unaryBias += pCost[pVar[0]->getValue()];
                        }
                        else
                        {
                                double acc = 0;
                                for (size_t j = 0; j < validValues.size(); j++)
                                {
                                        acc += pCost[validValues[j]];
                                }
                                unaryBias += 0.5 * acc;
                        }
                }
        }

        double bias = binaryBias + unaryBias + lb;

        return bias;
}

vector<vector<int>> assignmentInit(wcsp &w)
{
        const vector<wcspvar *> &pvar = w.getVariables();
        size_t nbVar = pvar.size();
        vector<vector<int>> assignment;

        for (size_t i = 0; i < nbVar; i++)
        {
                size_t domSize = pvar[i]->domainSize();
                vector<int> domain(domSize);
                assignment.push_back(domain);
        }

        return assignment;
}

// draw a random vector from the standard normal distribution N(0,1)
// and normalize it
// In :
//      - DnVec &v
//      - size_t k
//      - default_random_engine &generator
//      - normal_distribution<Double> &distribution
void randUnit(DnVec& v, size_t k, default_random_engine& generator, normal_distribution<double>& distribution)
{
    assert(k > 0);

    for (size_t i = 0; i < k; i++) {
        v(i) = distribution(generator);
    }

    v = v / v.norm();
}

DnVec rounding(const DnMat& V, const vector<int>& domains, size_t n, size_t k, int i)
{
    assert(n > 0);
    assert(k > 0);

    default_random_engine generator;
    generator.seed(i + 1);
    normal_distribution<double> distribution(0, 1);

    DnVec r = DnVec::Zero(k);
    DnVec intSol = DnVec::Constant(n, -1.0);
    intSol(n - 1) = 1; // last value for homogenization must be equal to 1

    randUnit(r, k, generator, distribution);

    for (size_t i = 0; i < domains.size() - 1; i++) {
        int first = domains[i];
        int last = domains[i + 1];

        int ind = first;
        double max = fabs(r.dot(V.col(first)));
        double previous_max = max;

        for (size_t j = first + 1; j < last; j++) {
            // compute new trial point
            max = fabs(r.dot(V.col(j)));

            if (max > previous_max) {
                previous_max = max;
                ind = j;
            }
        }

        intSol(ind) = 1;
    }

    return intSol;
}

template <typename T>
double evalFun(T const &C, const DnMat &V)
{
        DnMat A = C * V.transpose();
        A = V * A;
        double eval = A.trace();

        return eval;
}

// Evaluate vector objective function
// In :
//      - const DnMat &C: objective matrix
//      - const DnVec &V: vector
// Out :
//      - Double eval : function value
double evalFun(const DnMat& C, const DnVec& V)
{

    DnVec A = C * V;
    double eval = V.dot(A);

    return eval;
}

/*
Build the objective vectors for the block-optimization
sub-problems
in : 
- int first : first block index
- int last : last block index  
- const SpMat& C : cost matrix
- const DnMat& V : SDP variable
- int k : rank of V 
out :
- DnMat G : (k,-first)-matrix of objective vectors
*/
DnMat objectiveVec(int first, int last, const DnMat &C, const DnMat &V, int k)
{
        int d = last - first;
        DnMat G = DnMat::Zero(k, d);
        DnVec g(k);

        for (int i = first; i < last; i++)
        {
                g = V * C.col(i);
                G.col(i - first) = g;
        }

        return G;
}

DnVec relaxedSol(const DnMat &G, const DnVec &u, int d, int k)
{

        int n = k * d;
        DnVec g = DnVec::Zero(n);
        DnVec w = DnVec::Zero(n);

        for (int i = 0; i < n; i++)
        {
                int q = i / k;
                int r = i % k;

                g(i) = G(r, q);
                w(i) = u(r);
        }

        double r = 4 - (4. / d);

        double gamma = g.squaredNorm() - (1. / d) * pow(w.dot(g), 2);
        double beta = -sqrt(r / gamma);
        double alpha = -(1. / d) * (beta * w.dot(g) + d - 2);
        DnVec res = alpha * w + beta * g;

        return res;
}

double dualInit(const DnMat &G, const DnVec &u, const DnVec &rSol, int k)
{
        DnVec x = rSol.head(k);
        DnVec g = G.col(0);

        double nrm_g = g.norm();
        double nrm_x = x.norm();
        double angle_g_x = acos(g.dot(x) / (nrm_g * nrm_x));
        double angle_u_x = acos(u.dot(x) / nrm_x);
        double dualInit = -nrm_g * sin(angle_g_x) / sin(angle_u_x);

        return dualInit;
}

DnMat vecSol(const DnMat &G, const DnVec &u, const double dualOpti, int d, int start, int k)
{

        DnMat res(k, d);
        for (int i = 0; i < d; i++)
        {
                DnVec g = G.col(i);
                DnVec sol = g + dualOpti * u;
                double nrm_sol = sol.norm();

                res.col(i) = - sol / nrm_sol;
        }

        return res;
}

double currentDelta(const DnMat &G, const DnMat &V, const DnMat &Vb, int first,
                    int last)
{
        double currDelta = 0;
        for (int i = first; i < last; i++)
        {
                currDelta += 2 * (V.col(i) - Vb.col(i - first)).dot(G.col(i - first));
        }

        return currDelta;
}

void solUpdate(DnMat &V, const DnMat &Vb, int first, int last)
{
        for (int i = first; i < last; i++)
        {
                V.col(i) = Vb.col(i - first);
        }
}

double jac_h(const DnMat &G, const DnVec &u, double x, int d)
{
        double res = 0;

        for (int i = 0; i < d; i++)
        {
                DnVec w = G.col(i) + x * u;
                res += (G.col(i).dot(u) + x) / w.norm();
        }
        res = res - (d - 2);

        return res;
}

double hess_h(const DnMat &G, const DnVec &u, double x, int d)
{
        double res = 0;

        for (int i = 0; i < d; i++)
        {
                DnVec g = G.col(i);
                DnVec w = g + x * u;
                res += (g.squaredNorm() - pow(g.dot(u), 2)) / pow(w.norm(), 3);
        }

        return res;
}

double newton(const DnMat &G, const DnVec &u, double x, int d, int maxiter, int &N_it)
{
        double x_curr = x;
        double jac = jac_h(G, u, x, d);
        double jac_curr = jac;
        int it = 0;
        double eps = pow(10, -7);

        double step_size = 0.80;
        double alpha = 0.60; // correction parameter for the step size

        
        while (abs(jac) > eps && it < maxiter)
        {
                x_curr = x_curr - step_size * jac_h(G, u, x_curr, d) / hess_h(G, u, x_curr, d);
                jac = jac_h(G, u, x_curr, d);
                it++;

                if (abs(jac) >= abs(jac_curr)) // step_size has to be updated
                {
                        step_size *= alpha; // reduce the step size
                        x_curr = x;
                        jac = jac_h(G, u, x, d);
                        it = 0;
                }

                jac_curr = jac;
        }
     
        N_it = N_it + it;

        return x_curr;
}

void solveBCD(const DnMat &C, DnMat &V, const DnVec &u, const vector<int> &domains, double &delta, int k, int i, int &N_it)
{
        int first = domains[i];
        int last = domains[i + 1];
        int d = last - first;
        int maxiter = 100;

        DnMat G = objectiveVec(first, last, C, V, k);

        //Solve relaxed optimization problem 
        DnVec solRelax = relaxedSol(G, u, d, k);
        double dual = dualInit(G, u, solRelax, k);

        //Dual opti
        double dualOpti = newton(G, u, dual, d, maxiter, N_it);

        //Build solution vectors
        DnMat Vb = vecSol(G, u, dualOpti, d, first, k);

        delta += currentDelta(G, V, Vb, first, last);
        solUpdate(V, Vb, first, last);
}

DnMat doBCDMixing(wcsp &wcsp, const DnMat &C, double tol, int maxiter, int k)
{
        vector<int> domains = wcsp.relatDomains();
        int d = wcsp.getSDPSize();
        int BCD_it = 0;
        int N_it = 0;
        DnVec q(d + 1);
        DnVec g(k);
        bool stop = false;

        // Stopping criteria
        double delta = 0;

        //V init
        DnMat V = mixingInit(wcsp, k);
        DnMat V_old = V;
        double f_val = evalFun(C, V); 

        for (int i = 0; i < maxiter; i++)
        {
                DnVec u = V.col(d);
                delta = 0;

                for (size_t j = 0; j < domains.size() - 1; j++)
                {
                        solveBCD(C, V, u, domains, delta, k, j, N_it);
                }

                BCD_it++;

                assert(delta > 0); // delta should always be positive
                DnMat V_diff = V - V_old;
                double fun_criterion = delta / (1 + fabs(f_val));
                double step_criterion = V_diff.norm() / (1 + V_old.norm()); 
                
                stop = (fun_criterion < tol) || (step_criterion < tol);

                V_old = V;
                f_val -= delta;

                if (stop)
                {
                        break;
                }
        }

        return V;
}

DnMat doMixing(wcsp &wcsp, const DnMat &Q, double tol, int maxiter, int k)
{
        int d = wcsp.getSDPSize();
        int it = 0;
        DnVec g(k);
        DnVec v(k);
        DnVec q(d + 1);
        bool stop = false;

        // Stopping criteria
        double delta = 0;

        //V init
        DnMat V = mixingInit(wcsp, k);
        DnMat V_old = V;
        double f_val = evalFun(Q, V);



        for (int i = 0; i < maxiter; i++)
        {
                delta = 0;
                for (int j = 0; j < d + 1; j++)
                {
                        q = Q.row(j);
                        q(j) = 0;

                        g = V * q;

                        double gnrm = g.norm();

                        // Ensure non-degeneration of the g-vector
                        if (gnrm < MEPS)
                        {
                                continue;
                        }

                        v = -g / gnrm;
                        delta += gnrm * (1 - V.col(j).dot(v)) * 2;
                        V.col(j) = v;
                }

                it++;

                assert(delta > 0); // delta should always be positive
                DnMat V_diff = V - V_old;
                double fun_criterion = delta / (1 + fabs(f_val));
                double step_criterion = V_diff.norm() / (1 + V_old.norm()); 
                
                stop = (fun_criterion < tol) || (step_criterion < tol);

                if (stop)
                {
                        break;
                }

                V_old = V;
                f_val -= delta;

        }

        return V;
}

DnMat preProcessing(DnMat &Q, wcsp &w)
{
        const vector<wcspvar *> &pVar = w.getVariables();
        vector<int> domains = w.domains();
        int n = Q.cols();

        //setValue makes _assigned = true
        bool tag = true;

        for (size_t i = 0; i < domains.size() - 1; i++)
        {
                int start = domains[i];
                int end = domains[i + 1];

                for (int j = start; j < end; j++)
                {
                        if (Q.col(j).isZero(0))
                        {
                                pVar[i]->setValue(j - start, tag);
                                break;
                        }
                }
        }

        vector<int> vRows = w.validRows();
        int nSize = vRows.size();
        DnMat Q_tmp(n, nSize + 1);
        DnMat Q_f(nSize + 1, nSize + 1);

        for (int i = 0; i < nSize; i++)
        {
                Q_tmp.col(i) = Q.col(vRows[i]);
        }
        Q_tmp.col(nSize) = Q.col(n - 1);

        for (int i = 0; i < nSize; i++)
        {
                Q_f.row(i) = Q_tmp.row(vRows[i]);
        }
        Q_f.row(nSize) = Q_tmp.row(n - 1);

        return Q_f;
}

// Apply one opt search to the integer solution
// In :
//      - const DnMat &C
//      - const vector<size_t> &domains
//      - DnVec &x
//      - Double f
//      - size_t n
// Out :
//      - Double min : value of the new solution after 1-opt
double oneOptSearch(const DnMat& C, const vector<int>& domains, DnVec& x, double f, size_t n)
{
    assert(n > 0);

    bool changed = true;
    DnVec work = DnVec::Zero(n);
    double min = f;
    double cont = 0;

    while (changed) {

        changed = false;

        for (size_t i = 0; i < domains.size() - 1; i++) {

            int first = domains[i];
            int last = domains[i + 1];

            // flip the entry 1 to -1 in the domain of each variable
            size_t ind = first;
            double current_value = min;

            for (size_t j = first; j < last; j++) {
                if (x(j) == 1) {
                    x(j) = -1;
                    ind = j;
                }
            }
            size_t indMin = ind;

            // remove contribution of x(ind)
            work = C.col(ind);
            work(ind) = 0;

            cont = x.dot(work);
            current_value += 4.0 * cont * x(ind);

            for (size_t j = first; j < last; j++) {
                // add contribution of x(j)
                // remove contribution of x(ind)
                work = C.col(j);
                work(j) = 0;

                cont = x.dot(work);
                current_value -= 4.0 * cont * x(j);

                if (current_value < min - 1e-7) {
                    min = current_value;
                    indMin = j;
                    changed = true;
                }

                current_value += 4.0 * cont * x(j);
            }
            // f = current_min;
            // cout << "\nValues of f: " << f;
            x(indMin) = 1;
        }
    }

    return min;
}

// Do multiple roundings and keep the best solution
// In :
//      - WeightedCSP* wcsp
//      - const DnMat &C
//      - DnMat &V
//      - const vector<int> &domains
//      - size_t nbRound
//      - size_t n
//      - size_t k
// Out :
//      - Double min : value of the new solution after multiple 1-opt
//      - DnVec bestSol : best integer solution found
tuple<double, DnVec> multipleRounding(wcsp& wcsp, const DnMat& C, DnMat& V,
    const vector<int> domains, size_t nbRound, size_t n, size_t k)
{
    assert(nbRound > 0);
    assert(n > 0);
    assert(k > 0);

    DnVec work(n);
    DnVec bestSol = work;
    double min = DBL_MAX;
    double trial_sol = min;
    double b = bias(wcsp);

    for (size_t i = 0; i < nbRound; i++) {
        work = rounding(V, domains, n, k, i);
        trial_sol = evalFun(C, work) + b;
        trial_sol = oneOptSearch(C, domains, work, trial_sol, n);

        if (trial_sol < min) {
            min = trial_sol;
            bestSol = work;
        }
    }

    return make_tuple(min, bestSol);
}

void writeSol(string f_o, const double ub, const DnVec& solution)
{
        string filename(f_o);
        fstream file_out;
        file_out.open(f_o, std::ios_base::out);

        if (!file_out.is_open()) 
        {
                cout << "failed to open " << filename << '\n';
        }
        else 
        {
                for (size_t i = 0; i < solution.size() - 1; i++)
                {
                        if (solution(i) == -1)
                        {
                                file_out << 0 << " ";
                        } else {
                                file_out << 1 << " ";
                        }
                }
                file_out << endl;
                file_out << ub << endl;
        }
}

int main(int argc, char *argv[])
{
        //use current time as seed for random generator
        srand(time(0));

        //usage
        if (argc != 7)
        {
                cout << "usage: " << argv[0] << " <wcsp-input>  "
                     << " 1 - Dualization method, 2 - BCD method "
                     << " -it=-1, i : maximum number of iterations "
                     << " -k=-1, -2, -4, r : chosen value for the method rank "
                     << " -nbR=m : number of output integer solutions "
                     << " -f=sol.txt : return integer solution in txt file\n";
                return 1;
        }

        //read wcsp
        ifstream ifs(argv[1]);

        if (!ifs)
        {
                cout << "could not open " << argv[1] << "\n";
                return 1;
        }

        wcsp w = readwcsp(ifs);

        //set number of iterations
        string sit = argv[3];
        string subit = sit.substr(4, sit.size() - 4);
        int maxiter = stoi(subit);

        //default setting for maxiter
        if(maxiter == -1)
        {
                maxiter = 100000;
        }

        //set tolerance for the stopping criterion
        double tol = 1e-5;

        //set rank
        string srank = argv[4];
        string subrank = srank.substr(3, srank.size() - 3);
        int k = stoi(subrank);

        switch ( k )
        {
        case -1:
                k = w.getRank(stoi(argv[2]));
                break;
        case -2:
                k = w.getRank(stoi(argv[2]))/2;
                break;
        case -4:
                k = w.getRank(stoi(argv[2]))/4;
                break;
        }
      
        //set number of solutions to output
        string ssol = argv[5];
        string subsol = ssol.substr(5, ssol.size() - 5); 
        int nbR = stoi(subsol);

        //set filename for output solutions 
        string file = argv[6];
        string fo = file.substr(3, file.size() - 3);

        vector<vector<vector<int>>> listRdAssignment;

        int method = stoi(argv[2]); // 1 for LR-LAS, 2 for LR-BCD
        DnMat Q;
        DnMat V;

        // Compute cost matrix
        if (method == 1){ // LR-LAS
                Q = w.dualMat();

        } else {        // LR-BCD
                DnMat C = w.SDPmat();
                Q = preProcessing(C, w);
        }
        
        auto begin = cpuTime();
        if (method == 1){ // LR-LAS
                V = doMixing(w, Q, tol, maxiter, k);
        } else {        // LR-BCD
                V = doBCDMixing(w, Q, tol, maxiter, k);
        }
        auto end = cpuTime();

        double lb = evalFun(Q, V) + bias(w);
        cout << std::fixed << "\n" << ceil(lb) << ' ' << (end - begin);

        // compute upper bound with GW heuristic
        vector<int> dom = w.domains();
        DnVec intSol;
        double ub_oneopt = DBL_MAX;
        int n = w.getSDPSize() + 1;

        begin = cpuTime();
        tie(ub_oneopt, intSol) = multipleRounding(w, Q, V, dom, nbR, n, k);
        end = cpuTime();

        cout << std::fixed << ' ' << long(ub_oneopt) << ' ' << (end - begin) << endl;

        writeSol(fo, ub_oneopt, intSol);
                        
        return 0;
}

#endif
