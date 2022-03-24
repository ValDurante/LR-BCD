#ifndef __MIXING_HH
#define __MIXING_HH

#define _USE_MATH_DEFINES

#include <vector>
// #include <intrin.h>
#include <iostream>
#include <memory>
#include <random>
#include <algorithm>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <ctime>
#include <sys/time.h>
#include <sys/resource.h>

#include "wcsp.h"
#include "wcspreader.hh"

using namespace std;

const bool debug = false;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::VectorXd DnVec;
typedef Eigen::MatrixXd DnMat;

const double MEPS = 1e-24;

static inline double cpuTime(void)
{
        struct rusage ru;
        getrusage(RUSAGE_SELF, &ru);
        return (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1000000;
}

// Initializes to 0 the memory block pointed by v
void szero(double *v, int l);

// y = a*x + y
void saxpy(const double *y, double a, const double *x, int l);

// x = a*x
inline void sscal(vector<double> &x, double a)
{
        for (size_t i = 0; i != x.size(); i++)
        {
                x[i] = a * x[i];
        }
}

// dot product x^Ty
inline double sdot(const vector<double> &x, const vector<double> &y)
{
        double acc = 0;
        size_t vSize = x.size();

        for (size_t i = 0; i != vSize; i++)
        {
                acc = acc + x[i] * y[i];
        }

        return acc;
}

// 2-norm of a vector
inline double snrm2(const vector<double> &v)
{
        return sqrt(sdot(v, v));
}

// Random vector from the standard normal distribution N(0,1)
void randUnit(vector<double> &v)
{
        default_random_engine generator;
        normal_distribution<double> distribution(0, 1);

        for (size_t i = 0; i != v.size(); i++)
        {
                v[i] = distribution(generator);
        }

        double normV = snrm2(v);
        sscal(v, 1 / normV);
}

DnMat mixingInit(wcsp &wcsp, int k)
{
        default_random_engine generator;
        generator.seed(time(0));
        normal_distribution<double> distribution(0, 1);

        size_t n = wcsp.getSDPSize() + 1;
        //size_t k = wcsp.getRank();
        DnMat V = DnMat::Zero(k, n);

        for (size_t j = 0; j != n; j++)
        {
                for (int i = 0; i != k; i++)
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

        for (size_t i = 0; i != pfun.size(); i++)
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
                                for (size_t j = 0; j != validValues.size(); j++)
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
                                for (size_t j = 0; j != validValues.size(); j++)
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

DnMat constrainedMatrixUpdate(DnMat &Q, wcsp &wcsp)
{
        const vector<wcspvar *> pVar = wcsp.getVariables();
        vector<int> domains = wcsp.domains();
        vector<int> vRows = wcsp.validRows();
        size_t sizeR = vRows.size();
        int d = wcsp.getSDPSize();
        int nCol = Q.cols() - 1;
        double rho = wcsp.penaltyCoeff();
        double bb = Q(nCol, nCol);

        for (size_t i = 0; i != pVar.size(); i++)
        {

                int d_i = pVar[i]->domainSize();

                if (pVar[i]->isAssigned())
                {
                        int value = pVar[i]->getValue();

                        for (int j = 0; j != d_i; j++)
                        {

                                if (j == value)
                                {
                                        Q.col(nCol) -= Q.col(domains[i] + j);
                                }
                                else
                                {
                                        Q.col(nCol) -= 2 * Q.col(domains[i] + j);
                                }
                        }
                        bb = bb - (2 * rho + 1) * pow((1 - d_i / 2.), 2);
                }
                else
                {
                        vector<int> dom = pVar[i]->getDomain();
                        int d_n = pVar[i]->relatDomainSize();
                        int n = d_i - d_n;

                        for (int j = 0; j != d_i; j++)
                        {
                                if (dom[j] == -1)
                                {
                                        Q.col(nCol) -= 2 * Q.col(domains[i] + j);
                                }
                        }

                        if (d_i != d_n)
                        {
                                for (int j = 0; j != d_i; j++)
                                {
                                        Q(domains[i] + j, nCol) += n * (2 * rho + 1) / 4.;
                                }

                                bb = bb + (2 * rho + 1) * (1. / 4) * (n * n + 4 * n * (2 - d_i));
                        }
                }
        }
        Q.row(nCol) = Q.col(nCol).transpose();

        //create the new matrix
        DnMat Q_tmp(nCol + 1, sizeR + 1);
        for (size_t i = 0; i != sizeR; i++)
        {
                Q_tmp.col(i) = Q.col(vRows[i]);
        }
        Q_tmp.col(d) = Q.col(nCol);

        DnMat Q_final(d + 1, d + 1);
        for (size_t i = 0; i != sizeR; i++)
        {
                Q_final.row(i) = Q_tmp.row(vRows[i]);
        }
        Q_final.row(d) = Q_tmp.row(nCol);
        Q_final(d, d) = bb;

        return Q_final;

        //Eigen::VectorXi indicesToKeepVector = Eigen::VectorXi(vRows.data(), vRows.size());
        //Q = Q(all,indicesToKeepVector);
}

DnMat SDPMatrixUpdate(DnMat &Q, wcsp &wcsp)
{
        const vector<wcspvar *> pVar = wcsp.getVariables();
        vector<int> domains = wcsp.domains();
        vector<int> vRows = wcsp.validRows();
        size_t sizeR = vRows.size();
        int d = wcsp.getSDPSize();
        int nCol = Q.cols() - 1;

        for (size_t i = 0; i != pVar.size(); i++)
        {
                int d_i = pVar[i]->domainSize();

                if (pVar[i]->isAssigned())
                {
                        int value = pVar[i]->getValue();

                        for (int j = 0; j != d_i; j++)
                        {

                                if (j == value)
                                {
                                        Q.col(nCol) -= Q.col(domains[i] + j);
                                }
                                else
                                {
                                        Q.col(nCol) -= 2 * Q.col(domains[i] + j);
                                }
                        }
                }
                else
                {
                        vector<int> dom = pVar[i]->getDomain();

                        for (int j = 0; j != d_i; j++)
                        {
                                if (dom[j] == -1)
                                {
                                        Q.col(nCol) -= 2 * Q.col(domains[i] + j);
                                }
                        }
                }
        }
        Q.row(nCol) = Q.col(nCol).transpose();

        //create the new matrix
        DnMat Q_tmp(nCol + 1, sizeR + 1);
        for (size_t i = 0; i != sizeR; i++)
        {
                Q_tmp.col(i) = Q.col(vRows[i]);
        }
        Q_tmp.col(d) = Q.col(nCol);

        DnMat Q_final(d + 1, d + 1);
        for (size_t i = 0; i != sizeR; i++)
        {
                Q_final.row(i) = Q_tmp.row(vRows[i]);
        }
        Q_final.row(d) = Q_tmp.row(nCol);
        Q_final(d, d) = 0;

        return Q_final;
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

vector<vector<int>> rounding(const DnMat &V, wcsp &wcsp, int k, int i)
{
        //int k = wcsp.getRank();
        //vector<int> domains = wcsp.relatDomains();
        vector<int> domains = wcsp.domains();
        const vector<wcspvar *> &pVar = wcsp.getVariables();

        default_random_engine generator;
        generator.seed(i + 1);
        normal_distribution<double> distribution(0, 1);

        DnVec r(k);
        vector<vector<int>> rdAssignment = assignmentInit(wcsp);

        for (int i = 0; i != k; i++)
        {
                r(i) = distribution(generator);
        }

        r = r / r.norm();

        int var = 0; //if a var is already assigned we need to shift the remaining indices in domains
        for (size_t i = 0; i != pVar.size(); i++)
        {
                if (pVar[i]->isAssigned())
                {
                        rdAssignment[i] = pVar[i]->getDomain();
                }

                else
                {
                        int l = domains[var];
                        int ind = 0;

                        double max = r.dot(V.col(l));
                        double n_max = max;

                        for (int j = domains[var] + 1; j != domains[var + 1]; j++)
                        {
                                max = r.dot(V.col(j));
                                if (max > n_max)
                                {
                                        ind++;
                                        n_max = max;
                                }
                        }

                        rdAssignment[i][ind] = 1;
                        var++;
                }
        }

        return rdAssignment;
}

template <typename T>
double evalFun(T const &C, const DnMat &V)
{
        DnMat A = C * V.transpose();
        A = V * A;
        double eval = A.trace();

        return eval;
}

// Check the gangster operator constraint
template <typename T>
double evalConst(T const &M, const DnMat &V)
{
        DnMat A = M * V.transpose();
        A = V * A;
        double eval = A.trace();

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

        for (int i = first; i != last; i++)
        {
                g = V * C.col(i);
                G.col(i - first) = g;
        }

        return G;
}

/*
Compute the angles between the objective vectors
and v_{d+1}
in :
- const DnMat& G
- const DnVec& u
out
- vector<double> angles  
*/
vector<double> angles(const DnMat &G, const DnVec &u, int d)
{
        vector<double> angles(d);

        for (int i = 0; i != d; i++)
        {
                DnVec g = G.col(i);
                double a = acos(g.dot(u) / g.norm());
                angles[i] = a;
        }

        return angles;
}

DnVec relaxedSol(const DnMat &G, const DnVec &u, int d, int k)
{

        int n = k * d;
        DnVec g = DnVec::Zero(n);
        DnVec w = DnVec::Zero(n);

        for (int i = 0; i != n; i++)
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

vector<double> solAngle(const DnMat &G, const vector<double> &angles, double dualOpti, int d)
{
        vector<double> solAngle(d);

        for (int i = 0; i != d; i++)
        {
                DnVec g = G.col(i);
                double nrm_g = g.norm();
                double theta = angles[i];

                solAngle[i] = fmod(M_PI - atan(nrm_g * sin(theta) / (nrm_g * cos(theta) + dualOpti)), M_PI);
        }

        return solAngle;
}

DnMat vecSol(const DnMat &G, const vector<double> &solAngle,
             const vector<double> angles, const DnVec &u, int d, int start, int k)
{

        DnMat res(k, d);
        for (int i = 0; i != d; i++)
        {
                DnVec g = G.col(i);
                double nrm_g = g.norm();
                double vdot = u.dot(g);
                double x = solAngle[i];
                double theta = angles[i];
                double c_x = cos(x);
                double c_tx = cos(x + theta);
                double denom = g.squaredNorm() - pow(vdot, 2);

                double b = (nrm_g * c_tx - c_x * vdot) / denom;
                double a = c_x - vdot * b;

                res.col(i) = a * u + b * g;
        }

        return res;
}

double currentDelta(const DnMat &G, const DnMat &V, const DnMat &Vb, int first,
                    int last)
{
        double currDelta = 0;
        for (int i = first; i != last; i++)
        {
                currDelta += 2 * (V.col(i) - Vb.col(i - first)).dot(G.col(i - first));
        }

        return currDelta;
}

void solUpdate(DnMat &V, const DnMat &Vb, int first, int last)
{
        for (int i = first; i != last; i++)
        {
                V.col(i) = Vb.col(i - first);
        }
}

double jac_h(const DnMat &G, const DnVec &u, double x, int d)
{
        double res = 0;

        for (int i = 0; i != d; i++)
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

        for (int i = 0; i != d; i++)
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
        int it = 0;
        double eps = pow(10, -7);

        while (abs(jac) > eps && it < maxiter)
        {
                x_curr = x_curr - 0.90 * jac_h(G, u, x_curr, d) / hess_h(G, u, x_curr, d);
                jac = jac_h(G, u, x_curr, d);
                it++;
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
        vector<double> a = angles(G, u, d);

        DnVec solRelax = relaxedSol(G, u, d, k);
        double dual = dualInit(G, u, solRelax, k);

        //Dual opti
        double dualOpti = newton(G, u, dual, d, maxiter, N_it);

        //Build solution vectors
        vector<double> angleOpti = solAngle(G, a, dualOpti, d);
        DnMat Vb = vecSol(G, angleOpti, a, u, d, first, k);

        delta += currentDelta(G, V, Vb, first, last);
        solUpdate(V, Vb, first, last);
}

DnMat doBCDMixing(wcsp &wcsp, const DnMat &C, double tol, int maxiter, int k)
{
        vector<int> domains = wcsp.relatDomains();
        //int k = wcsp.getRank();
        int d = wcsp.getSDPSize();
        double delta = 0;
        int BCD_it = 0;
        int N_it = 0;
        DnVec q(d + 1);
        DnVec g(k);

        if (debug)
        {
                cout << "\nLe rang k est : " << k;
        }

        //V init
        DnMat V = mixingInit(wcsp, k);

        if (debug)
        {
                cout << "\n"
                     << delta;
        } //test

        for (int i = 0; i != maxiter; i++)
        {
                DnVec u = V.col(d);
                delta = 0;

                for (size_t j = 0; j != domains.size() - 1; j++)
                {
                        solveBCD(C, V, u, domains, delta, k, j, N_it);
                }

                BCD_it++;

                if (delta < tol)       //* 1e9)
                {
                        break;
                }

                if (debug)
                {
                        cout << "\n"
                             << delta;
                }
        }

        if (debug)
        {
                cout << "\nNb of iterations : " << BCD_it;
                cout << N_it << ' ' << BCD_it << ' ';
        }
        return V;
}

DnMat doMixing(wcsp &wcsp, const DnMat &Q, double tol, int maxiter, int k)
{
        //int k = wcsp.getRank();
        int d = wcsp.getSDPSize();
        int it = 0;
        DnMat V = mixingInit(wcsp, k);
        DnVec g(k);
        DnVec v(k);
        DnVec q(d + 1);

        //cout << "\nLe rang k est : " << k;

        for (int i = 0; i != maxiter; i++)
        {
                double delta = 0;
                for (int j = 0; j != d + 1; j++)
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

                if (delta < tol)           //* 1e9)
                {
                        break;
                }

                if (debug)
                {
                        double iter_n;
                        iter_n = evalFun(Q, V);
                        cout << "\n"
                             << iter_n;
                }
        }

        if (debug)
        {
                cout << "\nNb of iterations : " << it;
                cout << it << ' ';
        }
        return V;
}

DnMat doMixing_2(wcsp &wcsp, const DnMat &Q, double tol, int maxiter, int k)
{
        //int k = wcsp.getRank();
        int d = wcsp.getSDPSize();
        DnMat V = mixingInit(wcsp, k);
        DnVec g = DnVec::Zero(k);
        DnVec v(k);
        DnVec q(d + 1);

        for (int i = 0; i != maxiter; i++)
        {
                double delta = 0;

                for (int j = 0; j != d + 1; j++)
                {
                        q = Q.row(j);
                        q(j) = 0;

                        for (int l = 0; l != q.size(); l++)
                        {
                                if (q(l) != 0)
                                {
                                        g += q(l) * V.col(l);
                                }
                        }

                        double gnrm = g.norm();

                        // Ensure non-degeneration of the g-vector
                        if (gnrm < MEPS)
                        {
                                continue;
                        }

                        v = -g / gnrm;
                        delta += gnrm * (1 - V.col(j).dot(v)) * 2;
                        V.col(j) = v;
                        g.setZero();
                }

                if (delta < tol)
                {
                        break;
                }

                //double iter_n;
                //iter_n = evalFun(Q,V);
                //cout << "\nObjective fun value is : " << iter_n;
        }

        return V;
}

DnVec buildDual(const DnMat &V, const DnMat &Q, double tol)
{
        int n = Q.rows();
        DnVec diag = Q.diagonal();
        DnMat Mdiag = diag.asDiagonal();
        DnMat Qd = Q - Mdiag;

        double inftyNorm = Qd.cwiseAbs().rowwise().sum().maxCoeff() / 100;
        if (debug)
        {
                cout << "\nLa constante de Lip est : " << inftyNorm;
        }
        DnVec d(n);
        DnVec q(n);

        for (int i = 0; i != n; i++)
        {
                q = Qd.row(i);
                d(i) = (V * q).norm() + tol * inftyNorm / n;
        }

        return d;
}

double evalDual(const DnVec &d, const DnMat &Q)
{
        //Dual problem feasibility
        DnMat dg = d.asDiagonal();
        DnVec diag = Q.diagonal();
        DnMat Mdiag = diag.asDiagonal();
        DnMat Qd = Q - Mdiag;
        DnMat Q_dual = Qd + dg;

        //Check Q + diag(d) >= 0
        Eigen::EigenSolver<DnMat> es(Q_dual, false);
        DnVec eigval = es.eigenvalues().real();
        double eigmin = eigval.minCoeff();

        if (eigmin < 0)
        {
                cout << "\nThe dual point is not feasible";
        }

        double dval = -1 * d.sum() + Q.trace();

        return dval;
}

DnMat preProcessing(DnMat &Q, wcsp &w)
{
        const vector<wcspvar *> &pVar = w.getVariables();
        vector<int> domains = w.domains();
        int n = Q.cols();

        //setValue makes _assigned = true
        bool tag = true;

        for (size_t i = 0; i != domains.size() - 1; i++)
        {
                int start = domains[i];
                int end = domains[i + 1];

                for (int j = start; j != end; j++)
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

        for (int i = 0; i != nSize; i++)
        {
                Q_tmp.col(i) = Q.col(vRows[i]);
        }
        Q_tmp.col(nSize) = Q.col(n - 1);

        for (int i = 0; i != nSize; i++)
        {
                Q_f.row(i) = Q_tmp.row(vRows[i]);
        }
        Q_f.row(nSize) = Q_tmp.row(n - 1);

        return Q_f;
}

double objectiveFunction(wcsp &w)
{
        const vector<wcspfun *> &pfun = w.getFunctions();
        double objectiveValue = 0;

        for (size_t i = 0; i < pfun.size(); i++)
        {
                double value = pfun[i]->getAssignment();
                objectiveValue = objectiveValue + value;
        }

        return objectiveValue;
}

double deltaObjVar(wcsp &w, size_t var_index)
{
        wcspvar *varp = w.getVariables()[var_index];
        const vector<wcspfun *> &pfun = varp->functions;
        double objectiveValue = 0;

        for (size_t i = 0; i < pfun.size(); i++)
        {
                objectiveValue += pfun[i]->getAssignment();
        }

        return objectiveValue;
}

double oneOptSearch(vector<vector<int>> &rdAssignment, wcsp &w)
{
        //setValue makes _assigned = false
        bool tag = false;

        w.assignmentUpdate(rdAssignment, tag);
        const vector<wcspvar *> &pVar = w.getVariables();
        double initial = objectiveFunction(w);
        size_t size = rdAssignment.size();
        double min = initial;
        double curr_val;
        bool changed = true;

        while (changed)
        {
                changed = false;
                for (size_t i = 0; i != size; i++)
                {
                        size_t size_i = rdAssignment[i].size();
                        size_t value = pVar[i]->getValue();
                        double objBase = initial - deltaObjVar(w, i);

                        rdAssignment[i][value] = 0;
                        int indMin = value;

                        for (size_t j = 0; j != size_i; j++)
                        {
                                if (j != value)
                                {
                                        rdAssignment[i][j] = 1;
                                        w.assignmentUpdate(rdAssignment, tag);
                                        curr_val = objBase + deltaObjVar(w, i);

                                        if (curr_val < min)
                                        {
                                                min = curr_val;
                                                indMin = j;
                                                changed = true;
                                        }
                                        rdAssignment[i][j] = 0;
                                }
                        }
                        initial = min;
                        rdAssignment[i][indMin] = 1;
                        w.assignmentUpdate(rdAssignment, tag);
                }
        }
        return min + w.getLowerBound();
}

tuple<vector<double>,vector<vector<vector<int>>>> multipleRounding(const DnMat &V, wcsp &wcsp, int nbRound, int k)
{
        vector<double> pSol;
        vector<vector<int>> rdAssignment_opti;
        vector<vector<vector<int>>> aSol;
        double sol;

        for (int i = 0; i != nbRound; i++)
        {
                rdAssignment_opti = rounding(V, wcsp, k, i);
                sol = oneOptSearch(rdAssignment_opti, wcsp);
                pSol.push_back(sol);
                aSol.push_back(rdAssignment_opti);
        }

        return make_tuple(pSol, aSol);
}

vector<DnMat> positivityConst(wcsp &w)
{
        int d = w.getSDPSize();
        vector<DnMat> constMat;
        const vector<wcspvar *> &pVar = w.getVariables();
        int acc = pVar[0]->domainSize();

        for(size_t i = 0; i != pVar.size() - 1; i++)
        {
                for(int j = 0; j != acc; j++)
                {
                        for(size_t l = acc; l != acc + pVar[i+1]->domainSize(); l++)
                        {
                                DnMat Q = DnMat::Zero(d + 1, d + 1);
                                Q(l,j) = 0.5;
                                Q(j,l) = 0.5;
                                Q(l,d) = 0.5;
                                Q(j,d) = 0.5;
                                Q(d,l) = 0.5;
                                Q(d,j) = 0.5;
                                constMat.push_back(Q);

                        }
                }
                acc = acc + pVar[i+1]->domainSize();
        }

        return constMat;
}

vector<DnMat> gangsterConst(wcsp &w)
{
        int d = w.getSDPSize();
        vector<DnMat> constMat;
        const vector<wcspvar *> &pVar = w.getVariables();
        int acc = 0;

        for(size_t i = 0; i != pVar.size(); i++)
        {
                for(size_t j = acc; j != acc +  pVar[i]->domainSize(); j++)
                {
                        for(size_t l = acc; l != j; l++)
                        {
                                DnMat Q = DnMat::Zero(d + 1, d + 1);
                                Q(l,j) = 0.5;
                                Q(j,l) = 0.5;
                                Q(l,d) = 0.5;
                                Q(j,d) = 0.5;
                                Q(d,l) = 0.5;
                                Q(d,j) = 0.5;
                                constMat.push_back(Q);

                        }
                }
                acc = acc + pVar[i]->domainSize();
        }

        return constMat;
}

void writeSol(string f_o, vector<vector<vector<int>>>& aSol, vector<double>& pSol)
{
        vector<vector<int>> currentSol;
        string filename(f_o);
        fstream file_out;
        file_out.open(f_o, std::ios_base::out);

        if (!file_out.is_open()) 
        {
                cout << "failed to open " << filename << '\n';
        }
        else 
        {
                for (size_t i = 0; i != aSol.size(); i++)
                {
                        currentSol = aSol[i];

                        for (size_t j = 0; j != currentSol.size(); j++)
                        {
                                for (size_t k = 0; k != currentSol[j].size(); k++)
                                {
                                        file_out << currentSol[j][k] << " ";
                                }
                        }
                        file_out << endl;
                }
                for (size_t i = 0; i != pSol.size(); i++)
                {
                        file_out << pSol[i] << " ";
                }
                file_out << endl;
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
        double tol = 10e-2; //* 1e9;

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

        if (stoi(argv[2]) == 1)
        {

                if (debug)
                {
                        cout << "\nDualization method";
                }
                if (debug)
                {
                        cout << "\nThe penalty coefficient is : " << w.penaltyCoeff();
                }

                //compute the Rho lower bound
                DnMat Q = w.dualMat();

                if (debug)
                {
                        double rho = w.penaltyCoeff();
                        cout << "\nPenalty coefficient" << rho;
                }

                auto begin = cpuTime();
                DnMat V = doMixing(w, Q, tol, maxiter, k);
                auto end = cpuTime();

                double lb = evalFun(Q, V) + bias(w);
                cout << std::fixed << ceil(lb) << ' ' << (end - begin);

                if (debug)
                {
                        DnVec d = buildDual(V, Q, tol);
                        double dValue = evalDual(d, Q) + bias(w);

                        cout << "\nThe dual lower bound is : " << dValue;
                }

                // Constraints checking
                if (debug)
                {
                        DnMat M = w.gOperator();
                        double gVal = evalConst(M, V);
                        cout << "\nGangster constraint value : " << gVal << "\n";
                }

                begin = cpuTime();
                auto [pSol, aSol] = multipleRounding(V, w, nbR, k);
                end = cpuTime();

                double min = *min_element(pSol.begin(), pSol.end());
                cout << std::fixed << ' ' << long(min) << ' ' << (end - begin);

                writeSol(fo, aSol, pSol);
        }

        else
        {
                if (stoi(argv[2]) == 2)
                {
                        if (debug)
                        {
                                cout << "\nBCD method";
                        }

                        //Compute the BCD lower bound
                        DnMat C = w.SDPmat();

                        if (debug)
                        {
                                cout << "\nC rows" << C.rows();
                                cout << "\nC cols" << C.cols();
                        }

                        DnMat C_f = preProcessing(C, w);

                        if (debug)
                        {
                                cout << "\nC_f rows" << C_f.rows();
                                cout << "\nC_f cols" << C_f.cols();
                        }

                        auto begin = cpuTime();
                        DnMat V = doBCDMixing(w, C_f, tol, maxiter, k);
                        auto end = cpuTime();
                         
                        if (debug)
                        {
                                DnMat M = w.gOperator();
                                double gVal = evalConst(M, V);
                                cout << "\nGangster constraint value : " << gVal << "\n";
                        }

                        double lb = evalFun(C_f, V) + bias(w);
                        cout << ceil(lb) << ' ' << (end - begin);

                        begin = cpuTime();
                        auto [pSol, aSol] = multipleRounding(V, w, nbR, k);
                        end = cpuTime();

                        double min = *min_element(pSol.begin(), pSol.end());
                        cout << std::fixed << ' ' << long(min) << ' ' << (end - begin);

                        writeSol(fo, aSol, pSol);
                }
                else
                {
                        cout << "\nTry again !";
                }
        }

        return 0;
}

#endif
