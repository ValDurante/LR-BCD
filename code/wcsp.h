//wcsp.h
#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Sparse>
#include "wcspfun.h"
#include "wcspvar.h"

#ifndef WCSP_H
#define WCSP_H

using namespace std;

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::SparseVector<double> SpVec;
typedef Eigen::VectorXd DnVec;
typedef Eigen::MatrixXd DnMat;

class wcsp

{
private:
    double _ub;
    double _lb;
    vector<wcspvar *> _variables;
    vector<wcspfun *> _functions;

public:
    wcsp(double, double, const vector<wcspvar *> &, const vector<wcspfun *> &);
    wcsp();
    ~wcsp(void);
    double getUpperBound();
    double getLowerBound();
    void setUpperBound(double lb);
    const vector<wcspvar *> &getVariables();
    const vector<wcspfun *> &getFunctions();
    size_t getSDPSize();
    size_t getRank(int m);
    void assignmentUpdate(const vector<vector<int>> &assignment);
    void resetWcsp();
    vector<int> domains();
    vector<int> relatDomains();
    DnVec unaryCostVector();
    DnMat binaryCostMatrix();
    DnMat SDPmat();
    SpMat constMat();
    DnVec rhs();
    DnMat gOperator();
    DnMat dualMat();
    DnMat exactlyOne();
    double penaltyCoeff();
    vector<int> validRows();
    void writeSol();
};

#endif
