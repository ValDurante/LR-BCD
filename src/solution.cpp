#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// Solution class to store the integer solution in a priority queue
class Solution

{
private:
	vector<vector<int>> _rdAssignment; // integer solution
	double _value; 					   // corresponding objective valued 

public:
	Solution(vector<vector<int>> rdAssignment, double value) {
		_rdAssignment = rdAssignment;
		_value = value;
	}

	Solution() {}; // default constructor

	~Solution() {}; // destructor

	bool operator<(const Solution& sl) const {
		return (_value < sl.getValue());
	}	

	vector<vector<int>> getAssignment() const {
		return _rdAssignment;
	}

	double getValue() const {
		return _value;
	}
};