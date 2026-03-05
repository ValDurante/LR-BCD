# LR-BCD

Low Rank SDP Solvers for Pairwise Graphical Models.

# Installation from sources

Compilation requires git and cmake.

Required library: Eigen3

Debian derived: sudo apt install libeigen3-dev
MacOS: brew install eigen

# lrbcd compilation

```bash
mkdir build 
cd build 
cmake .. 
cmake --build .
``` 

# Usage 

./lrbcd INPUT_INSTANCE OPTIONS

# OPTIONS

	solver: "1" for LR-LAS
	"2" for LR-BCD

	iterations -it: "-1" for running with default number of iterations
	"i" for running with specified number of iterations it = i

	rank -k: "-1" for running with default rank 
	"-2" for running with default rank/2
	"-4" for running with default rank/4
	"r" for running with specified rank k = r

	rounding -nbR: "n" for computing the best integer solution value
	with nbR = n rounding schemes

	output file -f: return a file with the best rounded integer solutions and its objective value

example : ./lrbcd <instance.wcsp> 2 -it=-1 -k=-1 -nbR=10 -f=sol.txt

OUTPUT :
	[Lower bound value] [Cpu time SDP resolution] [Best upper bound value after rounding schemes] [Cpu time rounding schemes]

# Benchmarks

Random instances used in the paper can be found at: https://forgemia.inra.fr/thomas.schiex/cost-function-library/-/tree/master/random/pairwise-MRFs

Real instances used in the paper can be found at: https://forgemia.inra.fr/thomas.schiex/cost-function-library/-/tree/master/real/fish

# Code version

To run the ICML paper experiments, use the version with tag 'ICML2022' from the repository.
