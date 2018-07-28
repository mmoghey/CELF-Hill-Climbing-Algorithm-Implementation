# CELF-Hill-Climbing-Algorithm-Implementation

the influence maximization problem and the greedy hill-climbing approach
to solving it. Even though the algorithm can solve the problem within a factor of 1 − #
$ of the
optimal solution, it can still be quite slow. Leskovec et al. [1] (in Google drive’s lecture folder)
introduce the CELF (Cost-Effective Lazy Forward-selection) algorithm, which is an optimization
of the hill-climbing algorithm to solve the problem of outbreak detection as we learned in class.
The performance improvement also works for the influence maximization problem. In this
project, you will implement the CELF algorithm using Python, and test it on two datasets.
CELF is based on a “lazy-forward” optimization in selecting the seeds. The idea is that the
marginal gain of a node in the current iteration cannot be better than its marginal gain in the
previous iterations. In other words, CELF exploits the following fact from submodularity: given
two sets A ⊆ B which are subsets of the nodes V, the marginal gain of adding a node s into B is
no greater than that of adding it into A. Hence, for two nodes u, v ∈ V and sets A ⊆ B ⊆ V,
suppose that the marginal gain of u for A is greater than that of v for A. When considering B, if
the marginal gain of u for B is greater than that of v for A, we need not compute the marginal
gains of v for B. This avoids unnecessary computation. The above idea leads to a dramatic speedup
in runtime.
What you
