# QAOA hybrid algorithm to solve the Travelling Salesman's problem

The Travelling Salesman's problem is a NP-hard problem, commonly solved using bruteforce. The problem is that a Salesman travels from city to city selling products. How can the salesman/slaesperson travel from city to city in the most efficient way, to maximise the amount of time the salesperson can sell products for?

This was developed by W.R Hamilton in the 19th century.
(**More will be added to this Readme soon**)

When the algorithm is tested, it runs significantly faster than the  Christofides algorithm, which indicates that quantum computing can very well exceed/ beat classical computers at the effiency of this problem.

# Drawbacks 
In this algorithm, a Hamiltonian Cycle is described by N^2 variables, meaning if there were 100 bubbles/cities, the cycle is represented by 100^2 variables, something that can be painfully long when using big numbers. I will continue to make edits to the algorithm, to increase the overall speed of the algorithm.
