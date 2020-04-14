/* answer to (a)*/

bigorder(X) :- order(X,Y,Z), Z > 100.

/* answer to (b) */

notEnough(X,W) :- order(X,Y,Z), inventory(Y,A), part(W,Y,B), Z > A.

/*answer to (c)*/

orderedmore(C1,C2) :- order(C1,Y,Z1), order(C2,Y,Z2), Z1 > Z2. 



