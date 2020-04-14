#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def pigDie(niter, n):

    import pandas as pd
    import numpy as np
    import random


#We initialize values of our matrix here
    iterations = niter
    holdn = n

    probofn = np.array([0,0,1,2,3,4,5])
    probofn [1:len(probofn)] = probofn [1:len(probofn)] + holdn
    rownames = ["nzero","n","n+1","n+2", "n+3", "n+4", "n+5"]
    colnames = ["Probability"]
    probmatrix = np.array([0,0,0,0,0,0,0], dtype=np.float64)
    dievalues = [1,2,3,4,5,6]


# (b) Here we run the number of iterations desired and store the values in a table, later broadcasting a division by the number of iterations
    for i in range(0, iterations-1):
        turnvalue = 0
        
# (a) The while statement runs a single turn
        while turnvalue < holdn:
        
            die = random.sample(dievalues,1)
            die = die[0]
            if die == 1:
                turnvalue = 0
                break
            else:
                turnvalue += np.float64(die)

        probmatrix[np.where(probofn == turnvalue)] += np.float64(1)
   
    probmatrix /= np.float64(iterations)
    probmatrix = pd.DataFrame(probmatrix, columns = colnames, index = rownames)

    print("This is the table for n = ", holdn," and ", iterations, "iterations" )
    print(probmatrix)
    return(probmatrix)


# In[ ]:


import pandas as pd
import numpy as np

# we declare the names of our table of hold at "n" scenarios and the probability of results
nrows = ["hold@15","hold@16","hold@17","hold@18","hold@19","hold@20","hold@21","hold@22","hold@23","hold@24","hold@25"]
ncols = ["nzero","n","n+1","n+2", "n+3", "n+4", "n+5"]
colname = ["expectedValue"]

# this "database" table will hold all the probabilities at different hold@n scenarios
database = pd.DataFrame( columns = ncols, index = nrows)

#(e) this loop  populates the "database" table
for i in range(0,len(nrows)):
    holdn = 15 + i
    aux = pigDie(1000000,holdn)
    database.iloc[i,:] = list(aux.iloc[:,0])

# this database will be populated  with the expected values of the different hold@n scenarios   
exValues = pd.DataFrame( columns = colname, index = nrows )
tarray = [0,1,2,3,4,5]
narray = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

#this "for" loop broadcasts EACH member of 'narray' into 'tarray' to generate
#the n's -([15,16,17,18,19,20]), then ([16,17,18,19,20,21]) and so on -.
#then each of these vectors is multiplied by their corresponding simulated
#probability in "database", then summed and stored in "exValues"
for m in range(0, len(narray)):
    
    exValues.iloc[m] = np.sum(np.multiply(np.append(0,np.add(narray[m],tarray)), database.iloc[m,:]))
print("This is the expected value tables for each hold@n enunciated")
print(exValues)


# In[ ]:


print("This is the expected value tables for each hold@n enunciated")
print(exValues)


# In[ ]:




