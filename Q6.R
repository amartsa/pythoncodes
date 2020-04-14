my4KBC <- skulls[1:30,] #inputs the c4000BC data into my4KBC
my4KBC[,1]=NULL #deletes the C400BC value from the table of values
plot(my4KBC) #Generate plots for all pairs of variables
m4Kdist<-mahalanobis(my4KBC,colMeans(my4KBC),var(my4KBC)) #stores mahalanobis distances in M4Kdist
qqPlot(m4Kdist, dist="chisq", df=4)
cor(my4KBC)