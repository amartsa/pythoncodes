myVar<-heptathlon
myVar[,4]=NULL #Repeated five times to delete the columns we won't be using
myFake<-mvrnorm(50,c(mean(myVar$hurdles),mean(myVar$highjump),mean(myVar$shot)),var(myVar)) #saves the simulation in the "myFake" variable
mFkdist<-mahalanobis(myFake,colMeans(myFake),var(myFake))
qqPlot(mFkdist, dist="chisq", df=7)
