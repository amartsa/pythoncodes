library(HSAUR)#Opens the HSAUR package
library(MASS)#Opens the MASS package
library(car)
myhept <- heptathlon #loads the heptathlon dataframe into "myhept"
myhept[,8]=NULL #deletes the score column
mymahala<- mahalanobis(myhept,colMeans(myhept),var(myhept)) #inputs the mahalanobis dataframe into "mymahala"
qqPlot(mymahala,dist="chisq",df=7) 
