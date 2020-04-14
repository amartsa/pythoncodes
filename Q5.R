mynewVars <-heptathlon
mynewVars[,8]=NULL#Deletes the score column
mynewVars[,1]<-max(mynewVars[,1])-mynewVars[,1]#Substracts largest value of referenced columns to each value of such column
mynewVars[,2]<-max(mynewVars[,2])-mynewVars[,2]#Ibid
mynewVars[,3]<-max(mynewVars[,3])-mynewVars[,3]#Ibid
mynewVars[,4]<-max(mynewVars[,4])-mynewVars[,4]#Ibid
mynewVars[,5]<-max(mynewVars[,5])-mynewVars[,5]#Ibid
mynewVars[,6]<-max(mynewVars[,6])-mynewVars[,6]#Ibid
mynewVars[,7]<-max(mynewVars[,7])-mynewVars[,7]#Ibid
stars(mynewVars,cex=.55)#creates the star plot