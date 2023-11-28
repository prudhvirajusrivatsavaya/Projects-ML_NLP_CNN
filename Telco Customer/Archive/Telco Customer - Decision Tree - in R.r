
install.packages("ggplot2")
install.packages("dplyr")
install.packages("rpart")
# install.packages("rattle")
install.packages("rpart.plot")
# install.packages("RColorBrewer")
# install.packages("party")
# install.packages("partykit")
# install.packages("caret")

library(ggplot2)
library(dplyr)
library(rpart)
# library(rattle)
library(rpart.plot)
# library(RColorBrewer)
# library(party)
# library(partykit)
# library(caret)

data.set = read.csv(file="Datasets/Teleco_Cust_Attr.csv",header = T)
str(data.set)
summary(data.set)
prop.table(table(data.set$Churn))

data.set$customerID = NULL
sapply(data.set,function(x){sum(is.na(x))})

data.set$TotalCharges = ifelse(is.na(data.set$TotalCharges),data.set$MonthlyCharges*data.set$tenure,data.set$TotalCharges)

unique(data.set$SeniorCitizen)
data.set$SeniorCitizen = as.factor(data.set$SeniorCitizen)

str(data.set)

set.seed(987)
sample = sample(1:nrow(data.set),size=0.7*nrow(data.set))
train.data.set = data.set[sample,]
test.data.set = data.set[-sample,]
nrow(data.set)
nrow(train.data.set)
nrow(test.data.set)

dec.tree.1 = rpart( Churn ~ . , data=test.data.set,  method = "class")
summary(dec.tree.1)

rpart.plot(dec.tree.1)
?rpart