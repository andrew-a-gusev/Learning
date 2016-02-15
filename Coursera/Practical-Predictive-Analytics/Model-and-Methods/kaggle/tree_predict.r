dataset = read.csv(file="testing_features.csv", head=TRUE, sep=",")
summary(dataset)

library(caret)

library(rpart)

# Test the tree
load('decision.tree')
predictedResult <- predict(m, dataset, type = "class")
print (as.vector(predictedResult))