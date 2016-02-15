dataset = read.csv(file="dump.csv", head=TRUE, sep=",")
summary(dataset)

library(caret)
split_index <- createDataPartition (dataset$digit, p = .5, list = FALSE, times = 1)
trainingDataset <- dataset[split_index, ]
testingDataset <- dataset[-split_index, ]

library(rpart)
#fol <- formula(digit ~ c0sd + c0sd2 + c1sd + c1sd2 + c2sd + c2sd2)
fol <- formula(digit ~ .)
training <- as.data.frame (trainingDataset)
m <- rpart(formula=fol, method="class", data=training)
print(m)

# Test the tree
predictedResult <- predict(m, testingDataset, type = "class")
actualResult <- as.vector(testingDataset$digit)
qty <- predictedResult == actualResult
print(paste("Length:",length(actualResult),length(predictedResult),sep=" "))
print(paste("Qty:",sum(qty),sep=" "))
print(paste("Tree Q:",sum(qty)/length(actualResult),sep=" "))
confusionMatrix(data = predictedResult, reference = actualResult)

# randomForrest
library(randomForest)
rfm <- randomForest(fol, data=training)

# Test the forrest
rfPredicted <- predict(rfm, testingDataset, type = "class")
qty <- rfPredicted == actualResult
print(paste("Length:",length(actualResult),length(rfPredicted),sep=" "))
print(paste("Qty:",sum(qty),sep=" "))
print(paste("Forrest Q:",sum(qty)/length(actualResult),sep=" "))
#confusionMatrix(data = rfPredicted, reference = actualResult)

# Check importance
importance(rfm)

# Support Vector Machine (SVM)
library(e1071)
svm <- svm(fol, training)

# Test SVM
svmPredicted <- predict(svm, testingDataset)
qty <- svmPredicted == actualResult
print(paste("Length:",length(actualResult),length(svmPredicted),sep=" "))
print(paste("Qty:",sum(qty),sep=" "))
print(paste("SVM Q:",sum(qty)/length(actualResult),sep=" "))
#confusionMatrix(data = svmPredicted, reference = actualResult)