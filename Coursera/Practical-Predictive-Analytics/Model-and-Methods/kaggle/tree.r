dataset = read.csv(file="dump42k.csv", head=TRUE, sep=",")
summary(dataset)

library(caret)
split_index <- createDataPartition (dataset$digit, p = .5, list = FALSE, times = 1)
trainingDataset <- dataset[split_index, ]
testingDataset <- dataset[-split_index, ]

library(rpart)
#fol <- formula(digit ~ c0sd + c0sd2 + c1sd + c1sd2 + c2sd + c2sd2)
fol <- formula(digit ~ .)
training <- as.data.frame (trainingDataset)
# Train the tree
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
save(m, file='decision.tree')