## Loadup Caret & Libraries
install.packages("caret")
library(caret)
library(ggplot2)

## Load in the training dataset
training <- read.csv('pml-training.csv', na.strings=c("NA","#DIV/0!",""))

## Load in the testing dataset
testing <- read.csv('pml-testing.csv', na.strings=c("NA","#DIV/0!",""))

## Cleanup the data
## Remove columns with more than 20% NAs
totcol <- ncol(training)
totrow <-nrow(training)
remove_na <- c(0)
for(i in 1:totcol) {
  testna <- training[ ,i]
  if (length(testna)*0.80 <= length(testna[is.na(testna)])) {
    remove_na[length(remove_na)+1] <- i
  }
}
training <- training[ ,-remove_na]

## Remove NAs from testing 
testing <- testing[ ,-remove_na]

## Remove first 7 variables from datasets as they have no bearing on the physical technique
training <- training[,-c(1:7)]
testing <- testing[,-c(1:7)]

## Check that training and test set columns are identical for all predictors
colnames(training)
colnames(testing)

## Check for Near Zero Values ("NZV")
nzv <- nearZeroVar(training, saveMetrics=TRUE)
print(nzv)

## Slice up the training set into training 60%  and cross-validation testing 40%
inTrain <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
xtraining <- training[inTrain,]
vtest <- training[-inTrain,]

## Training Random Forest - No Additional Preprocessing 
modfit <- train(classe ~., method="rf", data=xtraining)

## Training Random Forest - Preprocessing: Normalise skewed variables & remaining NAs
modfit2 <- train(classe ~., method="rf", preProcess=c("center", "scale", "knnImpute"), data=xtraining)

## Training Random Forest - Using K-Fold Cross Validation
modfit3 <- train(classe ~., method="rf", trControl=trainControl(method = "cv", number = 3), data=xtraining)

## Predict all 3 models on validation & show accuracy & show models
pred1 <- predict(modfit, newdata=vtest)
confusionMatrix(pred1, vtest$classe)$overall[1]
## pred1 - 0.992% accuracy which is mid and was used to quiz

pred2 <- predict(modfit2, newdata=vtest)
confusionMatrix(pred2, vtest$classe)$overall[1] 
## pred2 - 0.989% accuracy which is lower so no advantage using these preprocessing techniques

pred3 <- predict(modfit3, newdata=vtest)
confusionMatrix(pred3, vtest$classe)$overall[1]  
## pred3 - 0.993% accuracy which is best but pred1 used for quiz

## Calculate & Plot the 20 most important variables for Model 1 - as it is the most accurate
varimpobj = varImp(modfit)
plot(varimpobj, main = "Top 20 Variables", top=20)
modfit$finalModel

## Predict on testing and print results
finpred1 <- predict(modfit, newdata=testing)
print(finpred1)

## Out of Sample Error on Validation Set
ooserr <- 1 - confusionMatrix(pred1, vtest$classe)$overall[1]
print(ooserr)