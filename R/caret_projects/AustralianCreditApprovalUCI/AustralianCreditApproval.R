#!/usr/bin/env Rscript

set.seed(1001)

require("doMC")
require("ROCR")
require("psych")
require("caret")
require("parallel")
require("corrplot")

source("../../MyFunction.R")

### PARALLEL CALCULATION IN R _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
registerDoMC(cores = detectCores())

### DATA PREPARATION _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
df <- read.table("AustralianCreditApproval.txt", header = TRUE, sep = " ") 

nrow(df)
str(df)
summary(df)

table(is.na(df)) # there is no missing value

### DATA INVESTIGATI0N _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
df[, "class"] <- "one"
df$class[df$A15 == 0] <- "zero"
df$class <- as.factor(df$class)

table(df$class)
table(df$A15)

df$A15 <- NULL

### CORRELATION INVESTIGATION _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
corrMatrix <- cor(df[,1:(ncol(df) - 1)])
corrplot(corrMatrix, method = "number")

### PLOTS _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
pairs.panels(df)

### DATA SPLITTING _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
splitting_index <- createDataPartition(df[, ncol(df)], p = 0.75, list = FALSE)
df_trn <- df[ splitting_index,] # select 75% for train-data set
df_tst <- df[-splitting_index,] # select 25% for test-data set

ncol <- ncol(df)

x_trn <- df_trn[,1:(ncol-1)]
y_trn <- df_trn[,ncol]

x_tst <- df_tst[,1:(ncol-1)]
y_tst <- df_tst[,ncol]

cv_ctrl <- caret::trainControl(method = "repeatedcv", 
                               number = 10, 
                               repeats = 10, 
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary,
                               allowParallel = TRUE)

### EVALUATING SOME ALGORITHM _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
## k-Nearest Neighbors
model_knn <- caret::train(x_trn,
                          y_trn,
                          method = "knn",
                          metric = "ROC",
                          preProc = c("center","scale"),
                          trControl = cv_ctrl)

print(model_knn)

predict_y <- caret::predict.train(model_knn, x_tst, type = "raw")
caret::confusionMatrix(predict_y, y_tst, positive = "yes")

importance <- varImp(model_knn, scale = FALSE)
print(importance)
plot(importance)

MyFuntion.ROCPLOT(model_knn, x_tst, y_tst, "kNN")

## Linear Discriminant Analysis
model_lda <- caret::train(x_trn,
                          y_trn,
                          method = "lda",
                          metric = "ROC",
                          trControl = cv_ctrl)

print(model_lda)

predict_y <- caret::predict.train(model_lda, x_tst, type = "raw")
caret::confusionMatrix(predict_y, y_tst, positive = "yes")

importance <- varImp(model_lda, scale = FALSE)
print(importance)
plot(importance)

MyFuntion.ROCPLOT(model_lda, x_tst, y_tst, "LDA")

### RANDOM FOREST
model_rf <- caret::train(x_trn,
                         y_trn,
                         method = "rf",
                         preProc = c("center","scale"),
                         metric = "ROC",
                         trControl = cv_ctrl)

print(model_rf)

predict_y <- caret::predict.train(model_lda, x_tst, type = "raw")
caret::confusionMatrix(predict_y, y_tst, positive = "yes")

importance <- varImp(model_knn, scale = FALSE)
print(importance)
plot(importance)

MyFuntion.ROCPLOT(model_rf, x_tst, y_tst, "RF")
# -----------------------------------------------------------------------------