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
df_0 <- read.csv("BankMarketingUCI.csv", sep = ";", stringsAsFactors = FALSE) 

str(df_0)

str(df_0$age)
table(df_0$age)
table(is.na(df_0$age))

str(df_0$job)
table(df_0$job)
table(is.na(df_0$job))

str(df_0$marital)
table(df_0$marital)
table(is.na(df_0$marital))

str(df_0$education)
table(df_0$education)
table(is.na(df_0$education))

str(df_0$default)
table(df_0$default)
table(is.na(df_0$default))
df_0$default <- NULL

str(df_0$housing)
table(df_0$housing)
df_0$housing <- NULL

str(df_0$loan)
table(df_0$loan)

str(df_0$contact)
table(df_0$contact)

str(df_0$month)
table(df_0$month)

str(df_0$day_of_week)
table(df_0$day_of_week)

str(df_0$duration)
table(df_0$duration)

str(df_0$campaign)
summary(df_0$campaign)

str(df_0$pdays)
hist(df_0$pdays)
table(df_0$pdays)
df_0$pdays <- NULL

str(df_0$previous)
hist(df_0$previous)
table(df_0$previous)

str(df_0$poutcome)
table(df_0$poutcome)

str(df_0$emp.var.rate)
summary(df_0$emp.var.rate)
table(df_0$emp.var.rate)

str(df_0$cons.price.idx)
summary(df_0$cons.price.idx)

str(df_0$cons.conf.idx)
summary(df_0$cons.conf.idx)

str(df_0$euribor3m)
summary(df_0$euribor3m)
hist(df_0$euribor3m)

str(df_0$nr.employed)
summary(df_0$nr.employed)
hist(df_0$nr.employed)

str(df_0$y)
summary(df_0$y)
table(df_0$y)

df_0[df_0 == "unknown"] <- NA
df <- df_0[complete.cases(df_0),]

### DATA INVESTIGATI0N _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
df$job <- MyFuntion.S2N(df$job)
df$marital <- MyFuntion.S2N(df$marital)
df$education <- MyFuntion.S2N(df$education)
df$loan <- MyFuntion.S2N(df$loan)
df$contact <- MyFuntion.S2N(df$contact)
df$month <- MyFuntion.S2N(df$month)
df$day_of_week <- MyFuntion.S2N(df$day_of_week)
df$poutcome <- MyFuntion.S2N(df$poutcome)
df$y <- as.factor(df$y)

summary(df)
str(df)

### CORRELATION INVESTIGATION _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
corrMatrix <- cor(df[,1:(ncol(df) - 1)])
corrplot(corrMatrix, method = "number")

df$previous <- NULL
df$emp.var.rate <- NULL
df$euribor3m <- NULL

corrMatrix <- cor(df[,1:(ncol(df) - 1)])
corrplot(corrMatrix, method = "number")

### PLOTS _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
pairs.panels(df)

### MODEL PREPARATION _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
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