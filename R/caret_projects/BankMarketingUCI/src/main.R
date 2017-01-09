#!/usr/bin/env Rscript

set.seed(1001)

require("doMC")
require("ROCR")
require("psych")
require("caret")
require("parallel")
require("corrplot")
require("futile.logger")

source("src/functions.R")

### PARALLEL CALCULATION IN R _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
registerDoMC(cores = detectCores())

### DATA INVESTIGATI0N _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
df0 <- read.csv("data/BankMarketingUCI.csv", sep = ";", stringsAsFactors = FALSE)
df <- functions.dataPreparation(df0)

### CORRELATION INVESTIGATION _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
corrMatrix <- cor(df[,1:(ncol(df) - 1)])
corrplot(corrMatrix, method = "number")

df$emp.var.rate <- NULL
df$euribor3m <- NULL

### PLOTS _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
pairs.panels(df)

### MODEL PREPARATION _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
splittingIndex <- caret::createDataPartition(df$label,
                                             p = 0.75,
                                             list = FALSE,
                                             times = 1)

dfTrain <- df[splittingIndex, ] # select 75% for train-data set
dfTest <- df[-splittingIndex, ] # select 25% for test-data set

nX <- ncol(df) - 1
xTrain <- dfTrain[, 1:nX] 
yTrain <- dfTrain$label
xTest <- dfTest[, 1:nX]
yTest <- dfTest$label

### TRAIN MODELS _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
modelKNN <- functions.trainModel(xTrain, yTrain, xTest, yTest, methodTrain = "knn")
modelRF <- functions.trainModel(xTrain, yTrain, xTest, yTest, methodTrain = "rf")

functions.plotROCcurve(modelKNN, xTest, yTest, "KNN")
functions.plotROCcurve(modelRF, xTest, yTest, "RF")
# --------------------------------------------------------------------------------------------------