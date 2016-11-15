#!/usr/bin/env Rscript

set.seed(1001)

require("doMC")
require("ROCR")
require("psych")
require("caret")
require("parallel")
require("corrplot")

### PARALLEL CALCULATION IN R _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
registerDoMC(cores = detectCores())

### DATA PREPARATION _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
df <- read.table("AustralianCreditApproval.txt", header = TRUE, sep = " ") 

nrow(df)
str(df)
summary(df)

table(is.na(df)) # there is no missing value

### DATA INVESTIGATI0N _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
df[,15] <- as.factor(df[,15])

summary(df)
str(df)

### CORRELATION INVESTIGATION _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
corrMatrix <- cor(df[,1:ncol(df)-1])
print(corrMatrix)
corrplot(corrMatrix, method = "number")

### PLOTS _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
pairs.panels(df)

### DATA SPLITTING _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
splitting_index <- createDataPartition(df[, ncol(df)], p = 0.75, list = FALSE)
df_trn <- df[ splitting_index,] # select 75% for train-data set
df_tst <- df[-splitting_index,] # select 25% for test-data set

### EVALUATING SOME ALGORITHM _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
## Linear Discriminant Analysis (LDA)
cv_ctrl <- trainControl(method = "repeatedcv",
                        number = 10,
                       repeats = 10,
                             p = 0.75)

model_lda <- train(df_trn[,1:14],
                   df_trn[,15],
                   method = "lda",
                   preProc = c("center","scale"),
                   metric = "Accuracy",
                   trControl = cv_ctrl)
print(model_lda)

predict_lda <- predict(model_lda, df_tst[,1:ncol(df_tst)-1], type = "raw")
confusionMatrix(predict_lda, df_tst[,ncol(df_tst)])

### ROC CURE PLOT _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
### (y-axis: tpr, x-axis: fpr)
probability_lda <- predict(model_lda, df_tst[,1:ncol(df_tst)-1], type = "prob")
prob <- prediction(probability_lda[2], df_tst[,ncol(df_tst)])
perf <- performance(prob, measure = "tpr", x.measure = "fpr") 

auc_s4 <- performance(prob, "auc")
auc_no <- slot(auc_s4, "y.values")
auc_no <- round(as.double(auc_no), 6)

legend_vec <- c("lda","auc", auc_no)
legend_str <- toString(legend_vec)

plot(perf, main = "ROC curve", colorize = TRUE)
abline(a = 0, b = 1)
legend(0.50, 0.25, c(legend_str), col = c('black'), lwd = 1)
# ------------------------------------------------------------------------------