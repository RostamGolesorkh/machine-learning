library(corrplot)
library(psych)
library(caret)
library(ROCR)

set.seed(1001)

## DATA PREPARATION ___________________________________________________________
df = read.table("AustralianCreditApproval.txt", header=T, sep = " ") 

nrow(df)
str(df)
summary(df)

table(is.na(df)) # there is no missing value
#------------------------------------------------------------------------------

### DATA INVESTIGATI0N ________________________________________________________
df[,15] = as.factor(df[,15])

summary(df)
str(df)
#------------------------------------------------------------------------------

### CORRELATION INVESTIGATION ________________________________________________
correlationMatrix = cor(df[,1:ncol(df)-1])
print(correlationMatrix)
corrplot(correlationMatrix, method = "number")
# there is no strong correlation
#------------------------------------------------------------------------------

### PLOTS _____________________________________________________________________
pairs.panels(df)
#------------------------------------------------------------------------------

### DATA SPLITTING ____________________________________________________________
set.seed(1001)

splitting_index = createDataPartition(df[, ncol(df)], p = 0.75, list = FALSE)
train_df = df[ splitting_index,] # select 75% for train-data set
 test_df = df[-splitting_index,] # select 25% for test-data set
str(train_df)
# -----------------------------------------------------------------------------

### EVALUATING SOME ALGORITHM _________________________________________________
### Linear Discriminant Analysis (LDA)
set.seed(1001)
cv_ctrl = trainControl(method = "repeatedcv",
                       number = 10,
                      repeats = 10,
                            p = 0.75)

lda_model = train(train_df[,1:14],train_df[,15],
                     method = "lda",
                    preProc = c("center","scale"),
                     metric = "Accuracy",
                  trControl = cv_ctrl)
print(lda_model)

str(test_df)

predict_y = predict(lda_model, test_df[,1:ncol(test_df)-1], type = "raw")
confusionMatrix(predict_y, test_df[,ncol(test_df)])
# -----------------------------------------------------------------------------

### ROC CURE PLOT _____________________________________________________________
### computing a simple ROC curve
### (y-axis: tpr, x-axis: fpr)
probability_y = predict(lda_model, test_df[,1:ncol(test_df)-1], type = "prob")
prob = prediction(probability_y[2], test_df[,ncol(test_df)])
perf = performance(prob, measure = "tpr", x.measure = "fpr") 

auc_s4 = performance(prob, "auc")
auc_no = slot(auc_s4, "y.values")
auc_no = round(as.double(auc_no), 6)

legend_vec = c("lda","auc", auc_no)
legend_str = toString(legend_vec)

plot(perf, main="ROC curve", colorize=T)
abline(a=0, b=1)
legend(0.50, 0.25, c(legend_str), col=c('black'), lwd=1)
# -----------------------------------------------------------------------------

### EVALUATING SOME ALGORITHM _________________________________________________
### RANDOM FOREST
set.seed(1001)
rf_model = train(train_df[,1:ncol(train_df)-1],train_df[,ncol(train_df)],
                  method = "rf",
                 preProc = c("center","scale"),
                  metric = "Accuracy",
               trControl = cv_ctrl)
print(rf_model)

predict_y = predict(rf_model, test_df[,1:ncol(test_df)-1], type = "raw")
confusionMatrix(predict_y, test_df[,ncol(test_df)])
# -----------------------------------------------------------------------------

## ROC CURE PLOT ## ___________________________________________________________
## computing a simple ROC curve
## (y-axis: tpr, x-axis: fpr)
probability_y = predict(rf_model, test_df[,1:ncol(test_df)-1], type = "prob")
prob = prediction(probability_y[2], test_df[,ncol(test_df)])
perf = performance(prob, measure = "tpr", x.measure = "fpr") 

auc_s4 = performance(prob, "auc")
auc_no = slot(auc_s4, "y.values")
auc_no = round(as.double(auc_no), 6)

legend_vec = c("rf","auc", auc_no)
legend_str = toString(legend_vec)

plot(perf, main="ROC curve", colorize=T)
abline(a=0, b=1)
legend(0.50, 0.25, c(legend_str), col=c('black'), lwd=1)
# -----------------------------------------------------------------------------