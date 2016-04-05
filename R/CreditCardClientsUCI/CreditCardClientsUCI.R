library(corrplot)
library(kernlab) 
library(caret)
library(dplyr)
library(psych)
library(ROCR)
library(pROC)

### GO PARALLEL IN R __________________________________________________________
library(parallel)
library(doMC)

no_cores = detectCores()-1    # Calculate the number of cores
registerDoMC(cores=no_cores)  # All subsequent models are run in parallel
# -----------------------------------------------------------------------------
set.seed(1001)
## DATA PREPARATION ___________________________________________________________
df = read.csv("CreditCardClientsUCI.csv")
head(df)

table(duplicated(df))
df = unique(df)
table(duplicated(df))

table(is.na(df))
dim(df)
# no missing value and dublicated rows

str(df)
summary(df)
#------------------------------------------------------------------------------

### DATA INVESTIGATION ________________________________________________________
# To apply svm algorithm, we have to do following transformation. 
df[,ncol(df)] = factor(df[,ncol(df)],
                       levels = unique(df[,ncol(df)]),
                       labels = c("one", "zero"))
head(df)
summary(df)
str(df)
#------------------------------------------------------------------------------

### CORRELATION INVESTIGATION ________________________________________________
correlationMatrix = cor(df[,1:ncol(df)-1])
corrplot(correlationMatrix, method = "number")

highCorr = findCorrelation(correlationMatrix, cutoff=0.75)
df = df[, -highCorr]

correlationMatrix = cor(df[,1:ncol(df)-1])
corrplot(correlationMatrix, method = "number")
#------------------------------------------------------------------------------

### PLOTS _____________________________________________________________________
pairs.panels(df)
#------------------------------------------------------------------------------

### SAMPLE OF XX% OF THE DATA _________________________________________________
dfs = df[sample(1:nrow(df),ceiling(nrow(df)*.2),replace=FALSE),]

dim(dfs)
str(dfs)
summary(dfs)
#------------------------------------------------------------------------------

### DATA SPLITTING ____________________________________________________________
set.seed(1001)

splitting_index = createDataPartition(dfs[,ncol(dfs)], p = 0.75, list = FALSE)

trn_df = dfs[ splitting_index,] # select 75% for train-data set
tst_df = dfs[-splitting_index,] # select 25% for test-data set
str(trn_df)
# -----------------------------------------------------------------------------

### EVALUATING SOME ALGORITHM _________________________________________________
### RANDOM FOREST
set.seed(1001)
cv_ctrl = trainControl(method = "repeatedcv",
                       number = 10,
                      repeats = 10,
                            p = 0.75)

rf_model = train(trn_df[,1:ncol(trn_df)-1],
                 trn_df[,  ncol(trn_df)],
                     method = "rf",
                    preProc = c("center","scale"),
                     metric = "Accuracy",
                  trControl = cv_ctrl)
print(rf_model)

predict_y = predict(rf_model, tst_df[,1:ncol(tst_df)-1], type = "raw")
confusionMatrix(predict_y, tst_df[, ncol(tst_df)])
# -----------------------------------------------------------------------------

## ROC CURE PLOT ______________________________________________________________
## computing a simple ROC curve
## (y-axis: tpr, x-axis: fpr)
probability_y = predict(rf_model, tst_df[,1:ncol(tst_df)-1], type = "prob")
prob = prediction(probability_y[2], tst_df[,ncol(tst_df)])
perf = performance(prob, measure = "tpr", x.measure = "fpr")

auc_s4 = performance(prob, "auc")
auc_no = slot(auc_s4, "y.values")
auc_no = round(as.double(auc_no), 6)

legend_vec = c("rf","auc", auc_no)
legend_str = toString(legend_vec)

plot(x = as.numeric(unlist(slot(perf, "x.values"))),
     y = as.numeric(unlist(slot(perf, "y.values"))),
     xlab = toString(slot(perf, "x.name")),
     ylab = toString(slot(perf, "y.name")),
     main = "ROC curve", type = "o")

#plot(perf, main="ROC curve", colorize=TRUE)
abline(a=0, b=1)
legend(0.50, 0.25, c(legend_str), col=c('black'), lwd=1)
# -----------------------------------------------------------------------------

### SETUP THE CROSS VALIDATION ________________________________________________
cv_ctrl = trainControl(method = "repeatedcv",
                       number = 10, 
                       repeats = 10,
                       p = 0.75,
                       summaryFunction = twoClassSummary,
                       classProbs = TRUE)
# -----------------------------------------------------------------------------

### EVALUATING SVM ALGORITHM __________________________________________________
# Automatic Grid Search
# Tune and Train the SVM algorithm
svm_tune = train(x = trn_df[,1:ncol(trn_df)-1],
                 y = trn_df[,  ncol(trn_df)],
                 method = "svmRadial",          # Radial kernel
                 tuneLength = 10,		   		      # 9 values of the cost function
                 preProc = c("center","scale"), # Center and scale data
                 metric = "ROC",
                 trControl = cv_ctrl)
svm_tune
plot(svm_tune)

predict_y = predict(svm_tune, tst_df[,1:ncol(trn_df)-1], type = "raw")
confusionMatrix(predict_y, tst_df[,  ncol(trn_df)])
# -----------------------------------------------------------------------------

### EVALUATING SVM ALGORITHM __________________________________________________
# Manually Grid Search
# Tune and Train the SVM algorithm
grid = expand.grid(sigma = seq(from=0.052, to=0.058, by=0.001), C = c(4.00))

svm_tune = train(x = trn_df[,1:ncol(trn_df)-1],
                 y = trn_df[,  ncol(trn_df)],
                 method = "svmRadial",          # Radial kernel
                 tuneGrid = grid,
                 preProc = c("center","scale"),  # Center and scale data
                 metric = "ROC",
                 trControl = cv_ctrl)
svm_tune
plot(svm_tune)

predict_y = predict(svm_tune, tst_df[,1:ncol(trn_df)-1], type = "raw")
confusionMatrix(predict_y, tst_df[,  ncol(trn_df)])
# -----------------------------------------------------------------------------

### EVALUATING SVM ALGORITHM __________________________________________________
# Linear Kernel
set.seed(1001)
svmL_tune = train(x = trn_df[,1:ncol(trn_df)-1],
                  y = trn_df[,  ncol(trn_df)],
                  method = "svmLinear",
                  preProc = c("center","scale"),
                  metric = "ROC",
                  trControl = cv_ctrl)
svmL_tune

predict_y = predict(svmL_tune, tst_df[,1:ncol(trn_df)-1], type = "raw")
confusionMatrix(predict_y, tst_df[,ncol(trn_df)])
# -----------------------------------------------------------------------------

### RESULTS ___________________________________________________________________
results = resamples(list(svm=svm_tune, svmL_tune))
results$values

summary(results)

bwplot(results, metric = "ROC",
       ylab = c("linear kernel", "radial kernel"))# boxplot
# -----------------------------------------------------------------------------