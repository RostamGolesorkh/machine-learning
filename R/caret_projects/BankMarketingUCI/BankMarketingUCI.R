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
df_full <- read.csv("BankMarketingUCI.csv", sep = ";") 

str(df_full)
summary(df_full[,5])
df_full[df_full == "unknown"] = NA
summary(df_full[,5])

nrow(df_full)
df = df_full[complete.cases(df_full),]
nrow(df)

dim(df)
str(df)
summary(df)
#------------------------------------------------------------------------------

### FUNCTION ------------------------------------------------------------------
str2num = function(x){
  lvls = as.vector(unique(x))
  lbls = c(1:length(lvls))
  xn = as.integer(factor(x, levels = lvls, labels = lbls))
  xn
}
#------------------------------------------------------------------------------

### DATA INVESTIGATI0N ________________________________________________________
df[,2] = str2num(df[,2])
df[,3] = str2num(df[,3])
df[,4] = str2num(df[,4])
df[,5] = str2num(df[,5])
df[,6] = str2num(df[,6])
df[,7] = str2num(df[,7])
df[,8] = str2num(df[,8])
df[,9] = str2num(df[,9])
df[,10]= str2num(df[,10])
df[,11]= str2num(df[,11])
df[,12]= str2num(df[,12])
df[,15]= str2num(df[,15])
df[,16]= str2num(df[,16])

summary(df)
str(df)
#------------------------------------------------------------------------------

### CORRELATION INVESTIGATION ________________________________________________
correlationMatrix = cor(df[,1:ncol(df)-1])
corrplot(correlationMatrix, method = "number")

highCorr = findCorrelation(correlationMatrix, 0.70)
df = df[, -highCorr]

correlationMatrix = cor(df[,1:ncol(df)-1])
corrplot(correlationMatrix, method = "number")
summary(df)
str(df)
#------------------------------------------------------------------------------

### Data Splitting ____________________________________________________________
set.seed(1001)
splitting_index = createDataPartition(df[,ncol(df)], p = 0.75, list = FALSE)

trn_df = df[ splitting_index,] # select 75% for train-data set
tst_df = df[-splitting_index,] # select 25% for test-data set
# -----------------------------------------------------------------------------

### CROSS-VALIDATION DEFINITION _______________________________________________
cv_ctrl = trainControl(method = "repeatedcv",
                       number = 10,
                      repeats = 10,
                            p = 0.75)
# -----------------------------------------------------------------------------

### EVALUATING SOME ALGORITHM _________________________________________________
### Linear Discriminant Analysis (LDA)
lda_model = train(trn_df[,1:ncol(trn_df)-1], 
                  trn_df[,  ncol(trn_df)  ],
                     method = "lda",
                     metric = "Accuracy",
                  trControl = cv_ctrl)
print(lda_model)

predict_y = predict(lda_model, tst_df[,1:ncol(tst_df)-1], type = "raw")
confusionMatrix(predict_y, tst_df[,ncol(tst_df)])

# estimate variable importance
importance = varImp(lda_model, scale=FALSE)
print(importance)
plot(importance)
# -----------------------------------------------------------------------------

### ROC CURE PLOT _____________________________________________________________
### computing a simple ROC curve
### (y-axis: tpr, x-axis: fpr)
probability_y = predict(lda_model, tst_df[,1:ncol(tst_df)-1], type = "prob")
prob = prediction(probability_y[2], tst_df[,ncol(tst_df)])
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
### k-Nearest-Neighbor (knn)
knn_model = train(trn_df[,1:ncol(trn_df)-1], 
                  trn_df[,  ncol(trn_df)  ],
                  method = "knn",
                  metric = "Accuracy",
                  trControl = cv_ctrl)
print(knn_model)

predict_y = predict(knn_model, tst_df[,1:ncol(tst_df)-1], type = "raw")
confusionMatrix(predict_y, tst_df[,ncol(tst_df)])

# estimate variable importance
importance = varImp(knn_model, scale=FALSE)
print(importance)
plot(importance)
# -----------------------------------------------------------------------------

### ROC CURE PLOT _____________________________________________________________
### computing a simple ROC curve
### (y-axis: tpr, x-axis: fpr)
probability_y = predict(knn_model, tst_df[,1:ncol(tst_df)-1], type = "prob")
prob = prediction(probability_y[2], tst_df[,ncol(tst_df)])
perf = performance(prob, measure = "tpr", x.measure = "fpr") 

auc_s4 = performance(prob, "auc")
auc_no = slot(auc_s4, "y.values")
auc_no = round(as.double(auc_no), 6)

legend_vec = c("knn","auc", auc_no)
legend_str = toString(legend_vec)

plot(perf, main="ROC curve", colorize=T)
abline(a=0, b=1)
legend(0.50, 0.25, c(legend_str), col=c('black'), lwd=1)
# -----------------------------------------------------------------------------

### EVALUATING SOME ALGORITHM _________________________________________________
### RANDOM FOREST
set.seed(1001)
rf_model = train(trn_df[,1:ncol(trn_df)-1],
                 trn_df[,  ncol(trn_df)  ],
                 method = "rf",
                preProc = c("center","scale"),
                 metric = "Accuracy",
              trControl = cv_ctrl)
print(rf_model)

predict_y = predict(rf_model, tst_df[,1:ncol(tst_df)-1], type = "raw")
confusionMatrix(predict_y, tst_df[,ncol(tst_df)])

# estimate variable importance
importance = varImp(rf_model, scale=FALSE)
print(importance)
plot(importance)
# -----------------------------------------------------------------------------

### ROC CURE PLOT _____________________________________________________________
### computing a simple ROC curve
### (y-axis: tpr, x-axis: fpr)
probability_y = predict(rf_model, tst_df[,1:ncol(tst_df)-1], type = "prob")
prob = prediction(probability_y[2], tst_df[,ncol(tst_df)])
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

### REMOVEING THE UNIMPOTANT COL. AND RF ALGORITHM ____________________________
rf_trn_df = trn_df
rf_tst_df = tst_df

rf_trn_df$default = NULL
rf_tst_df$default = NULL

### RANDOM FOREST
set.seed(1001)
rf_model = train(rf_trn_df[,1:ncol(rf_trn_df)-1],
                 rf_trn_df[,  ncol(rf_trn_df)  ],
                 method = "rf",
                 preProc = c("center","scale"),
                 metric = "Accuracy",
                 trControl = cv_ctrl)
print(rf_model)

predict_y = predict(rf_model, rf_tst_df[,1:ncol(rf_tst_df)-1], type = "raw")
confusionMatrix(predict_y, rf_tst_df[,ncol(rf_tst_df)])

# estimate variable importance
importance = varImp(rf_model, scale=FALSE)
print(importance)
plot(importance)
# -----------------------------------------------------------------------------

### ROC CURE PLOT _____________________________________________________________
### computing a simple ROC curve
### (y-axis: tpr, x-axis: fpr)
probability_y = predict(rf_model, rf_tst_df[,1:ncol(rf_tst_df)-1], type = "prob")
prob = prediction(probability_y[2], rf_tst_df[,ncol(rf_tst_df)])
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