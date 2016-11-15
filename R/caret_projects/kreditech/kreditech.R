library(corrplot)
library(psych)
library(caret)
library(ROCR)

set.seed(1001)

## DATA PREPARATION ___________________________________________________________
df = read.csv("Training.csv", , sep = ";") 
?read.csv
table(duplicated(df))

nrow(df)
str(df)
summary(df)

df[df=="?"] = NA
df = df[complete.cases(df),]

nrow(df)
str(df)
summary(df)
#------------------------------------------------------------------------------

### FUNCTIONN STR2NUM _________________________________________________________
str2num = function(x){
  lvls = as.vector(unique(x))
  lbls = c(1:length(lvls))
  xn = as.integer(factor(x, levels = lvls, labels = lbls))
  xn
}

df[,1] = str2num(df[,1])
df[,2] = as.double(as.character.factor(df[,2]))
df[,4] = str2num(df[,4])
df[,5] = str2num(df[,5])
df[,6] = str2num(df[,6])
df[,7] = str2num(df[,7])
df[,9] = str2num(df[,9])
df[,10] = str2num(df[,10])
df[,12] = str2num(df[,12])
df[,13] = str2num(df[,13])
df[,14] = as.double(as.character.factor(df[,14]))

summary(df)
str(df)
#------------------------------------------------------------------------------

### CORRELATION INVESTIGATION ________________________________________________
correlationMatrix = cor(df[,1:15])
print(correlationMatrix)
corrplot(correlationMatrix, method = "number")

# Correlation plot shows columns 'A4' and 'A5' are identical and one of them
# must be removed. Therefore,

df$A5 = NULL

correlationMatrix = cor(df[,1:14])
print(correlationMatrix)
corrplot(correlationMatrix, method = "number")
#------------------------------------------------------------------------------

### PLOTS _____________________________________________________________________
pairs.panels(df)
#------------------------------------------------------------------------------

### DATA SPLITTING ____________________________________________________________
seed(1001)

splitting_index = createDataPartition(df[,15], p = 0.75, list = FALSE)

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

predict_y = predict(lda_model, test_df[,1:14], type = "raw")
confusionMatrix(predict_y, test_df[,15])
# -----------------------------------------------------------------------------

### ROC CURE PLOT _____________________________________________________________
### computing a simple ROC curve
### (y-axis: tpr, x-axis: fpr)
probability_y = predict(lda_model, test_df[,1:14], type = "prob")
prob = prediction(probability_y[2], test_df[,15])
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
rf_model = train(train_df[,1:14],train_df[,15],
                  method = "rf",
                 preProc = c("center","scale"),
                  metric = "Accuracy",
               trControl = cv_ctrl)
print(rf_model)

predict_y = predict(rf_model, test_df[,1:14], type = "raw")
confusionMatrix(predict_y, test_df[,15])
# -----------------------------------------------------------------------------

## ROC CURE PLOT ## ___________________________________________________________
## computing a simple ROC curve
## (y-axis: tpr, x-axis: fpr)
probability_y = predict(rf_model, test_df[,1:14], type = "prob")
prob = prediction(probability_y[2], test_df[,15])
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