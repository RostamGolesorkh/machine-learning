#!/usr/bin/env Rscript

set.seed(1001)

require("mlr")
require("doMC")
require("ROCR")
require("psych")
require("caret")
require("parallel")
require("corrplot")

### PARALLEL CALCULATION IN R _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
registerDoMC(cores = detectCores())

### DATA PREPARATION _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
df <- read.csv("JCS_dataset.csv") 

nrow(df)
str(df)
summary(df)

df[df == "?"] <- NA
df <- df[complete.cases(df),]

nrow(df)
str(df)
summary(df)

### STRING TO FACTOR TO INT NUMBER _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
S2N <- function(Scol) {
    
    # take a char vector as an input and return factor vector as an output with
    # the same length.
    #
    # Args:
    #   Scol: char[]
    # Returns:
    #   Ncol: int[]
    
    lvls <- as.vector(unique(Scol))
    lbls <- c(1:length(lvls))
    Ncol <- as.integer(factor(Scol, levels = lvls, labels = lbls))
    Ncol
}

### DATA PREPARATION _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
df[,1] <- S2N(df[,1])
df[,2] <- as.double(df[,2])
df[,4] <- S2N(df[,4])
df[,5] <- S2N(df[,5])
df[,6] <- S2N(df[,6])
df[,7] <- S2N(df[,7])
df[,9] <- S2N(df[,9])
df[,10] <- S2N(df[,10])
df[,12] <- S2N(df[,12])
df[,13] <- S2N(df[,13])
df[,14] <- as.integer(df[,14])

str(df)
summary(df)
table(df$A16)

### CORRELATION INVESTIGATION _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
correlationMatrix <- cor(df[,1:15])
print(correlationMatrix)
corrplot(correlationMatrix, method = "number")

df$A5 <- NULL

correlationMatrix <- cor(df[,1:14])
print(correlationMatrix)
corrplot(correlationMatrix, method = "number")

### PLOTS _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
pairs.panels(df)

### DATA SPLITTING _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
splitting_index <- caret::createDataPartition(df$A16,
                                              p = 0.75,
                                              list = FALSE)

df_trn <- df[ splitting_index,] # select 75% for train-data set
df_tst <- df[-splitting_index,] # select 25% for test-data set

remove(splitting_index)

### mlr PART _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
## Linear Discriminant Analysis (LDA)
classif_task <- mlr::makeClassifTask(id = "JCSPOSPrediction",
                                     data = df_trn,
                                     target = "A16",
                                     positive = "+")

mlr::getDefaultMeasure(classif_task)

# List classifiers that can output probabilities 
lrns <- listLearners("classif", properties = "prob")
head(lrns[c("class", "package")])
lrns[c("class", "package")]

# 2. Generate the learner
lrn_classif_lda <- mlr::makeLearner("classif.lda",
                                    predict.type = "prob",
                                    fix.factors.prediction = TRUE)
mlr::getDefaultMeasure(lrn_classif_lda)

lrn_classif_rf <- mlr::makeLearner("classif.randomForest",
                                   predict.type = "prob",
                                   fix.factors.prediction = TRUE)
mlr::getDefaultMeasure(lrn_classif_rf)

# 3. Specify the resampling strategy (5-fold cross-validation)
rs_desc <- mlr::makeResampleDesc("CV",
                                 iters = 5,
                                 stratify = TRUE)

rs_lda <- mlr::resample(learner = lrn_classif_lda,
                        task = classif_task,
                        resampling = rs_desc,
                        show.info = TRUE)
rs_lda$aggr

rs_rf <- mlr::resample(learner = lrn_classif_rf,
                       task = classif_task,
                       resampling = rs_desc,
                       show.info = TRUE)
rs_rf$aggr

# 4. Train the learner
model_lda_mlr <- mlr::train(lrn_classif_lda, classif_task)
model_rf_mlr <- mlr::train(lrn_classif_rf, classif_task)

mlr::getLearnerModel(model_lda_mlr)
mlr::getLearnerModel(model_rf_mlr)

model_lda_mlr$learner.model
model_rf_mlr$learner.model

pred_lda <- stats::predict(model_lda_mlr, newdata = df_tst)
pred_rf <- stats::predict(model_rf_mlr, newdata = df_tst)

head(getPredictionProbabilities(pred_lda))
head(getPredictionProbabilities(pred_rf))

mlr::getConfMatrix(pred_lda, relative = TRUE)
mlr::getConfMatrix(pred_rf, relative = TRUE)

mlr::performance(pred_lda, measures = auc)
mlr::performance(pred_rf, measures = auc)

roc_lda <- mlr::generateThreshVsPerfData(pred_lda, list(fpr, tpr))
roc_rf <- mlr::generateThreshVsPerfData(pred_rf, list(fpr, tpr))

mlr::plotROCCurves(roc_lda)
mlr::plotROCCurves(roc_rf)

### caret PART _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
cv_ctrl <- caret::trainControl(method = "repeatedcv",
                               number = 10,
                               repeats = 10,
                               p = 0.75)

model_lda_crt <- caret::train(df_trn[,1:14],
                              df_trn[,15],
                              method = "lda",
                              preProc = c("center","scale"),
                              metric = "Accuracy",
                              trControl = cv_ctrl)
print(model_lda_crt)

pred_lda_crt <- predict(model_lda_crt, df_tst[,1:14], type = "raw")
confusionMatrix(pred_lda_crt, df_tst[,15])

### ROC CURE PLOT _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
## computing a simple ROC curve
## (y-axis: tpr, x-axis: fpr)
probability_y <- predict(model_lda_crt, df_tst[,1:14], type = "prob")
prob <- prediction(probability_y[2], df_tst[,15])
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