set.seed(1001)

library("doMC")
library("parallel")

### PARALLEL CALCULATION IN R _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
registerDoMC(cores = detectCores())
# ------------------------------------------------------------------------------


library("mlr")

data(BreastCancer, package = "mlbench")

df0 <- BreastCancer
df0$Id <- NULL
df <- na.omit(df0)

str(df)
for (i in c(1:9)) {
    df[,i] <- as.integer(df[,i])
}

# 0. Make train and test data set
n <- nrow(df)
train_set <- sample(n, size = n*0.75)
test_set <- setdiff(seq(1:n), train_set)

df_trn <- df[train_set,]
df_tst <- df[test_set,]

# 1. Generate the task
classif_task <- makeClassifTask(id = "BreastCancerPrediction",
                                data = df_trn,
                                target = "Class",
                                positive = "malignant")
getDefaultMeasure(classif_task)

# List classifiers that can output probabilities 
lrns <- listLearners("classif", properties = "prob")
head(lrns[c("class", "package")])
lrns[c("class", "package")]

# 2. Generate the learner
classif_lda_lrn <- makeLearner("classif.lda",
                               predict.type = "prob",
                               fix.factors.prediction = TRUE)
getDefaultMeasure(classif_lda_lrn)

classif_rf_lrn <- makeLearner("classif.randomForest",
                               predict.type = "prob",
                               fix.factors.prediction = TRUE)
getDefaultMeasure(classif_rf_lrn)

# 3. Specify the resampling strategy (5-fold cross-validation)
rs_desc <- makeResampleDesc("CV",
                            iters = 5,
                            stratify = TRUE)

rs <- resample(learner = classif_lda_lrn,
               task = classif_task,
               resampling = rs_desc,
               show.info = TRUE)
rs$aggr

# 4. Train the learner
model_lda <- train(classif_lda_lrn, classif_task)
getLearnerModel(model_lda)
model_lda$learner.model

pred_y <- predict(model_lda, newdata = df_tst)
head(getPredictionProbabilities(pred_y))

getConfMatrix(pred_y, relative = TRUE)

performance(pred_y, measures = auc)

roc <- generateThreshVsPerfData(pred_y, list(fpr, tpr))
plotROCCurves(roc)

## Performance measures for classification with multiple classes
listMeasures("classif", properties = "classif.multi")
?listMeasures

# ------------------------------------------------------------------------------