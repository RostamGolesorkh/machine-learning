library(mlr)
data(iris)

## Define the task
task <- makeClassifTask(id = "tutorial", data = iris, target = "Species")

## Define the learner
lrn <- makeLearner("classif.lda")

## Define the resampling strategy
rdesc <- makeResampleDesc(method = "CV", stratify = TRUE)

## Do the resampling
r <- resample(learner = lrn, task = task, resampling = rdesc, show.info = FALSE)

## Get the mean misclassification error
r$aggr
# mmce.test.mean 
#           0.02
# -----------------------------------------------------------------------------