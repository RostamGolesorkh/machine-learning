require("dplyr")
require("caret")
require("futile.logger")

### Data Preparation _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
functions.dataPreparation <- function(df) {
    str(df$age)
    table(df$age)
    table(is.na(df$age))
    
    str(df$job)
    table(df$job)
    table(is.na(df$job))
    
    str(df$marital)
    table(df$marital)
    table(is.na(df$marital))
    
    str(df$education)
    table(df$education)
    table(is.na(df$education))
    
    str(df$default)
    table(df$default)
    table(is.na(df$default))
    df$default <- NULL
    
    str(df$housing)
    table(df$housing)
    df$housing <- NULL
    
    str(df$loan)
    table(df$loan)
    
    str(df$contact)
    table(df$contact)
    
    str(df$month)
    table(df$month)
    
    str(df$day_of_week)
    table(df$day_of_week)
    
    str(df$duration)
    table(df$duration)
    
    str(df$campaign)
    summary(df$campaign)
    
    str(df$pdays)
    hist(df$pdays)
    table(df$pdays)
    df$pdays <- NULL
    
    str(df$previous)
    hist(df$previous)
    table(df$previous)
    
    str(df$poutcome)
    table(df$poutcome)
    
    str(df$emp.var.rate)
    summary(df$emp.var.rate)
    table(df$emp.var.rate)
    
    str(df$cons.price.idx)
    summary(df$cons.price.idx)
    
    str(df$cons.conf.idx)
    summary(df$cons.conf.idx)
    
    str(df$euribor3m)
    summary(df$euribor3m)
    hist(df$euribor3m)
    
    str(df$nr.employed)
    summary(df$nr.employed)
    hist(df$nr.employed)
    
    str(df$y)
    summary(df$y)
    table(df$y)
    
    df[df == "unknown"] <- NA
    df <- df[complete.cases(df),]
    
    df$jobNumeric <- as.integer(as.factor(df$job))
    df$maritalNumeric <- as.integer(as.factor(df$marital))
    df$educationNumeric <- as.integer(as.factor(df$education))
    df$loanNumeric <- as.integer(as.factor(df$loan))
    df$contactNumeric <- as.integer(as.factor(df$contact))
    df$monthNumeric <- as.integer(as.factor(df$month))
    df$day_of_weekNumeric <- as.integer(as.factor(df$day_of_week))
    df$contactNumeric <- as.integer(as.factor(df$contact))
    df$poutcomeNumeric <- as.integer(as.factor(df$poutcome))
    df$label <- as.factor(df$y)
    
    df <- dplyr::select(df,
                        age, duration, campaign, previous,jobNumeric, educationNumeric, educationNumeric, loanNumeric, contactNumeric,
                        monthNumeric, day_of_weekNumeric, contactNumeric, poutcomeNumeric, 
                        emp.var.rate, cons.price.idx, euribor3m, nr.employed,
                        label)
    
    return(df)
}

### TRAIN MODEL _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
functions.trainModel <- function(xTrain, yTrain, xTest, yTest, methodTrain) {
    
    # CROSS-VALIDATION DEFINITION
    cvControl <- caret::trainControl(method = "repeatedcv", 
                                     number = 10, 
                                     repeats = 10, 
                                     classProbs = TRUE,
                                     summaryFunction = twoClassSummary,
                                     allowParallel = TRUE)
    
    if (methodTrain == "lda") {
        # Linear Discriminant Analysis
        model <- caret::train(xTrain, yTrain,
                              method = "lda",
                              metric = "ROC",
                              trControl = cvControl)
    }
    
    if (methodTrain == "knn") {
        # k-Nearest-Neighbors (kNN)
        model <- caret::train(xTrain, yTrain,
                              method = "knn",
                              metric = "ROC",
                              preProc = c("center","scale"),
                              trControl = cvControl)
    }
    
    if (methodTrain == "rf") {
        # random forest
        model <- caret::train(xTrain, yTrain,
                              method = "rf",
                              metric = "ROC",
                              trControl = cvControl)
    }
    
    flog.info("Model information:", model, name = "functions", capture = TRUE)
    
    yModel <- caret::predict.train(model, xTest, type = "raw")
    confusion <- caret::confusionMatrix(yModel, yTest, positive = "yes")
    flog.info("Confusion matrix information:", confusion, name = "functions", capture = TRUE)
    
    importance <- caret::varImp(model, scale = FALSE)
    flog.info("Variable importance:", importance, name = "functions", capture = TRUE)
    plot(importance)

    return(model)
}

### ROC PLOT _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
functions.PlotROC <- function(model, xTest, yTest, mstr) {
    
    yPredict <- predict(model, xTest, type = "prob")
    yProbabity <- prediction(yPredict[2], yTest)
    yPerformance <- performance(yProbabity, measure = "tpr", x.measure = "fpr") 
    
    aucS4 <- performance(yProbabity, "auc")
    aucNo <- slot(aucS4, "y.values")
    aucNo <- round(as.double(aucNo), 6)
    
    plot(yPerformance, main = paste0(mstr, " ROC Curve"), colorize = T)
    abline(a = 0, b = 1)
    legend(0.50, 0.25, paste0("AUC = ", aucNo))
}

### REMOVE MISSING VALUES _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
functions.RemoveMissingValue <- function(df) {
    
    # replace missing values with the most repeated value in each column.
    #
    # Args:
    #   df: dataframe[]
    # Returns:
    #   df: dataframe[] 
    
    for(i in c(1:ncol(df))) {
        if (sum(is.na(df[,i])) != 0) {
            dfCol1 <- as.data.frame(table(df[,i]))
            dfCol2 <- dfCol1[order(dfCol1$Freq, decreasing = T),]
            x <- as.vector(dfCol2[1,1])
            df[is.na(df[,i]),] <- x
        }
    }
    
    return(df)
}
# --------------------------------------------------------------------------------------------------

