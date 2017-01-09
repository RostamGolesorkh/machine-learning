#!/usr/bin/env Rscript

set.seed(1001)

require("doMC")
require("ROCR")
require("psych")
require("caret")
require("parallel")
require("corrplot")

source("../../MyFunction.R")

### PARALLEL CALCULATION IN R _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
registerDoMC(cores = detectCores())

### DATA PREPARATION _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
df0 <- read.csv2("train.csv", stringsAsFactors = F)
df_tst <- read.csv2("test.csv", stringsAsFactors = F)

str(df0$v9)
table(df0$v9)
table(is.na(df0$v9))

str(df0$v17)
table(df0$v17)
table(is.na(df0$v17))

str(df0$v29)
summary(df0$v29)
table(is.na(df0$v29))

str(df0$v20)
table(df0$v20)
table(is.na(df0$v20))

str(df0$v41)
table(df0$v41)
table(is.na(df0$v41))

str(df0$v31)
table(df0$v31)
table(is.na(df0$v31))

str(df0$v36)
table(df0$v36)
table(is.na(df0$v36))

str(df0$v19)
summary(df0$v19)
hist(df0$v19)
table(is.na(df0$v19))

str(df0$v2)
table(df0$v2)
table(is.na(df0$v2))

str(df0$v37)
table(df0$v37)
table(is.na(df0$v37))

str(df0$v12)
hist(df0$v12)
table(df0$v12)
table(is.na(df0$v12))

str(df0$v7)
table(df0$v7)
table(is.na(df0$v7))

str(df0$v27)
table(df0$v27)
table(is.na(df0$v27))

str(df0$v21)
table(df0$v21)
table(is.na(df0$v21))

str(df0$v39)
summary(df0$v39)
table(is.na(df0$v39))

str(df0$v34)
hist(df0$v34)
summary(df0$v34)
boxplot(df0$v34)
df0$v34 <- NULL
df_tst$v34 <- NULL

str(df0$v18)
hist(df0$v18)
summary(df0$v18)
boxplot(df0$v18)

str(df0$v35)
table(df0$v35)
summary(df0$v35)

df_trn <- df0[complete.cases(df0),]

nrow(df_trn)
str(df_trn)
summary(df_trn)

df_trn[, "data"] <- "train"
df_tst[, "data"] <- "test"

df <- rbind(df_trn, df_tst)
df <- MyFunction.RMV(df)

str(df)
table(is.na(df$v17))
df$v9 <- MyFunction.S2N(df$v9)

df$v17 <- as.numeric(df$v17)
table(df$v17)
df$v29 <- as.numeric(df$v29)

str(df$v29)
table(is.na(df$v29))

df_c1 <- as.data.frame(table(df$v9))
df_c2 <- df_c1[order(df_c1$Freq, decreasing=T),]
x <- as.vector(df_c2[1,1])
df[is.na(df[,i]),] <- x


df$v9 <- MyFunction.S2N(df$v9)
table(df$v9)
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

df_f <- MyFunction.RMV(df_tmp)

table(is.na(df_f))



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

