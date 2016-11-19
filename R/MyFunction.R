### STRING TO FACTOR TO INT NUMBER _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
MyFuntion.S2N <- function(Scol) {
    
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
    
    return(Ncol)
}

### ROC PLOT _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
MyFuntion.ROCPLOT <- function(model, x_tst, y_tst, mstr) {
    y_prob <- predict(model, x_tst, type = "prob")
    prob <- prediction(y_prob[2], y_tst)
    perf <- performance(prob, measure = "tpr", x.measure = "fpr") 
    
    auc_s4 <- performance(prob, "auc")
    auc_no <- slot(auc_s4, "y.values")
    auc_no <- round(as.double(auc_no), 6)
    
    plot(perf, main = paste0(mstr, " ROC Curve"), colorize = T)
    abline(a = 0, b = 1)
    legend(0.50, 0.25, paste0("AUC = ", auc_no))
}
# ------------------------------------------------------------------------------
