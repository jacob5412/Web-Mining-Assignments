library(caret)
library(knitr)
library(e1071)

cancer <- read.csv("risk_factors_cervical_cancer.csv")
cancer[cancer=='?'] <- NA

#Splitting Dataset
intrain <- createDataPartition(y = cancer$Dx.Cancer, p = 0.7, list = FALSE)
training<-cancer[intrain,]
testing<-cancer[-intrain,]
dim(training);dim(testing)
summary(cancer)
training[["Dx.Cancer"]] = factor(training[["Dx.Cancer"]])
trctrl <- trainControl(method = "repeatedcv", number = 2, repeats = 3)

#Training Model
model <- naiveBayes(Dx.Cancer ~ ., data = training)
class(model)
summary(model)
print(model)

#Testing Model
preds <- predict(model, newdata = testing)
conmat <- table(preds,testing$Dx.Cancer)

#Accuracy
accuracy <- (conmat[1]+conmat[4])/(conmat[1]+conmat[2]+conmat[3]+conmat[4])*100
accuracy

#ROC
library(pROC)
library(rowr)
prediction <- rev(seq_along(cancer$Dx.Cancer))
prediction[1:len(preds)] <- mean(as.numeric(preds))
roc_obj <- roc(cancer$Dx.Cancer,prediction)
auc(roc_obj)