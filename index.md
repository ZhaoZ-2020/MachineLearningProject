---
title: "Machine Learning Project Report"
author: "Zhao Zheng"
date: "8/11/2020"
output:
  html_document: default
  pdf_document: default
---


## Link to my repository
Please refer to the link below for my repo, which contains the .md, .rmd, and data files:  
https://github.com/ZhaoZ-2020/MachineLearningProject


## Background

Devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity. The problem is that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  
In this project, I use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants, who were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal is to use the given training data to build a model, from which to predict what kind of activity the participants were performing in the testing data.

## Executive Summary
The training data set provided by the course is split into 60% (*training*) and 40% (*testing*), used for model building and model validation separately. Data cleaning and exploration are done on the *training* data set. Five models plus one combined model are built on this cleaning set, and they are:

1. Multinomial regression;
2. Decision tree;
3. Random Forest;
4. Boosting with trees;
5. Support vector machine;
6. Combined model using models 3-5.

Cross validation is done using K-fold method, and the final model I select is **Random Forest**. After apply it on my *testing* set, I manage to get an accuracy of 99.8%.   
Therefore, I use this model to predict the actives using the testing set provided to us (the one with 20 observations) and I get all the outcome correct (known from the quiz results).


## Data Slicing
I split the data into training set (60%) and testing set (40%), and all the analysis below was only done on the training set, except the model validation.  

```{r} 
data<-read.csv("./Data/pml-training.csv", header =T, na.strings=c("","NA","#DIV/0!"))
library(caret)
set.seed(123)
inTrain = createDataPartition(data$classe, p = 0.6)[[1]]
training = data[ inTrain,]; testing= data[-inTrain,]
dim(training); dim(testing) 
```
  
For cross-validation purpose, I further split the training data set into `k=5` folds. Later I will calculate the average of the accuracy from each model I built, across the five folds, to decide which is the best one.  

```{r}
set.seed(456)
folds<-createFolds(y=training$classe, k=5,
                   list=TRUE, returnTrain=TRUE)
sapply(folds,length)
```
  
As we can see the final *training* data set which we will be using to build the model has about 9420 observations, and it is still quite large.  

## Data Exploration
The training set (from the first fold) has 9420 observations and 160 variables. The outcome of interest is `classe` which has five categories.  


### Changing variables' format
Majority of the variables are numerical, except for user_name, cvtd_timestamp, new_window and classe.
The variable cvtd_timestamp contains the information of the data and time when the activity was performed, so I change it to the number of days comparing to the earliest date. For the rest of the string variables, I change them into factors.  

```{r}
train1<-training[folds[[1]],]
library(lubridate)
train1$cvtd_timestamp<-strptime(train1$cvtd_timestamp, "%d/%m/%Y %H:%M")
dates<-unique(train1$cvtd_timestamp)
firstdate<-dates[order(dates)][1]
train1$cvtd_timestamp<-as.double(train1$cvtd_timestamp - firstdate, units="days")

train1$user_name<-as.factor(train1$user_name)
train1$new_window<-as.factor(train1$new_window)
train1$classe<-as.factor(train1$classe)
str(train1[,c(2,5,6,160)])
```

  

### Removing variables
When looking at the summary tables of all the other variables, I notice there are quite a lot of NA values in many variables, and all of the NA values are under condition `new_window=='no'`.   

The p-value of the Chi-squared test is quite large at 0.6, which indicates the distribution of the five activities under `classe` is **not** significantly different for the two `new_window` categories. This gives me some confidence to remove those variables which have NAs for `new_window=='no'` (i.e. more than 90% of the values are NA).  

```{r}
table(train1$new_window, train1$class)
prop.table(table(train1$new_window, train1$class),1)
chisq.test(train1$new_window, train1$classe)
```
  
In addition, the two variables `x` and `raw_timestamp_part_1` are just running numbers for each observation, so do not contain useful information, and can be removed. The details of the R code please refer to the Appendix.  

Therefore, after the data cleaning mentioned above, the training data set now contains only 58 variables (including `classe`). 

  
## Model Building
(According to the project requirement, "Due to security concerns with the exchange of R code, your code will not be run during the evaluation by your classmates", I will not provide the full R codes here for my model fitting, but you can find the function I wrote in the Appendix as a reference.)  

I conduct a correlation check and find that some variables are highly correlated. However, the purpose of this study is to predict the `classe` outcome, rather than finding the relationship between the predictors and the outcome. The accuracy of the coefficients of the predictors are less of a concern, and if there is over-fitting problem, it will also be reflected by the out-of-sample error. Therefore, I decide to include all the variables into my model.  

Five models are fitted on the final training data set, and the accuracy were provided below comparing with the actual values of `classe` in the training set. This is also a indicator of the in-sample-error.  

The accuracy (on data set `train1`) are provided below:  

```{r}
accu_insample<-data.frame(0.7691, 0.5298, 0.9999, 0.9945, 0.9444)
names(accu_insample)<-c("Multinomial","Tree","Random_Forest","Boosting","SVM")
row.names(accu_insample)<-"In_sample_accuracy"
accu_insample
```
  
As we can see, the last three models all have accuracy more than 90%. I also fit a combined model with the predicted values from these three models as inputs. Then I apply these six models to the testing data set (from fold 1) to obtain their respective accuracy. This accuracy is also an indicator on the out-of-sample error (the higher the accuracy, the lower the out-of-sample error) of our models.  

Then I perform cross-validation by repeating the whole model fitting process using the other four folds and the final accuracy table below are the average of the five sets of out of sample accuracy.  

```{r}
accu_outsample<-data.frame(0.7658, 0.4521, 0.9962, 0.9882, 0.9350, 0.9968)
names(accu_outsample)<-c("Multinomial","Tree","Random_Forest","Boosting","SVM", "Combined")
row.names(accu_outsample)<-"Out_sample_accuracy"
accu_outsample
```
  
The combined model gives the highest accuracy, but is only slightly higher than the Random Forest (0.9968 vs 0.9962). Therefore I choose **Random Forest** as my final model, as it is simply and can be run faster than the combined model.  


## Model validation
All of the above analysis are done using 60% of the given data set, and I need to test my final model using the training set I created in the section: Data Slicing.  
The good news is the accuracy of the **Random Forest** is 0.997. This gives me confidence that my final model will provide a quite reliable prediction.  
Please note, in order to perform the model onto the testing data set, you will need to do the same data cleaning as described in the section: Data Exploration (refer to the 'dataprep()' in the Appendix).  


## Prediction
I apply my final model on the testing data set provided by the course (the one with 20 observations) and I get all the `classe` outcome correct.  

## Appendix 

### Removing varaibles with a lot of NAs
  
```{r}
removena<-function(data) {
    removeindex<-NULL
    for (i in 1:dim(data)[2]) {
        if ((sum(is.na(data[,i]))/dim(data)[1])>0.9) {
            removeindex<-c(removeindex, i)
        }
    }
    newdata<-data[,-removeindex]
    newdata
}

train1<-removena(train1)
library(dplyr)
train1<-train1 %>%
    select(-c(X,raw_timestamp_part_1))
dim(train1)
```
  
### Function for data preperation:
  
```{r}

dataprep<-function(testing, train1) {
    ## keep the same set of variables as of train1
    logicnm<-names(testing) %in% names(train1)
    testdata<-testing[,logicnm]
    
    ## change this variable to days (comparing to the earliest date)
    testdata$cvtd_timestamp<-strptime(testdata$cvtd_timestamp, "%d/%m/%Y %H:%M")
    dates<-unique(testdata$cvtd_timestamp)
    firstdate<-dates[order(dates)][1]
    testdata$cvtd_timestamp<-as.double(testdata$cvtd_timestamp - firstdate, units="days")
    
    ## change the format of user_name, new_window, and classe to factor
    testdata$user_name<-as.factor(testdata$user_name)
    testdata$new_window<-as.factor(testdata$new_window)
    testdata$classe<-as.factor(testdata$classe)
    
    ## Prepared data
    testdata
}
```
  
### Function for model fitting:
  
```{r}

getaccuracy<-function(train1, test1, seeds=888) {
    set.seed(seeds)
    fit.control <- trainControl(method = "repeatedcv", number = 2, repeats = 5)
    
    #model 1: Multinomial
    model1<-train(classe~., data=train1, preProcess=c("center","scale"), 
                  method = "multinom", trControl = fit.control, trace = FALSE)
    accu1<-confusionMatrix(predict(model1, newdata=test1), test1$classe)$overall[1]
    
    #model 2: decision tree
    model2<-train(classe~., data=train1, preProcess=c("center","scale"),
                  method="rpart", trControl = fit.control)
    accu2<-confusionMatrix(predict(model2, newdata=test1), test1$classe)$overall[1]
    
    #model 3: Random Forest
    model3<-train(classe~., data=train1, preProcess=c("center","scale"), 
                  method="rf", trControl = fit.control)
    accu3<-confusionMatrix(predict(model3, newdata=test1), test1$classe)$overall[1]
    
    #model 4: Boosting
    model4 <- train(classe~., data=train1, preProcess=c("center","scale"), 
                    method="gbm", verbose=FALSE, trControl = fit.control)
    accu4<-confusionMatrix(predict(model4, newdata=test1), test1$classe)$overall[1]
    
    #model 5: Support vector machine (svm)
    model5 <- svm(classe~., data=train1,
                  preProcess=c("center","scale"), trControl = fit.control)
    accu5<-confusionMatrix(predict(model5, newdata=test1), test1$classe)$overall[1]
    
    #combined model:
    pred1<-predict(model3, newdata=test1)
    pred2<-predict(model4, newdata=test1)
    pred3<-predict(model5, newdata=test1)
    predDF <- data.frame(pred1,pred2,pred3, classe=test1$classe)
    combModel <- train(classe ~., data=predDF, preProcess=c("center","scale"),
                       method="multinom",trControl = fit.control, trace = FALSE)
    accu.comb<-confusionMatrix(predict(combModel,predDF), test1$classe)$overall[1]
    
    #produce accuracy table
    modelaccuracy<-data.frame(accu1, accu2, accu3, accu4, accu5, accu.comb)
    names(modelaccuracy)<-c("Multinomial","Tree","Random_Forest","Boosting","SVM","CombinedModel")
    modelaccuracy  
}

```  
