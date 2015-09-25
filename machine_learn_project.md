# Predict Exercise Class
Aaron Augustine  
September 22, 2015  

#Executive Summary 
The goal of this analysis was to complete a class project project is to predict the manner in which an exercise was conducted.  More information about this dataset is available from the website: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

The analysis will apply multiple machine learning models to predict the "classe" variable.  From this work I found than the Random Forest model produced the best accuracy.  

#Data Analysis

##Download files
First I started by downloading the training and testing data files.

```r
#download files
fileUrl1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
if (!file.exists("./pml-training.csv")) {
  message("downloading pml-training.csv...")
  download.file(fileUrl1, destfile = "./pml-training.csv")
  message("done downloading pml-training.csv...")
} else {
  message("pml-training.csv already exists...")
}
```

```
## pml-training.csv already exists...
```

```r
fileUrl2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if (!file.exists("./pml-testing.csv")) {
  message("downloading pml-testing.csv...")
  download.file(fileUrl2, destfile = "./pml-testing.csv")
  message("done downloading pml-testing.csv...")
} else {
  message("pml-testing.csv already exists...")
}
```

```
## pml-testing.csv already exists...
```
From there I created a file called variable_info.csv.  In this file I indicated which variables I wanted to keep for analysis, specifically removing any variable that (a) Is an ID variable, (b) summary variable, (c) or not well populated.  I read in the training and testing datasets and the variable info file using the code below.

```r
varinfo <-fread("./variable_info.csv",sep=',')      
training<-fread("./pml-training.csv",sep=',',stringsAsFactors=TRUE)
testing <-fread("./pml-testing.csv" ,sep=',',stringsAsFactors=TRUE)
```
Then I kept only the desired variables & set the classe variable as a factor variable.

```r
nlist1<-varinfo[varinfo$keep==1]
nlist2<-as.vector(nlist1$colnum)
colnum<-as.numeric(nlist2)
training<-training[,colnum,with=FALSE]
training$classe<-as.factor(training$classe)
```

I further divided the training dataset into subtrain and subtest.  Subtrain would be used for modeling while subtest would be used for cross validation.  I used 50% of the data for subtrain.  When I used 60% the models seemed to over fit.

```r
set.seed(123)
inTrain <- createDataPartition(y=training$classe,p=0.50, list=FALSE)
# subset data to training
subtrain <- training[as.numeric(inTrain),]
# subset data (the rest) to test
subtest <- training[-as.numeric(inTrain),]
subtrain<-data.frame(subtrain)
```
##Examine the data
After subsetting the variables, I plotted all of the predictor variables against the classe variable.  The code for this is given in the appendix and the figures were written out to the working directory with the file, predictor_plots.pdf.  Each of the predictors alone did not give a clean classification so my approach was to start by feeding in all the predictors.

##Modeling
For predicting the classe variable, I first set the resampling method to do repeated cross fold validation using 5 folds repeated 3 times.

```r
fitControl <- trainControl(## 5-fold CV
  method = "repeatedcv",
  number = 5,
  ## repeated three times
  repeats = 3)
```
From there I executed three model setting the method option to each of the following options: linear discriminant analysis, boosting with trees, and random forest.

```r
set.seed(123)
modfit0<-train(classe ~ ., data=subtrain, method="lda",trControl = fitControl)
modfit0
```

```
## Linear Discriminant Analysis 
## 
## 9812 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold, repeated 3 times) 
## Summary of sample sizes: 7849, 7850, 7850, 7850, 7849, 7849, ... 
## Resampling results
## 
##   Accuracy   Kappa      Accuracy SD  Kappa SD 
##   0.6992462  0.6194962  0.01099457   0.0137568
## 
## 
```

```r
modfit1 <- train(classe ~ ., data=subtrain, method="gbm",trControl =fitControl,verbose=FALSE)
modfit1
```

```
## Stochastic Gradient Boosting 
## 
## 9812 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold, repeated 3 times) 
## Summary of sample sizes: 7848, 7850, 7850, 7850, 7850, 7850, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD
##   1                   50      0.7533637  0.6874580  0.010198613
##   1                  100      0.8176038  0.7691612  0.011516716
##   1                  150      0.8511009  0.8115749  0.010959295
##   2                   50      0.8528666  0.8136054  0.013006972
##   2                  100      0.9053193  0.8801919  0.009181184
##   2                  150      0.9306286  0.9122133  0.007632918
##   3                   50      0.8957402  0.8680216  0.010575862
##   3                  100      0.9403789  0.9245637  0.005565582
##   3                  150      0.9589272  0.9480370  0.004957030
##   Kappa SD   
##   0.013092130
##   0.014642409
##   0.013905508
##   0.016509853
##   0.011639749
##   0.009670256
##   0.013418490
##   0.007040396
##   0.006269703
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
modfit2<-train(classe ~ . , data=subtrain, method="rf", trControl = fitControl)
modfit2
```

```
## Random Forest 
## 
## 9812 samples
##   52 predictor
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold, repeated 3 times) 
## Summary of sample sizes: 7851, 7849, 7850, 7849, 7849, 7850, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9862414  0.9825936  0.002632960  0.003331064
##   27    0.9870226  0.9835819  0.002836592  0.003589266
##   52    0.9818932  0.9770918  0.004410586  0.005579853
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

##Results
Overall the Random Forest model produced the best accuracy.  Looking at the results the in-sample error would be about 1.3%.  Applying the model to the subtest dataset using the code below, I would expect the out-of-sample error to be around 1.04%.  It's interesting that the out-of-sample error is slightly lower than the in-sample-error.  If I used more data for training I would expect the in-sample error to increase. 

```r
confusionMatrix(subtest$classe,predict(modfit2,subtest))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2782    7    0    0    1
##          B   26 1868    4    0    0
##          C    0   28 1679    4    0
##          D    0    0   22 1584    2
##          E    0    0    5    3 1795
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9896          
##                  95% CI : (0.9874, 0.9915)
##     No Information Rate : 0.2862          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9868          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9907   0.9816   0.9819   0.9956   0.9983
## Specificity            0.9989   0.9962   0.9960   0.9971   0.9990
## Pos Pred Value         0.9971   0.9842   0.9813   0.9851   0.9956
## Neg Pred Value         0.9963   0.9956   0.9962   0.9991   0.9996
## Prevalence             0.2862   0.1940   0.1743   0.1622   0.1833
## Detection Rate         0.2836   0.1904   0.1712   0.1615   0.1830
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9948   0.9889   0.9890   0.9963   0.9987
```


##Project Submissions
Lastly, I wrote out the results from the random forest model for the project submission files

```r
results<-as.character(predict(modfit2,testing))
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(results)
```

#Appendix
Code used to plot predictors

```r
plotfile<-data.frame(subtrain)
end<-length(plotfile)-1
pdf(file="predictor_plots.pdf")   
for (i in 1:end){
  m1 <- ggplot(plotfile, aes(x=plotfile[,i])) + ggtitle(names(plotfile)[i])
  m2 <- m1 + geom_histogram(aes(y = ..density..),binwidth=10) + geom_density()+ facet_grid(classe ~ .)
  print(m2)
}
dev.off()
```

```
## png 
##   2
```

