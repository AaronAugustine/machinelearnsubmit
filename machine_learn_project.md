# Predict Exercise Class
Aaron Augustine  
September 22, 2015  

#Executive Summary
The goal of this analysis was to complete a class project project is to predict the manner in which an exercise was conducted.  More information about this dataset is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

The analysis will apply multiple machine learning models to predict the "classe" variable.  From this work we found than the Random Forest model produced the best accuracy.  

#Data Analysis

##Download files
First we'll start downloading the training and testing data files.

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
From there I identified created a file called variable_info.csv.  In this file I indicated which variable I wanted to keep for analysis, specifically removing any variable that (a) Is an ID variable, (b) summary variable, or not well populated.  I read in the training and testing datasets and the variable info file.

```r
varinfo <-fread("./variable_info.csv",sep=',')      
training<-fread("./pml-training.csv",sep=',',stringsAsFactors=TRUE)
testing <-fread("./pml-testing.csv" ,sep=',',stringsAsFactors=TRUE)
```
Then I keep only the desired variables.

```r
nlist1<-varinfo[varinfo$keep==1]
nlist2<-as.vector(nlist1$colnum)
colnum<-as.numeric(nlist2)
training<-training[,colnum,with=FALSE]
training$classe<-as.factor(training$classe)
```

I further divided the training dataset into subtrain and subtest.  Subtrain would be used for modeling while subtest would be used for cross validation.  I used 50% of the data for training.  When I used 60% the models seemed to over fit.

```r
inTrain <- createDataPartition(y=training$classe,p=0.50, list=FALSE)
# subset data to training
subtrain <- training[as.numeric(inTrain),]
# subset data (the rest) to test
subtest <- training[-as.numeric(inTrain),]
# dimension of original and training dataset
rbind("original dataset" = dim(training),
      "subtrain set" = dim(subtrain),
      "subtest  set" = dim(subtest),
      "original test"= dim(testing))
```

```
##                   [,1] [,2]
## original dataset 19622   53
## subtrain set      9812   53
## subtest  set      9810   53
## original test       20  160
```

```r
subtrain<-data.frame(subtrain)
```
##Examine the data
After subsetting the variables, I plotted all of the predictor variables against the classe variable.  The code for this is given in the appendix and the figures were written out to the working directory with the file, predictor_plots.pdf.  Each of the predictors alone will not give a clean classification so my approach will be to start by feeding in all the predictors.

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
print("Linear Discriminant Analysis Model")
```

```
## [1] "Linear Discriminant Analysis Model"
```

```r
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
## Summary of sample sizes: 7850, 7849, 7851, 7849, 7849, 7850, ... 
## Resampling results
## 
##   Accuracy   Kappa      Accuracy SD  Kappa SD  
##   0.6977177  0.6175512  0.009196121  0.01163589
## 
## 
```

```r
print("Boosting with trees Model")
```

```
## [1] "Boosting with trees Model"
```

```r
#modfit1<-modfit0
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
## Summary of sample sizes: 7850, 7850, 7849, 7849, 7850, 7850, ... 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD
##   1                   50      0.7521413  0.6858566  0.012115867
##   1                  100      0.8178768  0.7695147  0.009436470
##   1                  150      0.8501500  0.8103304  0.010572321
##   2                   50      0.8522224  0.8127946  0.010713280
##   2                  100      0.9044717  0.8790979  0.009586160
##   2                  150      0.9303919  0.9119148  0.008546155
##   3                   50      0.8939736  0.8657729  0.011009645
##   3                  100      0.9411609  0.9255419  0.006583217
##   3                  150      0.9585885  0.9476099  0.005620493
##   Kappa SD   
##   0.015290174
##   0.011876283
##   0.013359190
##   0.013536749
##   0.012134185
##   0.010803655
##   0.013899753
##   0.008326334
##   0.007106999
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

```r
print("Random Forest Model")
```

```
## [1] "Random Forest Model"
```

```r
#modfit2<-modfit0
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
## Summary of sample sizes: 7849, 7850, 7852, 7849, 7848, 7850, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9869552  0.9834958  0.003172722  0.004016757
##   27    0.9873628  0.9840116  0.003508797  0.004441132
##   52    0.9815196  0.9766187  0.004153145  0.005255211
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

#WARNING adjust modfit pieces

##Results
Overall the Random Forest model produced the best accuracy.  Looking at the results the in-sample error would be 1.84%.  

```r
confusionMatrix(subtest$classe,predict(modfit2,subtest))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2786    3    0    0    1
##          B   26 1869    3    0    0
##          C    0   26 1680    5    0
##          D    0    0   21 1584    3
##          E    0    1    5    4 1793
## 
## Overall Statistics
##                                           
##                Accuracy : 0.99            
##                  95% CI : (0.9878, 0.9919)
##     No Information Rate : 0.2866          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9874          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9908   0.9842   0.9830   0.9944   0.9978
## Specificity            0.9994   0.9963   0.9962   0.9971   0.9988
## Pos Pred Value         0.9986   0.9847   0.9819   0.9851   0.9945
## Neg Pred Value         0.9963   0.9962   0.9964   0.9989   0.9995
## Prevalence             0.2866   0.1936   0.1742   0.1624   0.1832
## Detection Rate         0.2840   0.1905   0.1713   0.1615   0.1828
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9951   0.9903   0.9896   0.9957   0.9983
```
Applying the model to the subtest dataset, I would expect the out-of-sample error to be 1.07%.

##Project Submissions
Lastly, we write out the results from the random forest model for the project submission files

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
verify writeup to < 2000 words and the number of figures to be less than 5.
It will make it easier for the graders if you submit a repo with a gh-pages
branch so the HTML page can be viewed online (and you always want to make
it easy on graders :-).

#Appendix

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

