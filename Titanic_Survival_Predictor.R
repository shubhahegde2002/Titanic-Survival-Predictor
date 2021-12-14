setwd("C:/Users/shubh/Downloads/KAGGLE-TITANIC")

install.packages("ggplot2")
install.packages("lattice")
library(ggplot2)

titanic.train <-read.csv(file="train.csv", header= TRUE)
titanic.test <-read.csv(file="test.csv", header= TRUE)

titanic.train$IsTrainSet <- TRUE
titanic.test$IsTrainSet <- FALSE

titanic.test$Survived <- NA

head(titanic.train)
head(titanic.test)

#Combining train and test data, and doing Data Preprocessing
titanic.full <- rbind(titanic.train,titanic.test)
#Structure 
str(titanic.full) 
#Number of indexes and columns 
dim(titanic.full)

#Check number of NaNs in dataset in each column; Age has 263 NaNs, Fare has 1 NaN and Embarked has 2 missing values ( empty not NaN) 
colSums(is.na(titanic.full))

#Return rows whose 'Embarked' is empty ( 2 rows)
#Only displayed the 'Embarked' of those rows O/P: "" ""
titanic.full[titanic.full$Embarked=='', "Embarked"]
titanic.full[titanic.full$Embarked=='', "Embarked"] <- 'S'#Fills with mode 'S'

table(is.na(titanic.full$Age))

#Check for outliers in fare. Values above the Upper Max are the outliers
boxplot(titanic.full$Fare)
#Gives Upper Bound ( Max ) 5th quartile
upper.whisker <- boxplot.stats(titanic.full$Fare)$stats[5]
outlier.filter <- titanic.full$Fare < upper.whisker #No outliers below lower whisker
#Rows that aren't outliers
titanic.full[outlier.filter,]
#Linear model for prediction of Fare for missing values
fare.equation = "Fare ~ Pclass + Sex + Age + SibSp + Parch + Embarked"
fare.model <- lm(
  formula = fare.equation,
  data = titanic.full[outlier.filter,]
)

#Index 1044 is the only one with Fare as NA; Displays columns of row 1044 
fare.row <- titanic.full[is.na(titanic.full$Fare),c("Pclass","Sex", "Age", "SibSp", "Parch", "Embarked")]
#Predicts fare for the row 1044
fare.predictions <- predict(fare.model, newdata = fare.row)
titanic.full[is.na(titanic.full$Fare), "Fare"] <- fare.predictions


#Same for Age; Here, there are outliers above upper AND below lower whisker
boxplot(titanic.full$Age)
upper.whisker2 <- boxplot.stats(titanic.full$Age)$stats[5]
lower.whisker2 <- boxplot.stats(titanic.full$Age)$stats[1]
outlier.filter2 <- titanic.full$Age < upper.whisker2 && titanic.full$Age > lower.whisker2

#Rows that aren't outliers
titanic.full[outlier.filter2,]
#Linear model for prediction of Age for missing values
age.equation = "Age ~ Pclass + Sex + Fare + SibSp + Parch + Embarked"
age.model <- lm(
  formula = age.equation,
  data = titanic.full[outlier.filter2,]
)

age.row <- titanic.full[is.na(titanic.full$Age),c("Pclass","Sex", "Fare", "SibSp", "Parch", "Embarked")]
#Predicts fare for the row with NA Age
age.predictions <- predict(age.model, newdata = age.row)
titanic.full[is.na(titanic.full$Age), "Age"] <- age.predictions

#Categorical casting: converts to ordinal factors, eg: Embarked only 3 levels: C, S , Q before it had missing values also
titanic.full$Pclass <- as.factor(titanic.full$Pclass)
titanic.full$Sex <- as.factor(titanic.full$Sex)
titanic.full$Embarked <- as.factor(titanic.full$Embarked)

subset(titanic.full,titanic.full$Age<0)
titanic.full = titanic.full[-c(181, 793, 864 , 1080), ]


#split dataset into train and test again
titanic.train <- titanic.full[titanic.full$IsTrainSet==TRUE,]
titanic.test <- titanic.full[titanic.full$IsTrainSet==FALSE,]

#Casts Survived into a category which tells us that it will be a binary classification
#Not regression or multiclass classification
titanic.train$Survived <- as.factor(titanic.train$Survived)

#Checking % of women that survived 
women <- nrow(subset(titanic.train, titanic.train$Sex=="female")) #Number of rows
x=subset(titanic.train, titanic.train$Sex=="female")
sur_women <- nrow(subset(x,x$Survived==1))
sur_women / women


#Checking % of men that survived
men <- nrow(subset(titanic.train, titanic.train$Sex=="male"))
x2=subset(titanic.train, titanic.train$Sex=="male")
sur_men <- nrow(subset(x2, x2$Survived==1))
sur_men/men

# The train set with the important features 
ind<-sample(1:dim(titanic.train)[1],500) # Sample of 500 out of 891
train1<-titanic.train[ind,] # The train set of the model
train2<-titanic.train[-ind,] # The validation set of the model


#MODEL
#Taking numeric features to train model
survived.equation <- "Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"
survived.formula <- as.formula(survived.equation)

install.packages("randomForest")
library(randomForest)

titanic.model <- randomForest(formula = survived.formula,data = train1, ntree = 500, mtry = 3, nodesize = 0.01 * nrow(titanic.test))

features.equation <- "Pclass + Sex + Age + SibSp + Parch + Fare + Embarked"

#Predicting on validation data
Survived_train2 <- predict(titanic.model, newdata = train2)

# Mean of the true prediction; Gives accuracy of Random Forest Model
accuracy = mean(Survived_train2==train2$Survived)
cat("Accuracy of Model is: ", accuracy*100,"%") #Prints accuracy of model using cat()

t1<-table(Survived_train2,train2$Survived)

# Precision and recall of the model
precision<- t1[1,1]/(sum(t1[1,]))
recall<- t1[1,1]/(sum(t1[,1]))
precision

# F1 score: Combines precision and recall, takes into account False Positives and False Negatives, More useful than accuracy
F1<- 2*precision*recall/(precision+recall)
cat('F1 Score of Model is: ', F1)
#Predicting on test data
Survived <- predict(titanic.model, newdata = titanic.test)
PassengerId <- titanic.test$PassengerId
output.df <- as.data.frame(PassengerId)
output.df$Survived <- Survived

#Output csv file with predictions 
write.csv(output.df, file= "Predictions_CLEANED.csv", row.names = FALSE)

