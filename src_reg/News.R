# packages needed
install.packages("dplyr")
install.packages("caret")
install.packages("ggplot2")
install.packages("GGally")
install.packages("elasticnet")
library(dplyr) # for data cleaning
library(caret) # for fitting models
library(ggplot2) # for visualizing density
library(GGally) # for correlation matrix
library(elasticnet) # for elastic net

setwd("~/Google Drive/2_SL/assignment2")
News = read.csv("OnlineNewsPopularity.csv", header = TRUE)
str(News)

# ==== data pre-processing ==== #
# missing value
mean(is.na(News))

# # target variable density
ggplot(News, aes(shares))+stat_density(color="darkblue", fill="lightblue")+xlab("Shares (crim)")

# near zero variance feature
colDelete <- nearZeroVar(News, names = F)

# eliminate unneccessary variables
News <- News[,-c(1,2,6,23)]

# # determine correlation between predictors 
# ggcorr(News, label = T, label_size = 2)+xlab('correlation coefficient between variables')

# split data in training and test set
set.seed(100)
train_ind <- createDataPartition(News$shares, p = 0.8, list = F)
train <- News[train_ind, ]
test <- News[-train_ind, ]

# ==== fit multiple regression models ==== #

# prepare training scheme
fitControl <- trainControl(method = "cv",
                           number = 10)

# ---- no regularisation ---- #
set.seed(2019)
lmfit <- train(shares ~., data = train,
               method = 'lm',
               trControl = fitControl,
               preProces = c('scale', 'center'))

# model coefficients 
coef(lmfit$finalModel)
summary(lmfit)
# predict on test set
lmfit.pred <- predict(lmfit, test)
sqrt(mean((lmfit.pred - test$shares)^2))
# lmfit.train <- predict(lmfit, train)
# sqrt(mean((lmfit.train - train$crim)^2))
# plot
plot(lmfit$finalModel)

# ----- ridge regression ---- #
set.seed(2019)
ridge <- train(shares ~., data = train,
              method='glmnet',
              tuneGrid = expand.grid(alpha = 0, 
                                    lambda = seq(5188.9,5189,length = 50)),
              trControl = fitControl,
              preProcess = c('scale', 'center'))
# prediction
ridge.pred <- predict(ridge, test)
sqrt(mean((ridge.pred - test$shares)^2))

# ridge.train <- predict(ridge, train)
# sqrt(mean((ridge.train - train$shares)^2))
# ridge regression result
ridge
plot(ridge, xlab = "lambda in ridge regression" )
plot(ridge$finalModel, xvar = "lambda", label = T, xlab = "log lambda in ridge regression")
abline(v=log(5188.951), col = "darkblue")
plot(ridge$finalModel, xvar = "dev", label = T)
plot(varImp(ridge, scale = T))
ridge$bestTune

# ---- lasso ---- #
set.seed(2019)
lasso <- train(shares ~., train,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha = 1, 
                                      lambda = seq(30, 31, length = 50)),
               preProcess = c('scale','center'),
               trControl = fitControl)
# prediction and model performance
lasso.pred <- predict(lasso, test)
sqrt(mean((lasso.pred - test$shares)^2))
# lasso.train <- predict(lasso, train)
# sqrt(mean((lasso.train - train$crim)^2))

# best model
lasso$bestTune

# lasso result
lasso
plot(lasso, xlab = "lambda in lasso regression" )
plot(lasso$finalModel, xvar = "lambda", label = T, xlab = "log lambda in lasso")
abline(v=log(30.79592), col = "darkblue")
plot(lasso$finalModel, xvar = "dev", label = T)
plot(varImp(lasso, scale = T))


# ---- elastic net ---- #
set.seed(2019)
elnet <- train(
  shares ~ ., 
  data = train,
  method = "glmnet",
  preProcess = c('scale','center'),
  trControl = fitControl,
  tuneGrid = expand.grid(lambda = seq(34, 35, length = 10), 
                         alpha = seq(0, 1, length = 50))
)
# best model
elnet$bestTune
coef(elnet$finalModel, s= elnet$bestTune$lambda)
# model predictions
elnet.pred <- predict(elnet, test)
sqrt(mean((elnet.pred - test$shares)^2))

# result
plot(elnet)
# plot(elnet)
plot(elnet$finalModel, xvar = "lambda", label = T, xlab = "log lambda in elastic net")
abline(v=log(34), col = "darkblue")
plot(elnet$finalModel, xvar = "dev", label = T)
plot(varImp(elnet))

# comparison
model_list <- list(LinearModel = lmfit, Ridge = ridge, Lasso = lasso, ElasticNet = elnet)
res <- resamples(model_list)
summary(res)
xyplot(res, metric = "RMSE")


# best model
get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}
get_best_result(elnet)
get_best_result(lasso)
get_best_result(ridge)
get_best_result(lmfit)
