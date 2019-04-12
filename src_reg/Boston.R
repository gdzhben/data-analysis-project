# packages needed
install.packages("dplyr")
install.packages("caret")
install.packages("ggplot2")
install.packages("GGally")
install.packages("elasticnet")

library(MASS) # for Boston data set
library(ISLR) # for Boston data set
library(dplyr) # for data cleaning
library(caret) # for fitting models
library(ggplot2) # for visualizing density
library(GGally) # for correlation matrix
library(elasticnet) # for elastic net
attach(Boston) # load the data

# ==== data pre-processing ==== #
# missing value
mean(is.na(Boston))

# target variable density
ggplot(Boston, aes(crim))+stat_density(color="darkblue", fill="lightblue")+xlab("Boston criminal rate (crim)")

# determine correlation between predictors 
ggcorr(Boston, label = T, label_size = 2)+xlab('correlation coefficient between variables')

# near zero variance feature
nearZeroVar(Boston, names = T)

# split data in training and test set
set.seed(100)
train_ind <- createDataPartition(Boston$crim, p = 0.8, list = F)
train <- Boston[train_ind, ]
test <- Boston[-train_ind, ]

# ==== fit multiple regression models ==== #

# prepare training scheme
fitControl <- trainControl(method = "cv",
                           number = 10)

# ---- no regularisation ---- #
set.seed(2019)
lmfit <- train(crim ~., data = train,
               method = 'lm',
               trControl = fitControl,
               preProces = c('scale', 'center'))

# model coefficients 
coef(lmfit$finalModel)
summary(lmfit)
# predict on test set
lmfit.pred <- predict(lmfit, test)
sqrt(mean((lmfit.pred - test$crim)^2))
# lmfit.train <- predict(lmfit, train)
# sqrt(mean((lmfit.train - train$crim)^2))
# plot
plot(lmfit$finalModel)

# ----- ridge regression ---- #
set.seed(2019)
ridge <- train(crim ~., data = train,
              method='glmnet',
              tuneGrid = expand.grid(alpha = 0, 
                                    lambda = seq(0.0001, 1, length = 50)),
              trControl = fitControl,
              preProcess = c('scale', 'center'))
# prediction
ridge.pred <- predict(ridge, test)
sqrt(mean((ridge.pred - test$crim)^2))

ridge.train <- predict(ridge, train)
sqrt(mean((ridge.train - train$crim)^2))
# ridge regression result
ridge
plot(ridge)
plot(ridge$finalModel, xvar = "lambda", label = T, xlab = "log lambda in ridge regression")
abline(v=log(0.5306), col = "darkblue")
plot(ridge$finalModel, xvar = "dev", label = T)
plot(varImp(ridge, scale = T))
ridge$bestTune

# ---- lasso ---- #
set.seed(2019)
lasso <- train(crim ~., train,
               method = 'glmnet',
               tuneGrid = expand.grid(alpha = 1, 
                                      lambda = seq(0.0001, 1, length = 50)),
               preProcess = c('scale','center'),
               trControl = fitControl)
# prediction and model performance
lasso.pred <- predict(lasso, test)
sqrt(mean((lasso.pred - test$crim)^2))
# lasso.train <- predict(lasso, train)
# sqrt(mean((lasso.train - train$crim)^2))

# best model
lasso$bestTune

# lasso result
lasso
plot(lasso)
plot(lasso$finalModel, xvar = "lambda", label = T, xlab = "log lambda in lasso")
abline(v=log(0.0817), col = "darkblue")
plot(lasso$finalModel, xvar = "dev", label = T)
plot(varImp(lasso, scale = T))


# ---- elastic net ---- #
set.seed(2019)
elnet <- train(
  crim ~ ., 
  data = train,
  method = "glmnet",
  preProcess = c('scale','center'),
  trControl = fitControl,
  tuneGrid = expand.grid(lambda = seq(0.0001, 1, length = 50), 
                         alpha = seq(0, 1, length = 50))
)
# best model
elnet$bestTune
coef(elnet$finalModel, s= elnet$bestTune$lambda)
# model predictions
elnet.pred <- predict(elnet, test)
sqrt(mean((elnet.pred - test$crim)^2))
# elnet.train <- predict(elnet, train)
# sqrt(mean((elnet.train - train$crim)^2))
# result
elnet
# plot(elnet)
plot(elnet$finalModel, xvar = "lambda", label = T, xlab = "log lambda in elastic net")
abline(v=log(0.285), col = "darkblue")
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
