install.packages("LiblineaR")


library(MASS)
library(caret)
library(LiblineaR)
data(Boston)


# Scaling data
Boston <- scale(Boston)


# data split
train_index <-sample(506 * 0.8)
train <- Boston[train_index, ]
test <- Boston[-train_index, ]


# find the best parameter "cost"
c_best <- heuristicC(train[, -1])


# parameter tuning for "svr_eps"
svr_eps_test <- seq(0.001, 0.1, 0.001)
cost <- NULL


for (i in svr_eps_test){
  cost <- c(cost, LiblineaR(data = train[, -1], target = train[, 1], type = 11, 
                                 cost = c_best, svr_eps = i, cross = 50))
}


svr_eps_best <- which.min(cost) * 0.001


# plot 
plot(svr_eps_test, cost)


# fitting model with l2 regularisation
model_l2 <- LiblineaR(data = train[, -1], target = train[, 1], type = 11, 
                    cost = c_best, svr_eps = svr_eps_best)
pred_l2 <- predict(model_l2, test[, -1])[[1]]
RMSE(pred_l2, test[, 1])


# fitting model without regularisation
model_ori <-LiblineaR(data = train[, -1], target = train[, 1], type = 11, 
                    cost = 1000000, svr_eps = svr_eps_best)
pred_ori <- predict(model_ori, test[, -1])[[1]]
RMSE(pred_ori, test[, 1])
