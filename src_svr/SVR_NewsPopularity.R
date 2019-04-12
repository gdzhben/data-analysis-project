install.packages("LiblineaR")


library(MASS)
library(caret)
library(LiblineaR)


# read data
news <- read.csv("c:/users/nculz/desktop/OnlineNewsPopularity.csv")


# data preprocessing using PCA
news[, 1:2] <- NULL
news_pca <- prcomp(news[, -59], center = TRUE, scale = TRUE)
var_prop <- news_pca$sdev ^ 2 / sum(news_pca$sdev ^ 2)
plot(cumsum(var_prop), xlab = "Principal Component", ylab = "Cumulative Variance", type = "b")


# first 30 variables contains 90% variance
news_new <- news_pca$x[, 1:30]


# only 1000 samples are used due to computational complexity
train_index <- sample(39644, 1000*0.8)
test_index <- sample(39644, 1000*0.2)
train <- news_new[train_index, ]
test <- news_new[test_index, ]


# find the best parameter "cost"
best_c <- heuristicC(train)


# fitting model with l2 regularisation
model_l2 <- LiblineaR(data = train, target = news[train_index, 59], type = 11, 
                      cost = best_c, svr_eps = 0.01)
pred_l2 <- predict(model_l2, test)[[1]]
RMSE(pred_l2, news[test_index, 59])


# fitting model without regularisation
model_ori <- LiblineaR(data = train, target = news[train_index, 59], type = 11, 
                       cost = 1000000, svr_eps = 0.01)
pred_ori <- predict(model_ori, test)[[1]]
RMSE(pred_ori, news[test_index, 59])