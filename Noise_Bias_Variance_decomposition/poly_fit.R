
fit_pro <- function(x,y,M,new_colnames){
  model <- array()
  for(m in M){
    fit <- lm(y~poly(x,m,raw=TRUE))
    tmp_y <- predict(fit,data.frame(x=x))
    model <- cbind(model,tmp_y)
  }
  model <- data.frame(model)
  colnames(model) <- new_colnames
  model$x <- x
  model$y <- y
  return(model)
}

require('ggplot2')
############################                   ##########################
# polt
# data
x <- seq(0,7,0.1)
y <- sin(x) + rnorm(length(x))
M <- c(1,5,10)
new_colnames <- c('model','fit1','fit5','fit10')
model <- fit_pro(x,y,M,new_colnames)
p <- ggplot(data = model)
#p <- p+geom_point(aes(x,y))
p <- p + geom_line(aes(x=x,y=fit1),colour='red')
p <- p + geom_line(aes(x=x,y=fit5),colour='blue')
p <- p + geom_line(aes(x=x,y=fit10),colour='green')
############################                   ##########################
x <- seq(0,7,0.1)
M <- c(15) # 1,3,5
new_colnames <- c('model','fit')
Y <- array()
y <- sin(x)
plot(x,y,type='l',lwd = 5)
for(i in 1:30){
  y <- sin(x) + rnorm(length(x))
  model <- fit_pro(x,y,M,new_colnames)
  lines(x,model$fit,lwd = 0.0001,col = 'red')
  Y <- cbind(Y,model$fit)
}
lines(x,rowMeans(Y[,-1]),lwd = 5,col = 'green')