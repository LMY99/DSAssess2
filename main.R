OLS_lossgrad <- function(x,y,b){
  if(length(x)!=length(y)){
    stop("x and y must have the same length")
  }
  if(length(b)!=1) stop("b must be a scalar")
  return(sum(2*(y-b*x)*(-x)))
}

grad_descent_OLS <- function(x,y,
                             learning_rate=1e-3,
                             max_iter=100000,
                             convergence_limit=1e-6,
                             return_trace=FALSE)
{
  if(length(x)!=length(y))
    stop("x and y must have the same length")
  b <- 0
  trace <- b
  for(i in 1:max_iter){
    b_new <- b - learning_rate * OLS_lossgrad(x,y,b)
    if(is.infinite(b_new)){
      b <- b_new
      break
    }
    if(abs(b-b_new) <= convergence_limit){
      attr(b, "Iter_to_converge") <- i-1
      attr(b, "Status") <- "Success"
      if(return_trace) attr(b, "Trace") <- trace
      return(b)
    }
    else{
      b <- b_new
      trace <- c(trace, b_new)
    }
  }
  attr(b, "Iter_to_converge") <- i
  attr(b, "Status") <- "Failure"
  if(return_trace) attr(b, "Trace") <- trace
  return(b)
}

test_GD <- function(n,dimension,mean_x=rep(0,dimension),sd_x=rep(1,dimension),
                    b0=1,sd_y=rep(0.1,dimension),lr=10^seq(from=-4,to=0,by=0.5)){
  # Generate n samples for each learning rate
  # x ~ normal(mean_x,sd_x)
  # y ~ normal(b0*x,sd_y)
  # Return the absolute difference of the estimate from the analytic solution
  # and the number of updates until convergence
  result <- matrix(nrow=length(lr)*n,ncol=5)
  status <- rep("",nrow(result))
  if((length(mean_x)!=dimension)||(length(sd_x)!=dimension)||(length(sd_y)!=dimension))
    stop(sprintf("mean_x,sd_x,sd_y must have dimension %d"),dimension)
  j <- 1
  for(learning_rate in lr){
    for(i in 1:n){
      x <- rnorm(dimension,mean_x,sd_x)
      y <- rnorm(dimension,x*b0,sd_y)
      b.gd <- grad_descent_OLS(x,y,learning_rate)
      b.analytic <- sum(x*y)/sum(x*x)
      result[j,] <- c(lr=learning_rate,
                               GD=b.gd,
                               Analytic=b.analytic,
                               Diff=abs(b.gd-b.analytic),
                               iterations=attr(b.gd,"Iter_to_converge"))
      status[j] <- attr(b.gd,"Status")
      j <- j + 1
    }
  }
  colnames(result) <- c("learning_rate","GD","Analytic","Diff","Iterations")
  result <- as.data.frame(result)
  result$Status <- status
  return(as.data.frame(result))
}
set.seed(1126)
test_df <- test_GD(100,10)
pdf("GD_Diagnosis.pdf",width=14,height=7)
library(ggplot2)
library(dplyr)
print(
ggplot(test_df[test_df$Status=='Success',]) + geom_boxplot(aes(x=log10(learning_rate),
                                   y=Iterations,group=factor(learning_rate))) +
  scale_y_log10(labels=scales::label_comma()) + ggtitle("Boxplot of number of iterations until convergence for gradient descent") +
  theme(plot.title=element_text(hjust=0.5)) + xlab("log10(learning rate)")
)
print(
  ggplot(test_df[test_df$Status=='Success',]) + geom_boxplot(aes(x=log10(learning_rate),
                                     y=Diff,group=factor(learning_rate))) +
    scale_y_log10(labels=scales::label_comma()) + ggtitle("Boxplot of number of iterations until convergence for gradient descent") +
    theme(plot.title=element_text(hjust=0.5)) + ylab("Difference to true solution when algorithm converges") +
    xlab("log10(learning rate)")
)
conv_summary <-
  test_df %>% group_by(learning_rate) %>% summarise(convergence_rate=mean(Status=='Success'))
print(ggplot(conv_summary) + geom_line(aes(x=log10(learning_rate),
                                           y=convergence_rate)) +
        geom_point(aes(x=log10(learning_rate),
                       y=convergence_rate)) +
        scale_y_continuous(limits=c(0,1)) +
        ggtitle("Estimated probability of convergence") +
        xlab("log10(learning rate)") +
        theme(plot.title=element_text(hjust=0.5)))
# Demonstrate a failed GD algorithm
x <- rnorm(10)
y <- 2*x + rnorm(10,sd=0.1)
truth <- sum(x*y)/sum(x*x)
fail_result <- grad_descent_OLS(x,y,learning_rate=1,return_trace=TRUE)
fail_result <- attr(fail_result,'Trace')
print(
ggplot(data.frame(Trace=fail_result,Iteration=1:length(fail_result))[1:5,]) +
  scale_y_continuous() + geom_line(aes(y=Trace,x=Iteration)) +
  ylab("Updated value") + ggtitle("First few iterations of gradient descent(learning rate=1)") +
  theme(plot.title=element_text(hjust=0.5)) +
  geom_hline(yintercept=truth,linetype='dashed',color='red')
)
dev.off()