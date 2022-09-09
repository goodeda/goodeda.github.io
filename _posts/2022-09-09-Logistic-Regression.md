---
layout: post
categories: English
title: Logistic Regression
tags: math, machine-learning
toc: true
math: true
date: 2022-09-09 15:23 +0300
---
As I'm taking the _Models and algorithms in NLP-application_ course, here I take notes of some classical machine learning models. In addition, I try to build the model without importing sklearn packages.

## A brief background
I believe linear regression is a widely-known quantitative methods in many fields especially in economics. It gives weights to variables and then add a bias as a random noise. The the formula: $\hat{y} = a_{1}*x_{1} + a_{2}*x_{2} + a_{3}*x_{3} +b$ written in matrix, $Y = A*X + b$ measures a linear relationship between variables and target. However, this method can't really deal with classification task because the range of $\hat{y}$ can be infinite. So, we try to limit the value in an interval of 0 and 1, representing the probability of the label. Then, the **Sigmoid** function is introduced. It is formulated as $y = \frac{1}{1+e^{-x}}$. The range of y is between 0 and 1, which perfectly meets the requirement. In another word, a linear regression plus the sigmoid function is the logistic regress  
Then next part discusses how to apply this approach in binary-classification task.

## Define a loss function
So far we have defined the function to fit data. There is a question that how we measure if the function fits data well or not? Loss function is what we need. Before defining the loss function, we should consider another thing.  
The probability of label should have something to do with the predicted result $\hat{y}$ and the true result $y$.
Here:
$$
\begin{equation}
P = \hat{y}^{y}*(1-\hat{y})^{1-y}
\end{equation}
$$
When y =1, the probability is $\hat{y}$, and when y = 0, the probability equals to $1-\hat{y}$. This means $\hat{y}=p\{y=1|x, a\}$, given data x and weight a.  
If we log on both sides, then it turns to:
$$
\log P = y*\log \hat{y} + (1-y)*\log (1-\hat{y})
$$
if y = 1, $\hat{y}$ should be close to 1, otherwise loss goes up
if y = 0, $\hat{y}$ should be close to 0, otherwise loss goes up.
However, in the above formula, when y =1, $\log P = \hat{y}$ is negative infinite. We hope it could be positive infinite. 
Therefore, the loss function is:
$$
\begin{equation}
    L(\hat{y}, y) = -(y*\log \hat{y} + (1-y)*\log (1-\hat{y}))
\end{equation}
$$

## Calculate gradients
Now there is a loss function, then we need to update the weights. So we need to calculate the gradients about A and b.
$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial \hat{y}} * \frac{\partial \hat{y}}{\partial (AX+b)} * \frac{\partial (AX+b)}{\partial A} \\
$$
The first and last items are pretty easy to compute. Let's look at the **Sigmoid** function.  
$$
\begin{equation}
    \begin{aligned}
Sigmoid(x)^{'}&=(\frac{1}{1+e^{-x}})^{'}\\
&= [(1+e^{-x})^{-1}]^{'} \\
&= -1* (1+e^{-x})^{-2}*(-e^{-x}) \\
&= \frac{1}{1+e^{-x}}*\frac{e^{-x}}{1+e^{-x}} \\
& = sig * (1- sig)
\end{aligned}
\end{equation}
$$
  
Thus, the equation can be concluded as:

$$
\begin{equation}
    \begin{aligned}
\frac{\partial L}{\partial A} &= -y*\frac{1}{\hat{y}}*\hat{y} (1-\hat{y})*X + (1-y)*\frac{1}{1-\hat{y}}(\hat{y}(1-\hat{y}))*X \\
&= y*(\hat{y}-1)*X + (1-y)*\hat{y}*X \\
&= (\hat{y}-y)*X \\
\end{aligned}
\end{equation}
$$  
The same for the bias gradient
$$
\begin{equation}
\begin{aligned}
\frac{\partial L}{\partial b} &=  \frac{\partial L}{\partial \hat{y}} * \frac{\partial \hat{y}}{\partial (AX+b)} * \frac{\partial (AX+b)}{\partial b} \\
&= \hat{y}-y
\end{aligned}
\end{equation}
$$
Then the weights will be updated through: $A = A - \alpha* \frac{\partial L}{\partial A}$

## More analysis
The original linear model is very useful in regression and classification (with sigmoid). There is also a concern about the formula due to the lack of weight limitation. We notice that the there is no limition on the values of weights. That's to say, the weights could be extremely large or small. There is a risk of overfitting and unfortunately, this will lead to very volatile results. --Suppose the data value is always large in training dataset, but the model doesn't work well when it meets small values.   

One solution is to add a penalty on the loss function. THe penalty can be $\Sigma a_{i}^{2}$ or $\Sigma |a_{i}|$, former of which is called **Ridge Regression** and the latter one is **Lasso Regression**. The benefit of doing so is the control of $A$ so that the values would not be too large. Besides, Lasso method is actually "selecting" useful features because it will make the parameter close to 0 as much as possible. For those variables whose weights are 0, they contribute nothing to the result and to some extent, only valuable variables will be kept. In fact, the penalty can be applied in classification to reduce the variation.

## Code
Coming soon.