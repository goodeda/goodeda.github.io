---
layout: post
categories: 中文(Chinese)
title: Poisson Process
tags: math
toc: true
math: true
date: 2020-5-01 20:00 +0800
---
## Poisson Process
泊松过程主要刻画小概率事件在一段时间内发生的情况，在排队论，等待时间，累计损耗计算方面有很大用处。

### 泊松过程的三种定义
有计数过程N(t),如果满足以下条件：  
$$
	\left\{  
	\begin{array}{llll}  
		N(0)=0, 0\;\;time\; happens\; in\; 0\; time  & \\
		N(t)\; is\; process\; with\; stationary\; independent\; increments &\\  
		P\left\lbrace N(\Delta t)=1\right\rbrace =\lambda \Delta t+o(\Delta t) , \lambda>0 & \\  
		P\left\lbrace N(\Delta t)\geq2\right\rbrace=o(\Delta t)    
	\end{array}  
	\right.
$$

那么 $$ \left\lbrace N(t),t\geq 0\right\rbrace $$  是参数为 $$ \lambda $$ 的齐次泊松过程。  

有计数过程N(t),如果满足以下条件：\\
$$
	\left\{  
	\begin{array}{lll}  
	N(0)=0, 0 time\, happens\, in\, 0 time  & \\
	N(t)\; is\, process\, with\, stationary\, independent\, increments &\\   
	N(t)-N(s)\sim P(\lambda(t-s))    
	\end{array}  
	\right.
$$  
  
那么 $$ \left\lbrace N(t),t\geq 0\right\rbrace $$ 是参数为 $$ \lambda $$ 的齐次泊松过程。 

有更新计数过程 $$ N(t),\;T_{n} $$ 是每一次更新间隔时间，$$ T_{1},T_{2} \dots T_n $$ 独立同分布，那么如果满足以下条件： 
$$ T_{n}\sim E(\lambda) $$ Exponential Distr. $$ \left\lbrace N(t),t\geq 0\right\rbrace $$ 是参数为 $$ \lambda $$ 的齐次泊松过程。  

### 多次伯努利实验逼近泊松分布
伯努利试验得到的就是二项分布，事件A发生的概率为p，不发生的概率为1-p，泊松过程实际上也是一个计数过程。有 $$ { X(t)=n} $$ . 如果我们在t时间段内，做n次伯努利试验，也就是将t时间无限切割，每个时间段 $$ \Delta t=\dfrac{t}{n} $$ 设t时间内有k次事件发生，
2
$$
\begin{align*}
P\left\lbrace N_{t}=k\right\rbrace&=\binom{n}{k}(\lambda\Delta t)^{k}(1-\lambda\Delta t)^{n-k}\\
&= \dfrac{n!}{k!(n-k)!}(\dfrac{\lambda t}{n})^{k}(1-\dfrac{\lambda t}{n})^{n-k}\\
&={\lim_{n \to +\infty}}\dfrac{(\lambda t)^{k}}{k!}\dfrac{n(n-1)(n-2)\dots(n-k+1)}{n^{k}}(1-\dfrac{\lambda t}{n})^{\frac{-n}{\lambda t}\frac{-\lambda t(n-k)}{n}}\\
&={\lim_{n \to +\infty}}\dfrac{(\lambda t)^{k}}{k!}e^{\frac{-\lambda t (n-k)}{n}}\\
&=\dfrac{(\lambda t)^{k} e^{-\lambda t}}{k!}\\
&\Sigma_{i=1}^{N}k_{i}\rho-\theta^{2};\phi,\Phi
\end{align*}
$$
这就是参数为 $$ \lambda\, t $$ 的Poisson过程，其强度 $$ \lambda $$，尽管这种证明是不严格的，但是它揭示了不同分布在样本足够大的情况下有着渐进统一的关系。  

## A strict proof of Poisson process  
要证明在t时刻事件出现n次的概率是泊松分布，因为有平稳独立增量可以构造: 
   
$$
\begin{array}{l}  
\begin{align*}
P\left\lbrace \left[ N(t)-N(0)\right] =n\right\rbrace = P\left\lbrace\left[ N(t+t_{0}-N(t))\right]= n \right\rbrace    
&= \dfrac{(\lambda t)^{k} e^{-\lambda t}}{k!},(n=0,1,2,\dots)  
\end{align*}
\end{array}
$$


First we consider the initial condition,  

$$
\begin{array}{l}  
\begin{align*}
P_{0}(t+t_{0})&=P\left\lbrace N(t+t_{0}=0)\right\rbrace \\  
&= P\left\lbrace N(t)=0,N(t+h)-N(t)=0 \right\rbrace    \\
&= P_{0}(t)\left[ 1-\lambda t_{0} +o(t_{0})\right]    
\end{align*}  
\end{array}
$$   

$$
\Longrightarrow \dfrac{P_{0}(t+t_{0})-P_{0}(t)}{t_{0}}=-\lambda P_{0}(t)+\dfrac{o(t_{0})}{t_{0}} 
$$
  

When  $$ t_{0}\longrightarrow 0 $$ , we have  $$ \dfrac{dP_{0}(t)}{dt}=-P_{0}(t)\lambda $$  and  $$ P_{0}(0)=1 $$   
Therefore, $$ P_{0}(t)=e^{-\lambda t} $$   
Then,  
  
$$
\begin{align*}
P_{n}(t+t_{0})&=P_{n}(t)P_{0}(h)+P_{n-1}(t)P_{1}(h)+o(h)\\ \\
{\lim_{t_{0} \to 0}}\dfrac{P_{n}(t+t_{0})-P_{n}(t)}{t_{0}}&={\lim_{t_{0} \to 0}}-\lambda P_{n}(t)+\lambda P_{n-1}(t)+ \dfrac{o(t_{0})}{t_{0}}\\
\dfrac{dP_{0}(t)}{dt}&=-\lambda P_{n}(t)+\lambda P_{n-1}(t)
\end{align*}
$$  

We will have: $$ \dfrac{d\left[ e^{\lambda t}P_{n}(t)\right]}{dt}=\lambda e^{\lambda t}P_{n-1}(t) $$  

Since $$P_{1}(t)=\lambda t e^{-\lambda t}$$   

We can find that: $$\lambda e^{\lambda t}P_{n-1}(t)=\dfrac{\lambda (\lambda t)^{n-1}}{(n-1)!}$$  

With the initial condition,  

$$ P_{n}(t)=\dfrac{(\lambda t)^{k} e^{-\lambda t}}{k!} $$  

Given the second definition, it is a poisson process.


### 泊松过程的数字特征
我们已经知道了 $$ {X(t),t\geq0} $$ 是一个Poisson过程，那么 $$ X(t)=\dfrac{(\lambda t )^ke^{-\lambda t}}{k!} $$ (密度函数) 且之前泊松过程特征函数那一章有: $$ m_{X}(t)=E(X_{t})=\lambda t $$
我们可以推导出:  

$$
\begin{align*}
	D_{X}(t) =Var(X(t)) &= E(X^{2}(t))-E^{2}(N(t))\\
	&=\sum_{n=0}^{\infty}\dfrac{n^{2}(\lambda t)^{n} e^{-\lambda t}}{n!}-(\lambda t)^{2}\\
	&=\lambda t e^{-\lambda t} \sum_{n=1}^{\infty}\dfrac{(n-1+1)(\lambda t)^{n-1} }{n-1!}-(\lambda t)^{2}\\
	&=\lambda t  e^{-\lambda t} \left[ \sum_{n=2}^{\infty}\dfrac{(\lambda t)(\lambda t)^{n-2} }{n-2!}+\sum_{n=1}^{\infty}\dfrac{(\lambda t)^{n-1}}{(n-1)!}\right] -(\lambda t)^{2}\\
	&=\lambda t  e^{-\lambda t} \left[ \lambda t e^{\lambda t}+e^{\lambda t}\right] -(\lambda t)^{2}\\
	&= \lambda t
\end{align*}
$$

发现泊松过程的方差仍为 $$ \lambda t $$ 
自相关函数 $$ R_{X}(s,t)=E(X(s)X(t))  (0<s<t) $$ , 协方差函数 $$ \gamma_{X}(s,t)=R_{X}(s,t)-m_{X}(s)m_{X}(s) $$    
因为泊松过程有独立平稳增量，可以改写  

$$
\begin{align*}  
E(X(s)X(t))&=E(X(s)[X(t)-X(s)+X(s)])\\
&=E(X(s)[X(t)-X(s)])+E(X^{2}(s))\\
&=\lambda s \lambda (t-s)+(\lambda s)^{2}+\lambda s\\
&=\lambda^{2}st+\lambda s  \\  
\gamma_{X}(s,t)&=\lambda^{2}st+\lambda s-\lambda^{2}st\\
&= \lambda min(s,t)
\end{align*}
$$

### 泊松过程相关

关于之前的等待时间，给出结论:    
$$ T_{n} $$ 是更新的时间间隔，也就是两次事件发生的间隔，$$ T_{n} $$ 会服从 $$ P\left\lbrace T_{n}\leq t\right\rbrace =1-e^{-\lambda t} $$ 的指数分布函数  
$$ W_{n} $$ 则是第n次时间发生/到达的时间，$$ W_{n} $$ 是[0,t]上均匀分布的顺序统计量，其联合分布可以写成:   

Given $$ N(t)=n $$ , we have $$ f(T_{1},T_{2}, \dots,T_{n})=\dfrac{n!}{t^{n}} $$

对于复合泊松过程，如 $$ X(t)=\sum_{i=1}^{N(t)}\xi_{i},\xi_{i} $$ 均独立同分布，根据特征函数可以求得：  

$$
\begin{array}{l}
\begin{align*}
E[e^{iu\sum_{n=1}^{k}\xi_{n}}]&=\prod_{n=1}^{k}E(e^{iu\xi_{n}})=\varphi_{\xi}^{k}(u)  \\
\Rightarrow\varphi_{X(t)}(u)&=E[\varphi_{\xi}^{N(t)}(u)]  
\end{align*}
\end{array}
$$   


Finally we will get: 
$$
\begin{align*}
E[X(t)]&=\lambda t E(\xi)\\
D(X(t))&=\lambda t E(\xi^{2})
\end{align*}
$$

[link] Python模拟泊松过程。  
### 几道经典的例题

医院从早上8点开始接诊，每次专家只能看一名患者，平均需要20分钟，每名患者花的时间都是独立同指数分布。现求早上8点到12点门诊结束，成功就诊患者总共的等待时间。  
(1)每人花费时间平均20mins $$ \Rightarrow\lambda=3 persons/hour $$  
(2)问题中有两个随机变量，就诊时间(服从指数分布)和就诊人数(泊松过程)，可以先固定一个量     
Solution: 

$$ 
E[\sum_{n=1}^{N(4)}T_{n}]=E\left\lbrace E\left[ \sum_{n=1}^{N(4)}T_{n}\arrowvert N(4)\right] \right\rbrace       
$$  

Suppose N(4)=k,    
then,  \\

$$
E\left[ \sum_{n=1}^{k}T_{n}\arrowvert N(4)=k\right] = \sum_{n=1}^{k}E\left[T_{n}\right] =k \dfrac{4}{2} 
$$   

$$ T_{n} $$ should obey a uniform Distr. for T in[0,4]   

$$\Rightarrow 2 E[N(4)] = 24 $$ 
 

平均每6分钟一名顾客进商场，这一现象可以视作服从泊松过程。顾客进入商场购物的概率是0.6，每位顾客是否购买相互独立，且不受进入商城人数影响，求商场从早上10点营业开始到22点关门，单次购买商品顾客的人数。    
tip: 复合泊松过程,进入商城人数的泊松过程X(t), $$ \lambda=10 $$ persons/hour ,购买人数期望值 $$ E(\xi)=0.6 $$  
Solution:   

$$
	\begin{align*}
	E(N(12))&=E[\sum_{n=1}^{X(t)}\xi_{n}]\\
	&=\lambda t E(\xi)\\
	&= 10 * 12 * 0.6=720
	\end{align*}
$$

[0,t]时间内某系统受到冲击的次数N(t)形成参数为 $$ \lambda $$ 的Possion过程。每次冲击会造成 $$ Y_{i}(i=1,2,3,\dots,n) $$ 独立同分布的指数分布，其均值为 $$ \mu $$ 。设累计损害超过A时，系统就会终止运行。以T记系统运行时间/寿命，求系统平均寿命E(T)。  
(对于非负随机变量 $$ E(T)=\int_{0}^{\infty}P\left\lbrace T>t\right\rbrace dt $$ )   

tip:这里需要引进另一个分布 $$ \varGamma(n,\lambda) $$ 的概率密度函数 
$$ f(t)=\dfrac{\lambda e^{-\lambda t}(\lambda t)^{n-1}}{\varGamma(n)} $$   
当n=1时，$$ \varGamma(1,\lambda) $$ 退化为指数分布,$$ \varGamma(n)=(n-1)! $$  \\
Another tip:求积分的时候会用到 $$ \int_{0}^{\infty}\lambda e^{-\lambda t}(\lambda t)^{n}dt=\varGamma(n+1)=n! $$  
  
Text finished in 1st May, 2020
