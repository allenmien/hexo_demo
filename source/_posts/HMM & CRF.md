---
title: HMM & CRF
---
# HMM & CRF

## HMM

### HMM的原理推导

Language 序列如下：

![](https://markdocpicture.oss-cn-hangzhou.aliyuncs.com/iPic/2019-03-27-031254.jpg)

> $x$是可以观察到的，而y是隐藏的状态。
>
> 转移概率：$P(PN|start)$，是从开始到第一个单词为专有名词的概率，同样的$P(V|PN)$是从专有名词，到动词的概率。
>
> 发射概率：$P(John|PN)$，是当词性为专有名词的时候，该词为John的概率。

那么对于如下的序列，产生该序列的概率为：
$$
P(x,y) = P(PN|start) * P(John|PN) *P(V|PN)*P(saw|V)*P(D|V)*P(the|D)*P(N|D)*P(saw|D) \\= P(y) * P(x|y)
$$
也就是说，序列的随机场等于，该位置的转移概率*发射概率，然后每一个位置叠成。简化后公式如下：
$$
P(x,y) = P(y)*P(x|y)
$$

$$
P(y) = P(y_1|start) \times \prod_{l=1}^{L-1}P(y_{l+1}|y_l) \times P(end|y_L)
$$

$$
P(x|y) = \prod_{l=1}^{L}P(x_l|y_l)
$$

#### HMM预测

**最有可能发生**

![](https://markdocpicture.oss-cn-hangzhou.aliyuncs.com/iPic/2019-03-27-060703.jpg)

也就是，给我们一个可以看到的序列，我们去预测最可能的隐藏状态。举例子来讲就是，给了一句话，去预测每个词等词性。在真正的使用时，其实是我们自己假设遍历所有可能得隐藏状态序列，然后计算出所有的序列的概率，然后取概率最大的一个就是最有可能发生的情况。其实也就是所有的可能中最大的一个$P(x,y)$ ，这里显然就是一个动态规划问题，也就是用维比特进行计算求全局最优解就是了。

用公式表示就是：
$$
y = \mathop{\arg\min}_{y\epsilon Y} P(y|x) = \mathop{\arg\min}_{y\epsilon Y} \frac{P(x,y)}{P(x)} = \mathop{\arg\min}_{y\epsilon Y} P(x,y) \\= \mathop{\arg\min}_{y\epsilon Y} \ \ P(y_1|start)\prod_{l=1}^{L-1}P(y_{l+1}|y_l)P(end|y_L)\prod_{l=1}^{L}P(x_l|y_l)
$$

#### HMM训练

**推测依据**

我们自然会问，你遍历所有的可能计算的概率，其实你必须有每一个的转移概率和发射概率，那么这些概率你是怎么得来的，难道是瞎猜的么。

不是的，这些概率是需要从历史的样本中去观察的。

这个过程叫做训练模型，实际上，我们真的是只用观察就可以统计到所有的转移概率和发射概率。

就比方说，我们有1000个序列，找一些语言学家，帮助我们标注好，那些词是动词，那些是名词。

那么我们就可以这样统计：

转移概率：$P(V|PN) = \frac{PN后接V总数}{PN后接V、D等总数}$

发射概率：$P(John|PN) = \frac{PN发射为John的总数}{PN发射为John、Mark等总数}$

写的完整一点就是：
$$
P(x,y) = P(y_1|start)\prod_{l=1}^{L-1}P(y_{l+1}|y_l)P(end|y_L)\prod_{l=1}^{L}P(x_l|y_l)
$$

$$
P(y_{l+1}=s'|y_l =s) = \frac{count(s \to s')}{count(s)}
$$

$$
P(x_l =t|y_l=s) = \frac{count(s \to t)}{count(s)}
$$

其实HMM的训练过程是一个统计过程，HMM的预测过程就是叠乘，而为了减少HMM预测过程遍历导致的运算量的优化算法，就是维特比算法。

其实HMM到这里就已经结束了。

可是为什么还要有CRF或者Structure SVM接下来的算法呢？

因为HMM有他的算法缺陷

### HMM的弱点

其实HMM是隐马尔可夫模型。那么什么是马尔可夫呢？

下面是一系列定义：

> 随机场 ：随机场是由若干个位置组成的整体，当给每一个位置中按照某种分布随机赋予一个值之后，其全体就叫做随机场。还是举词性标注的例子：假如我们有一个十个词形成的句子需要做词性标注。这十个词每个词的词性可以在我们已知的词性集合（名词，动词...)中去选择。当我们为每个词选择完词性后，这就形成了一个随机场。
>
> 马尔科夫随机场：马尔科夫随机场是随机场的特例，它假设随机场中某一个位置的赋值仅仅与和它相邻的位置的赋值有关，和与其不相邻的位置的赋值无关。
>
> 条件随机场是马尔科夫随机场的特例，它假设马尔科夫随机场中只有X和Y两种变量，X一般是给定的，而Y一般是在给定X的条件下我们的输出。这样马尔科夫随机场就特化成了条件随机场。
>
> 线性链条件随机场：X和Y有相同的结构的CRF就构成了线性链条件随机场。

其实马尔可夫随机场的假设，是当前的位置的赋值，只和临近的位置相关，和其他位置无关，其实这种假设是非常粗糙的。这也是我们接下来举得例子会有问题的原因。

![](https://markdocpicture.oss-cn-hangzhou.aliyuncs.com/iPic/2019-03-27-064127.jpg)

假如:

N-V-c = 9

P-V-a =9

N-D-a= 1

那么：

$P(V|N) = \frac{9}{10}, P(D|N) = \frac{1}{10},P(V|P)=1​$

$P(a|V) = \frac{1}{2}, P(c|V) = \frac{1}{2},P(a|D)=1​$

可以计算：

$P(N, V, a) = \frac{9}{20}$

$P(N, V,c) = \frac{9}{20}$

 $P(N,D,a) = \frac{1}{10}$

可以看到一个在样本中从未出现的序列$P(N,V,a)$比在样本中出现过的$P(N,D,a)$还要大。

这是非常不正常的。

而这个问题是可以用CRF来解决的。

## CRF

### CRF原理推导

CRF和HMM最大的不同就是，HMM中的$P(y_1|start), P(x_l|y_l)$等是通过统计的方式获得的，但是在CRF中是经过SGD等方法收敛求解的。

先从HMM的$P(x, y)​$看：
$$
P(x,y) = P(y_1|start)\prod_{l=1}^{L-1}P(y_{l+1}|y_l)P(end|y_L)\prod_{l=1}^{L}P(x_l|y_l)
$$
两边求对数可得：
$$
logP(x,y) = logP(y_1|start) + \sum_{l=1}^{L-1}logP(y_{l+1}|y_l) + logP(end|y_L) + \sum_{l=1}^{L}logP(x_l|y_l)
$$
也就是说，上面的每一项，都是通过SGD求得的。

![](https://markdocpicture.oss-cn-hangzhou.aliyuncs.com/iPic/2019-03-27-092219.jpg)

以$\sum_{l=1}^LlogP(x_l|y_l)$举例：
$$
\sum_{l=1}^{L}logP(x_l|y_l) = logP(the|D) + logP(dog|N) + logP(ate|V) + logP(the|D) + logP(homework|N) \\= logP(the|D) \times 2 + logP(dog|N) \times 1 + logP(ate|V) \times 1 + logP(homework|N) \times 1
$$

从上面的推导中可以看到：
$$
\sum_{l=1}^{L}logP(x_l|y_l) = \sum_{s,t}logP(t|s) \times N_{s,t}(x,y)
$$
其中：

$P(t|s)$：类似$P(the|D)$，在s词性的情况下，是单词t的概率。该值和HMM中的发射概率是一个东西，但是HMM中可以通过简单的统计获得，而在CRF中，这个是未知的，也是SGD待求的参数。

$N_{s,t}(x,y)$：代表的是，在给的样本序列中，对应的P(t|s)的个数。



那么同理的其他三项也可以同样的表示：
$$
\sum_{l=1}^{L}logP(x_l|y_l) = \sum_{s,t}logP(t|s) \times N_{s,t}(x,y)
$$

$$
logP(y_1|start) = \sum_S logP(s|start) \times N_{start,s}(x,y)
$$

$$
logP(y_{l+1}|y_l) = \sum_{S,S'} logP(s'|s) \times N_{S,S'}(x,y)
$$

$$
logP(y_{end}|y_L) = \sum_{S} logP(end|S) \times N_{S,end}(x,y)
$$

其中，$logP(y_1|start)$ 的意义为：

在给定的样本中，$y_1$为序列开始的概率 $\times$ 所有以$y_1$为开始的数量。

其他亦然。

那么：$logP(y_1|start), logP(y_{l+1}|y_l), logP(y_{end}|y_L), logP(x_l,y_l)$都是待求的 参数。



那么：
$$
logP(x,y) = logP(y_1|start) + \sum_{l=1}^{L-1}logP(y_{l+1}|y_l) + logP(end|y_L) + \sum_{l=1}^{L}logP(x_l|y_l)
$$

$$
= \left[\begin{matrix} .\\.\\ logP(t|s)\\.\\.\\logP(s|start)\\.\\.\\logP(s'|s)\\.\\.\\logP(end|s)\\.\\.\\    \end{matrix}\right]\tag{1} \cdot \left[\begin{matrix} .\\.\\ N_{s,t}(x,y)\\.\\.\\N_{start,t}(x,y)\\.\\.\\N_{s,s'}(x,y)\\.\\.\\N_{s,end}(x,y)\\.\\.\\    \end{matrix}\right] = \omega \cdot \phi(x,y)
$$

其中：

$w$：为待求的参数矩阵。为发射概率和转移概率的矩阵。

$\phi(x,y)$：为count得到的值。对应的是某一个发射概率，或者某一个转移概率在样本中出现的次数。



由此可得，
$$
P(x,y) = exp(\omega \cdot \phi(x,y))
$$
但是其实他不是相等的，是正相关的，也就是让$P(x,y)$取得最大值的y和$exp(\omega \cdot \phi(x,y))$取得最大值的y是同一个y。
$$
P(x,y) \varpropto exp(\omega \cdot \phi(x,y))
$$

### CRF 训练

$(x^n, y^n)​$是training data，$x^n​$是给出的词，$\hat{y}^n​$是预测正确的词性，$y'​$是预测的错误的词性。

#### 目标函数

$$
O(w) = \sum_{n=1}^N logP(\hat{y}^n|x^n)
$$

其中：

$x^n$：给定的观察到的序列.

$\hat{y}^n$：正确的隐藏状态。

也就是去求一个$w$使得：
$$
\arg\max_{w} O(w)
$$
解释：

> 任务是最大化目标函数。也就是对于一个序列，在$x^n$发生的情况下，使得$\hat{y}^n$最大的$w$矩阵。

##### 第一阶段：贝叶斯

对于$ logP(\hat{y}^n|x^n)$:

由于：
$$
P(y|x) =\frac{P(x, y)}{\sum_{y'}P(x,  y')}
$$
则：
$$
logP(\hat{y}^n|x^n) \\=log \frac{P(x^n, \hat{y}^n)}{P(x^n)} \\=log \frac{P(x^n, \hat{y}^n)}{\sum_{y'} P(x^n, y')}\\=logP(x^n, \hat{y}^n) - log\sum_{y'}P(x^n, y')
$$
物理意义：

> 通过以上可以知道，使得$O(w)​$最大的物理意义，就是让预测正确的概率越大越好，让其他的预测错误的概率越小越好。

$$
O(w) = \sum_{n=1}^{N}logP(\hat{y}^n|x^n) = \sum_{n=1}^{N}O^n(w)
$$

而：

$$
O^n(w) = logP(\hat{y}^n |x^n)=logP(x^n, y^n) - log\sum_{y'}P(x^n, y')
$$


##### 第二阶段：求偏导

对$O^n(w)​$求偏导：

$$
\frac{\partial{O^n(w)}}{\partial{w_{s,t}}} \\=\frac{\partial{(logP(\hat{y}^n |x^n))}}{\partial{w_{s,t}}}  \\= \frac{\partial{logP(x^n, \hat{y}^n)}}{\partial{w_{s,t}}} - \frac{\partial{(log\sum_y'P(x^n, y'))}}{\partial{w_(s,t)}}
$$
分别求偏导：

###### 偏导一

$$
P(x^n , y^n) = exp(w \phi(x^n, y^n))
$$

对于$w\phi(x^n, y^n)​$:

$$
w\phi(x^n, y^n) = \sum_{s,t}w_{s,t} \cdot N_{s,t}(x^n, \hat{y}^n) + \sum_{s,s'}w_{s,s'} \cdot N_{s,s'}(x^n, \hat{y}^n)
$$
则：

$$
\frac{\partial{logP(x^n, y^n)}}{\partial{w_{s,t}}} \\= \frac{\partial{(w\phi(x^n, y^n))}}{\partial{w_{s,t}}} \\= N_{s,t}(x^n, \hat{y}^n)
$$


###### 偏导二

$$
P(x^n , y') = exp(w \phi(x^n, y'))
$$

令：

$$
Z(x^n) = \sum_{y'}exp(w \cdot \phi(x^n, y')) = \sum_{y'}exp(\sum_{s,t}w_{s,t} \cdot N_{s,t}(x^n, y') + \sum_{s,s'}w_{s,s'} \cdot N_{s,s'}(x^n, y'))=\sum_{y'}P(x^n, y')
$$
则：

$$
\frac{\partial{(log\sum_{y'}P(x^n, y'))}}{\partial{w_(s,t)}} \\=\frac{\partial(logZ(x^n))}{\partial w_{s,t}} \\= \frac{1}{Z(x^n)} \cdot  \frac{\partial{Z(x^n)}}{\partial{w_{s,t}}}
$$
而：
$$
\frac{\partial{Z(x^n)}}{\partial{w_{s,t}}} \\= Z(x^n) \cdot N_{s,t}(x^n, y') \\=\sum_{y'}P(x^n, y') \cdot N_{s,t}(x^n, y')  \\= \sum_{y'}\frac{P(x^n, y')}{1}\cdot N_{s,t}(x^n, y') \\= \sum_{y'}\frac{P(x^n, y')}{P(x^n)}\cdot N_{s,t}(x^n, y')  \\= \sum_{y'}P(y'|x^n)\cdot N_{s,t}(x^n, y')
$$
这里是对指数函数求导数，和前面的不太一样。

这里$P(x^n) = 1$，其实是考虑到所有的$x^n$都是给定的，忽略不会有什么影响。

##### 第三阶段：合并

$$
\frac{\partial{O^n(w)}}{\partial{w_{s,t}}} \\=\frac{\partial{(logP(\hat{y}^n |x^n))}}{\partial{w_{s,t}}}  \\= \frac{\partial{logP(x^n, \hat{y}^n)}}{\partial{w_{s,t}}} - \frac{\partial{(log\sum_y'P(x^n, y'))}}{\partial{w_(s,t)}} \\=N_{s,t}(x^n, \hat{y}^n) -  \sum_{y'}P(y'|x^n)\cdot N_{s,t}(x^n, y')
$$

前面的一项不用说，后面的一项是可以用维比特算法来解的。

物理意义

> 当前，预测正确的，词性和词 共同出现的词对。减去。所有预测不对的概率，再乘以预测不对的共同出现的词对。

### CRF预测

推导过程如下：
$$
y = \mathop{\arg\max _{y\epsilon Y}P(y|x)} = \mathop{\arg\max_{y\epsilon Y}}P(x,y) = \mathop{\arg\max_{y\epsilon Y} \omega \cdot \phi(x,y)}
$$
训练过程，基于样本$\omega$矩阵是已经得到的了，那么使用维比特算法，求一个动态规划问题，全局最优解就可以了。

## HMM和CRF的比较

HMM做的事情是最大化$P(x,\hat{y})$。

CRF不进最大化$P(x, \hat{y})$，还最小化$P(x,y')$，所以CRF要比HMM更有可能得到最优的结果。

## RNN和CRF的比较

### RNN

- Deep

### CRF

- 如果采用的是单方向RNN，每一个时间步，得到的输出，其实不是看完整个sequence得到的，是基于之前的信息得到的。这一点CRF求得是全局最优解。

- 可以对label进行强干预。比方说，动词后面不能接动词，我是可以直接去修改转移概率矩阵，使得在维比特求解的时候，动词后面接动词的路径都不走。而RNN很难做到这一点，RNN也不是不可以学到这些信息，但是需要更多的数据，需要更多的训练资源。
- RNN中的cost和error不一定是很直接相关的。你可能训练了一个loss非常小的模型，但是你考核error的方式和loss的方式不是一样的，导致loss可能很小，但是error不一定很小。但是CRF的cost和error是直接相关的。

## RNN和CRF的结合

HMM和CRF中：
$$
P(x,y) = P(y_1|start)\prod_{l=1}^{L-1}P(y_{l+1}|y_l)P(end|y_L)\prod_{l=1}^{L}P(x_l|y_l)
$$
我们把$P(x_l|y_l)$用RNN每一步$P(y_l|x_l)$的输出来代替。

![](https://markdocpicture.oss-cn-hangzhou.aliyuncs.com/iPic/2019-03-28-025318.png)

但是RNN的概率分布输出，到CRF的发射概率，需要有一个转化的过程：
$$
P(x_l|y_l) = \frac{P(x_l,y_l)}{P(y_l)} = \frac{P(y_l|x_l)P(x_l)}{P(y_l)} = \frac{P(y_l|x_l)}{P(y_l)}
$$
其中：

$P(y_l|x_l)$ ： 是RNN的输出。

$P(y_l)$ : 样本中某一个label出现的几率。count可得。

$P(x_l)$ : $x_l$是给定的，忽略不影响结果，大部分框架都是如此实现。