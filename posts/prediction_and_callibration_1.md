# Scott Alexander 2019 Predictions - Part 1

Scott Alexander is a darling of the Bayesian rationalist community, he has a lot
more epistemic humility than most, despite being an impressively well
calibrated predictor.

In this series we will try to achieve 2 things:

1. (this post) Understand what a likelihood function is, and use it to evaluate predictions
2. (next post) Make a Bayesian calibration model

## The likelihood function

Let's first look at Bayes Theorem

$$
p(\theta\mid{}y)= \frac{p(y\mid{}\theta)p(\theta)}{p(y)}
$$

In common parlance, the 4 parts of Bayes Theorem are called:

$$
posterior = \frac{likleyhood\times{}prior}{data}
$$

What we want is our posterior, the probability of some model parameters (often
$\theta$) given some data ($y$). We construct a model through two things, a
prior function which describes what we believe before seeing the data, and a
likelihood function ($p(y\mid\theta)$) which given a model ($\theta$, drawn from the
prior) scores the data.

The simplest and most relevant likelihood function is the Bernoulli

$$
p(y|\theta) = \theta^{y}(1 - \theta)^{1-y}
$$

Where $y$ is our data 1 when correct 0 otherwise, and $\theta$ is our model, as
we don't have a model yet, our model is 'what Scott predicted'

Notice, that while the Bernoulli distribution may look scary, if we take a
prediction of $\theta=0.6$ then if True ($y=1$), then the Bernoulli distribution
says that this is 0.6:

$$
\begin{aligned}
p(y=1|\theta=0.6) &= \theta^{y}(1 - \theta)^{1-y} \\
				  & = 0.6^1(1 - 0.6)^{1-1} = 0.6^1\times{}0.4^0=0.6
\end{aligned}
$$

And if the prediction turned out wrong ($y=0$) Then:

$$
\begin{aligned}
p(y=0|\theta=0.6) &= \theta^{y}(1 - \theta)^{1-y} \\
				  & = 0.6^0(1 - 0.6)^{1-0} = 0.6^0\times{}0.4^1=0.4
\end{aligned}
$$

The Bernoulli distribution says that there was a 40% chance you were wrong.
Which is the same a predicting not $\theta$ with 40%

If a person makes 3 predictions $\theta = [0.6, 0.6, 0.7]$ and the outcomes
were $Y = [1, 0, 1]$, then the likelihood of all 3 observations is simply the
product of the 3 Bernoulli likelihoods:

$$
\begin{aligned}
P(\theta|Y) &= \prod_{i=1}^{3} p(\theta_i|y_i) \\
			&= 0.6\times(1 - 0.6)\times{}0.7 = 0.168
\end{aligned}
$$

The higher this number, the more likely your predictions are. It can be useful
to divide the null predictor to compare against random performance:

$$
p(\theta=0.5|Y) = \prod_{i=1}^{N} p(\theta_i|y_i) = 0.5^N
$$

So the likelihood of 3 above predictions are: $\frac{0.168}{0.5^3}\approx{}1.34$
times more likely than random. Making this person slightly better than random


### How good a predictor is Scott

Because Scott has made a lot of prediction, and because we will later implement
a 'calibration' model of Scott, let's try to compare the likelihood of his 2019
predictions the null/random model which predicts everything with 50% (which
implicitly mean that it also predict it doesn't happen with 50%)

First we import `scipy` the scientific python library

```python
import scipy as sp
import scipy.stats
```

Then we code Scott Alexanders prediction data coded as \[Guess, Outcome\].

Because outcomes is what we want to "predict" we put that in the $y$ variable,
and put guess in the predictor variable $x$


```python
data = np.array((
    [[0.5, 1]] *  7 + [[0.5, 0]] * 4 +
    [[0.6, 1]] * 15 + [[0.6, 0]] * 7 +
    [[0.7, 1]] * 12 + [[0.7, 0]] * 5 +
    [[0.8, 1]] * 31 + [[0.8, 0]] * 6 +
    [[0.9, 1]] * 16 + [[0.9, 0]] * 1 +
    [[0.95, 1]] * 5 + [[0.95, 0]] * 0
))
y = data[:, 1]
X = data[:, 0]
```

The fictive person who made 3 predictions and got 2 correct were slightly
better than random, how much better than random is Scott?, let's take the
product of all his predictions


```python

scott_likelihood = sp.stats.bernoulli(X).pmf(y).prod()
random_predictor = 0.5 ** len(y)
f"{scott_likelihood / random_predictor:g}"
```

	'7.4624e+09'

So a 7 billion times more likely, there are two reasons why this number is so
large, 1) Scott made a lot of predictions, 2) Scott is a very good predictor,
now it is easy to become a better predictor than Scott, simply pick easy things
to predict. What is very hard it to be as well calibrated as Scott, but first
let's define our term:

<!-- Also notice that because $p(\theta=0.5|y=0)=p(\theta=0.5|y=1)=0.5$ The debate -->
<!-- of whether to include his 50:50 predictions is mute, as his likelihood ratio -->
<!-- compared to the null model remains unchanged. -->

**Predictor:**

 * A good predictor is a person who predict better than random:
	 - $\prod P(\theta|y) >> 0.5^N$
 * A bad predictor is a person who predicts close to random:
	 - $\prod P(\theta|y) \approx 0.5^N$
 * A terrible predictor is one who are worse than random:
	 - $\prod P(\theta|y) < 0.5^N$

It may be hard to understand how you can be worse than random, and that
of course takes skill, but if Scott had flipped all his guesses, his likelihood
ratio would be $\frac{1}{7\times{}10^9}$ which is much less than 1

<!-- Prediction is very task dependent and is therefore only comparable if the -->
<!-- prediction is on the same data, two set of predictions can be compared by comparing -->
<!-- the product of their Bernoulli likelihood function. -->

Now that we all agree that Scott is a good predictor, we can finally introduce
what I want to talk about, how well calibrated is Scott, and how do we measure
it?

**Calibrated:**

 * A well calibrated predictor is a person where the predictions match the
   outcome frequency.
 
**Example**
* Person A predict 100 things with 60% confidence, 61 of them turns out to
  occur, because $\frac{61}{100} \approx 0.6$ this person is very well
  calibrated.
* Person B predict 100 things with 80% confidence, 67 of them turns out to
  occur, because $\frac{67}{100} \ne 0.8$ this person is not very well
  calibrated

Because 67 > 61 Person B may be the better predictor, even though they're not
as well calibrated. Let's evaluate the likelihood of their claims:

Person A's prediction is equivalent to 61 'correct' 60% predictions and 39
'correct' 40% predictions, yielding the following likelihood:

$0.6^{61}\times{}0.4^{39} \approx 8.86\times{}10^{-30}$

Person A's prediction is equivalent to 69 'correct' 80% predictions and 21
'correct' 20% predictions, yielding the following likelihood

$0.8^{67}\times{}0.4^{33} \approx 2.76\times{}10^{-30}$

Because $8.86\times{}10^{-30} > 2.76\times{}10^{-30}$ Person A is also a better
predictor than person B, to get an intuition of why, let's consider person C:

Person C predict 100 things with 100% confidence, 99 of them turns out to
occur, person C a very bad and miss calibrated prediction, because something impossible
happened! This is also reflected in the likelihood of his predictions, which is
zero:

$1^{99}\times{}0^1=0$

**Summary so fare**

So we can improve our predictions likelihood by being very knowledgeable as Person B or well
calibrated as person A. Now it is of course much harder to achieve omniscience
than epistemic humility, which is why the Bayesian rationalist community and
the rest of this post will focus on the calibration part.


<!-- **I want to build a Golem, C'mon let's go and wreck Prage - Anna, famous model builder** -->

How good a predictor you are can be evaluated by the product of your likelihood
function, is there a better way to evaluate this, yes, make a model!

How do you find out how well calibrated you are?, again we make a model, which
is what we will explore in the rest of this post

