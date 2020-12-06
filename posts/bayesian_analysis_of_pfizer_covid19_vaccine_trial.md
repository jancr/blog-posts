# A Bayesian (re)Analysis of Pfizers COVID19 vaccine data.

This post is inspired by [this post at
r-bloggers.com](https://www.r-bloggers.com/2020/11/a-look-at-biontech-pfizers-bayesian-analysis-of-their-covid-19-vaccine-trial/)
They did the grunt work and reverse engineered [the study
plan](https://pfe-pfizercom-d8-prod.s3.amazonaws.com/2020-09/C4591001_Clinical_Protocol.pdf#page=102)

It uses data from the [press
release](https://www.pfizer.com/news/press-release/press-release-detail/pfizer-and-biontech-conclude-phase-3-study-covid-19-vaccine)
which can be summarized as follows:

 * They use a Beta prior with $a=0.700102$ and $b=1$
 * They use a binomial likelihood:
    - There were 162 COVID19 cases in the control group
    - There were 8 COVID19 in the vaccine group
    - thus $z=8$ and $N=162+8=170$
 * The trial was considered a success if the $2.5\%$ lower of end of the $VE$
   (Vaccine Efficacy) interval exclude 30%.

If you are unfamiliar with the dichotomous distributions (Beta, Bernoulli and
Binomial), then you should read [this](../../dichotomous-distributions) before
continuing, or wing it

It is obvious when there are 8 in one group and 162 in the other group, then the
vaccine is very effective, so while this analysis may seem overkill to conclude
something obvious, I can assure you that what Phizer has send to the FDA is
much longer!, so let's get started.

## Step 0, what do we want to model?

**$VE$ - what we care about**

If we define $\pi_v$ as the probability that a person in the vaccine group
falls ill, and $\pi_c$ as the probability that a person in the control group
falls ill, then $VE$, the vaccination efficacy is defined as:

$$
VE=1 - \frac{\pi_v}{\pi_c}
$$

Lets plug in numbers to help out intuition, let's assume that $\pi_v=4\pi_c$,
for every 4 sick in the control there is 1 sick in the vaccine group, then
$VE=1-\frac{1}{4}=0.75$ thus the Vaccine efficacy is 75%, meaning $\frac{3}{4}$ where
prevented and $\frac{1}{4}$ were not.

**$\theta$ - What we model and measure**

In their study plan they define $\theta$ as follows:

$$
\theta = \frac{1 - VE}{2 - VE}
$$

While this seems like black magic to me, let's try to plug in the definition
for $VE$ into the formula for $\theta$:

<!-- $$ -->
<!-- \begin{aligned} -->
<!--     \theta &= \frac{1 - VE}{2 - VE} \\ -->
<!--     \theta &= \frac{1 - (1 - \frac{\pi_v}{pi_c})} -->
<!--                    {2 - (1 - \frac{\pi_v}{pi_c})} \\ -->
<!--     \theta &= \frac{\frac{\pi_v}{pi_c}} -->
<!--                    {(1 + \frac{\pi_v}{pi_c})} \\ -->
<!--     (1 + \frac{\pi_v}{\pi_c})\theta &= \frac{\pi_v}{pi_c} \\ -->
<!--     (\pi_c + \pi_v)\theta &= \pi_v \\ -->
<!--     \theta &= \frac{\pi_v}{\pi_c + \pi_v} -->
<!-- \end{aligned} -->
<!-- $$ -->
$$
\begin{aligned}
    \theta &= \frac{1 - VE}{2 - VE} \\
    \theta &= \frac{1 - (1 - \frac{\pi_v}{\pi_c})}
                   {2 - (1 - \frac{\pi_v}{\pi_c})} \\
    \theta &= \frac{\frac{\pi_v}{\pi_c}}
                   {(1 + \frac{\pi_v}{\pi_c})} \\
    \theta &= \frac{\frac{\pi_v}{\pi_c}}
                   {\frac{\pi_c+\pi_v}{\pi_c}} \\
    \theta &= \frac{\pi_v}{\pi_c + \pi_v}
\end{aligned}
$$

So $\theta$ is the proportion of people in the vaccine group who got sick,
if $\theta=0.3$ then $30\%$ of the sick were in the vaccine group.

Since $\theta$ is a probability of a binary outcome, we can use distributions
such as $Beta$, $Bernuli$ and $Binomal$ to model it's uncertainly, and then
transform it (and it's uncertainly) back to the $VE$.

There are 3 steps to a Bayesian model, step 1 is defining a prior $p(\theta)$,
which is what we believe before seeing the data, step 2 is defining a
likelihood function ($p(D|\theta)$), which is what we believe about the data
($D$) given $\theta$. 3. We will use Bayes Theorem to get a posterior
$p(\theta|D)$, which is what we believe about $theta$ given the data.

We can use the formula $\theta=\frac{1-VE}{2-VE}$ to convert $VE$ to $\theta$, but
we need to rearrange the formula to be able to convert the $\theta$ to $VE$,
which is how we want to report the result.

$$
\begin{aligned}
    \theta &= \frac{1 - VE}{2 - VE} \\
    \theta(2 - VE) &= 1 - VE \\
    2\theta - VE\theta + VE &= 1 \\
    VE(1 - \theta{}) &= 1 - 2\theta \\
    VE &= \frac{1 - 2\theta}{1 - \theta}
\end{aligned}
$$

## (Optional) Code to help us visualize

The code will not be explained, but what they do is this: 
 <!-- * `norm_grid`: normalizes a grid of samples so they sum to 1 like 'true' -->
 <!--   probability density functions.  -->
 * `bayesian_update` takes a prior and likelihood, plots them, and also plots
   the (unnormalized) posterior via via grid approximation (more how that works
   later).
<!--    3 vectors `theta` (the grid) such as $\theta=[0.0, -->
<!-- 0.01, 0.02.., 1]$ and `prior` and a `likelihood` which are vectors evaluated at -->
<!-- all the $\theta$ points. It then plots the `prior`, the `likelihood` and the -->
<!-- unnormalized posterior.  -->
 * `plot_dist` plots a distribution both on the scale of $\theta$ and on the
   transformed scale of $VE$

<!-- def norm_grid(theta, likelihood): -->
<!--     norm_factor = ((likelihood[1:] + likelihood[:-1]) / 2).sum() / (theta.shape[0] - 1) -->
<!--     return likelihood / norm_factor  -->

```python
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def bayesian_update(theta, prior, likelihood):
    fig = plt.figure(figsize=(21, 7))
    posterior = prior * likelihood
    _iter = ((prior, "Prior"), (likelihood, "likelihood"), (posterior, "Posterior"))
    for i, (dist, dist_name) in enumerate(_iter, 1):
        ax = fig.add_subplot(1, 3, i, xlabel=r"$\theta$", ylabel='density', 
                             title=dist_name)
        ax.fill_between(theta, dist)
    return fig
    

def plot_dist(theta, likelihood, title=""):
    fig = plt.figure(figsize=(14, 7))
    ax_theta = fig.add_subplot(1, 2, 1, xlabel=r"$\theta$", ylabel='density', 
                               title=title) 
    ax_ve = fig.add_subplot(1, 2, 2, xlabel="VE (Vaccine efficacy)",
                            ylabel='density', title=title) 
    
    ax_theta.fill_between(theta, likelihood)
    ve = (1 - 2 * theta) / (1 - theta)
    ax_ve.fill_between(ve[ve>0], likelihood[ve>0])
    return fig
```

## Step 1, the prior

The prior that Phizer choose were $Beta(0.700102, 1)$, While this may seem
weirdly specific, let's remember that if $VE$ is less than $0.3$, then the
vaccine is considered ineffective, which is a 'high' bar compared to 'better
than no effect' which is unfortunately common in the NHST framework. Let's
calculate the mean of this prior, and transform it to $VE$, this may give us a
hit to the choice:

$$
\begin{aligned}
E[p(\theta)] &= \frac{\alpha}{\alpha+\beta} \\
             &= \frac{0.700102}{0.700102+1} = 0.4118 \\
E[VE] &= \frac{1 - 0.4118}{2 - 0.4118} = 0.30
\end{aligned}
$$

So they assume that vaccines which are good enough to enter a phase 3 trial will
on average have a $VE$ of $30\%$, the sample size of the prior is
$\alpha+\beta$ which is 1.7, so we should expect the prior to be very broad.


Let's plot it, we use the `plot_dist` function which plots the distribution of
$\theta$ to the left, we uses the formula $aE = \frac{1 - 2\theta}{1 -
\theta}$ to convert $\theta$ to $VE$ which is plot on the right.

```python
alpha, beta = 0.700102, 1
theta = np.linspace(0, 1, 101)
prior = sp.stats.beta(alpha, beta).pdf(theta)
fig = plot_dist(theta, prior, f"Beta({alpha}, {beta})")
```

![svg](pfizer_vaccine/prior_dist.svg)

And it is very broad given credibility to all values of $\theta$ between 0 and
1, but skews towards 0.

## Step 2, the likelihood

### Side quest - Bayesian Updating
Before we calculate the likelihood of the data, let's try to see how the prior
changes if 1 person from each group fell ill. We use the Bernoulli likelihood
to simulate a 'draw', the Bernoulli likelihood function is a triangle, because 
$p(1\mid\theta)=\theta$ (hint: like $y=x$) and $p(0\mid\theta)=1-\theta$ (hint:
$y=1-x$)

Here is the plot if we had only 1 data point and it was 0 (from the control),
on the left we have the prior (as above), in the middle we have the triangle
shaped Bernoulli likelihood, and on the right we have the unnormalized
posterior:

```python
bayesian_update(theta, prior, sp.stats.bernoulli.pmf(0, theta))
```

![svg](pfizer_vaccine/post0_dist.svg)

If it looks like the left and middle was multiplied together and
produced the right one, then that's because that's exactly what happened, why?

Here we have Bayes theorem with Bernoulli likelihood:

$$
p(\theta|z) = \frac{p(z\mid\theta)p(\theta)}{p(z)}
$$

Notice that the denominator does not depend on $\theta$ and can therefor be
viewed as a (very hard to compute) normalizing constant. If we remove it, we
screw up the scale on the y-axis, but the shape remains, so the unnormalized
posterior (which is not a proper pdf because it does not sum to 1) is:

$$
p(\theta|z) \propto p(z\mid\theta)p(\theta)
$$

Which is exactly how we created the right figure, let's try to see the shape of
the posterior if a vaccinated person had fallen ill:

```python
bayesian_update(theta, prior, sp.stats.bernoulli.pmf(1, theta))
```

![svg](pfizer_vaccine/post1_dist.svg)

If this was all the data we had, then because we have a very weak prior, we
would end up believing that the vaccine most likely does not work, though the
distribution is very broad, so we are not very confident in this assertion.

### likelihood of the actual data

<!-- Furthermore because denominator of Bayes Theorem does not contain $\theta$ it -->
<!-- can be taught of as a normalization constant: -->
<!--  -->
We assume that data is binomial distributed, with parameters $z=8$, $N=170$,
and $p=\theta$. We can then plot how likely this data is for different values of
$\theta$, $p(z,N\mid{}\theta)$, is the likelihood of the data ($z$ and $N$)
given $\theta$, the data is the data, so we are not interested in how $z$ and
$N$ could have been otherwise, we are interested in how likely they are given
different values of $\theta$, so even though we are conditioning on $\theta$
the likelihood is still a function of $\theta$. Let's try to plot the
likelihood for different values of $\theta$ and $VE$:

```
z, N = 8, 170
likelihood = sp.stats.binom.pmf(z, N, theta)
fig = plot_dist(theta, likelihood, f"$p(N={N},z={z}\\mid\\theta)$")
```
<!-- likelihood = norm_grid(theta, likelihood) -->

![svg](pfizer_vaccine/likelihood.svg)

In statistics, the uncertainly of a distribution is tied to the sample size, 
Because the likelihood has a sample size of 170 where the prior has one of 1.7,
the likelihood is much narrower than the prior.

## Step 3, the posterior

There are a few ways to get a posterior from a prior and a likelihood, 3 popular methods include:

1. Use sampling.
   * This is what modern Bayesians does, unless they have very simple models like
     this one.
2. Use grid approximation, as we did with the Bernoulli example.
   * This is excellent for teaching purposes, because it's intuitive, but in
     practice this only works for models with very few parameters,
3. Exploit the fact that the priors are conjugated, this we will also do.
   * This is super cool, and easy when you know the solution, but only works for
     small models and a small subset of prior-likelihood combinations.

### Getting the posterior via grid approximation
As previous we simply multiply the prior grid vector with the likelihood grid vector:

```python
bayesian_update(theta, prior, likelihood)
```

![svg](pfizer_vaccine/posterior_dist.svg)

Because there is much less certainly in the likelihood estimate, than the
prior, the posterior is dominated by the likelihood, to the point where it is
almost indistinguishable from the prior.

<!-- Grid approximation is simply the act of drawing a grid of parameters, such as  -->
<!-- 101 $\theta$ samples from 0 to 1 in increments of 0.01, and then evaluate prior -->
<!-- $p(\theta)$ and the likelihood $p(N,z\mid\theta)$ at all these points. The -->
<!-- product of these two 'likelihood vectors' would then have the shape of the -->
<!-- posterior because: -->
<!--  -->
<!-- $$ -->
<!-- p(\theta\mid{}N,z)=\frac{p(N,z\mid\theta)p(\theta)}{p(N,z)} \\ -->
<!-- p(\theta\mid{}N,z)=\frac{p(N,z\mid\theta)p(\theta)}{p(N,z)} -->
<!-- $$ -->

<!-- Since $p(N,z)$ does not depend on $\theta$ it can be viewed as the -->
<!-- normalization constant that makes the sum of the two vectors sum to 1, why 1?, -->
<!-- because the definition of a probability density function (which is what the -->
<!-- posterior is) is a function who's integral is 1. In any case, $p(N,z)$ is a -->
<!-- constant, so if we plot $p(N,z\mid\theta)p(\theta)$ then it will have the same -->
<!-- shape as the real posterior. -->

### Getting the posterior via conjugation
**TLDR:** The posterior of the Binomial likelihood $p(N,z\mid\theta)$ with Beta
prior $p(\alpha,\beta)$ is this Beta distribution $p(z+\alpha,N-z+\beta)$

Here we will use the 'conjugation trick' to analytically derive the posterior
for our Beta-Binomial model.

We start with Bayes Theorem:

$$
p(\theta|z,N) = \frac{p(z,N\mid\theta)p(\theta)}{p(z,N)} \\
$$

Then we plug in the definition for the Binomial likelihood and Beta prior:

$$
p(\theta|z,N) = {N \choose k}\theta^z(1-\theta)^{N-z} % likelihood
                 \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)} % prior
$$

That's ugly, let's rearrange. So we have the $\theta$ stuff in the numerator,
and the data stuff in the denominator

$$
p(\theta|z,N) = \frac{\theta^z(1-\theta)^{N-z}\theta^{\alpha-1}(1-\theta)^{\beta-1}} % \theta
                     {B(\alpha,\beta){{N \choose k}}^{-1}} % data
$$

Let's collect the powers in the numerator

$$
p(\theta|z,N) = \frac{\theta^{z+\alpha-1}(1-\theta)^{N-z+\beta-1}} % \theta
                     {B(\alpha,\beta){{N \choose k}}^{-1}} % data
$$

Here comes the conjugation shenanigans. If you squint, the top of the distribution looks like
the top of a Beta distribution:

$$
\begin{aligned}
\alpha'&= z+\alpha \\
\beta'&= N-z+\beta \\
p(\theta|z,N) &= \frac{\theta^{\alpha'-1}(1-\theta)^{\beta'-1}} % \theta
                     {B(\alpha,\beta){{N \choose k}}^{-1}} % data
\end{aligned}
$$

Let's continue the shenanigans, since the numerator looks like the numerator of
a beta distribution, we know that it would be a proper beta
distribution if we changed the denominator like this:

$$
\begin{aligned}
p(\theta|z,N) &= \frac{\theta^{\alpha'-1}(1-\theta)^{\beta'-1}} % \theta
                     {B(\alpha',\beta')} \\ % data
p(\theta|z,N) &= \frac{\theta^{z+\alpha-1}(1-\theta)^{N-z+\beta-1}} % \theta
                     {B(z+\alpha,N-z+\beta)} % data 
\end{aligned}
$$

And there we have it, a normalized posterior of $\theta$, let's plot it to make
sure it looks like when we got via grid search:

```python
posterior = sp.stats.beta(z + alpha, N - z + beta)
fig = plot_dist(theta, posterior.pdf(theta))
```

![svg](pfizer_vaccine/posterior.svg)

It looks like the grid one, so we did not mess up!

Now we can finally ask the original question, does the $97.5\%$ lower interval
exclude $VE=30\%$, of course you can use your eyes, and see that the transformed
posterior over $VE$ is fare away from $0.3$, what we want is the $\theta$
corresponding to a CDF of 97.5%, the PPF function is the inverse of the CDF, so
let's use it 

```python
theta = posterior.ppf(0.975)
``` 

Now we have $\theta = 0.088$, we can transform that with the formular we derived in the beginning

$$
VE_{2.5\%} = \frac{1 - 2\theta}{1 - \theta} = 0.90
$$

So there is a 97.5\% chance that $VE$ is at least $0.9$

What's the chance that it's actually worse than $30\%$, remember that
corresponded to $\theta=0.4118$, well that's the area between $0.4118$ and 1,
the survival function is defined as 1-cdf, so let's evaluate it at $0.4118$

```python
posterior.sf(0.4118)
```

Which returns $1.1\times10^{-16}$, so we are very sure that $VE>0.30$






