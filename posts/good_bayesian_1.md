# Good Bayesian - Part 1 - Probability 101

<!-- Bayesian Statistics Has become quite popular, in the following few posts I will -->
<!-- try to show how to  -->

The following short series is inspired by two papers by Kruschke and Liddell: 

1. [Bayesian data analysis for newcomers](https://link.springer.com/article/10.3758/s13423-017-1272-1)
2. [The Bayesian New Statistics: Hypothesis testing, estimation, meta-analysis, and power analysis from a Bayesian perspective](https://link.springer.com/article/10.3758/s13423-016-1221-4)

Bayesian inference in a nutshell can be boiled down to 3 things:
<!-- of which I plan to cover the first 3 in this series. -->

1. Rules of probability.
2. Priors matter
3. Think in distribution not point estimates
<!-- 4. (The generative process: how was my data generated) -->

## The Data
My wife has taken an COVID19 antibody test.
The test measures two antibodies, but for simplicity we will only look at the
most predictive antibody IgG. The Sensitivity of IgM is only 69% and
I suspect that the IgM and IgG test results are correlated, thus we
can't do simple Bayesian updating :).

According to **[Hoffman et al.
](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7178815/)** the performance
of the test is reported as follows: 

|              | Cases | Healthy | Total |
|--------------|-------|---------|-------|
| IgG positive | 27    | 1       | 28    |
| IgG negative | 2     | 123     | 125   |
| Total        | 29    | 124     | 153   |

## Rules of Inference
If we divide all the cells of the above table by the total (153), then we can turn the table of
counts into a table of probabilities.

|                             | Cases ($P(\theta)$)            | Healthy ($P(\neg\theta)$)       | Total  |
|-----------------------------|--------------------------------|---------------------------------|-------------------------------------|
| IgG positive ($P(D)$)       | $\frac{27}{153}\approx{}0.176$ | $\frac{1}{153}\approx{}0.007$   | $\frac{28}{153}\approx{}0.183$      |
| IgG negative ($P(\neg{}D)$) | $\frac{2}{153}\approx{}0.013$  | $\frac{123}{153}\approx{}0.804$ | $\frac{125}{153}\approx{}0.817$     |
| Total                       | $\frac{29}{153}\approx{}0.190$ | $\frac{124}{153}\approx{}0.810$ | $\frac{153}{153}=1$                  |


### Joint Probability
In the above table we have 4 joined probabilities, which are the probabilities
of two joined (simultaneous) events, such as IgG positive ($D$) **and** COVID19
positive ($\theta$). This will be written $P(\theta,D)$ and should be read as
"the (joined) probability of $\theta$ and $D$"
<!-- If we use $\theta$ to denote COVID19 positive and $D$ for a positive -->
<!-- test, then $P(\theta,D)$ is the (joined) probability of both occuring.   -->

### Marginal Probability
In the margins of the table (the Total row/colums) are the marginal
probabilities, where one of the variables ($\theta$ or $D$) has been
marginalized out. For dichotomous variables (variables of two outcomes) this is
easily done by adding them together. For continues variable, we need to resort
to integration. For example

$$
P(D) = \sum_{\theta}(P(\theta, D)) = P(D,\theta) + P(D,\neg\theta)
	 = \frac{27 + 1}{153} = \frac{28}{153}
$$

### Conditional Probability
A conditional probability is a probability where one thing is "given" (taken as
true). They can be derived from a joined and marginal probability as follows:

<!-- TODO: figure out how to align and how to force newline -->
$$
P(D\mid{}\theta) = \frac{P(\theta,D)}{P(\theta)}= \frac{\frac{27}{153}}{\frac{29}{153}}=\frac{27}{29}\approx{}0.931
$$
$$
P(\neg{}D\mid{}\neg\theta) = \frac{P(\neg\theta,\neg{}D)}{P(\neg\theta)}= \frac{\frac{123}{153}}{\frac{124}{153}}=\frac{123}{124}\approx{}0.992
$$

## Evaluate the Data

The two conditional probabilities above have special names:

* $P(D\mid{}\theta)$ read as "the probability of D given theta", is the tests
  sensitivity, and is the probability of getting a positive test if you have
  had COVID19. 
* $P(\neg{}D\mid{}\neg\theta)$ is the specificity of the test, the probability of a
  negative test if you have not had COVID19.

There are also two other conditional probabilities which can be derived from
the table:

* $P(\theta\mid{}D)=\frac{27}{28}$: The positive predictive value 
* $P(\neg\theta\mid{}\neg{}D)=\frac{123}{125}$: The negative predictive value

While at first glance these may seem more useful, because it is tempting to
interpret it as the probability of COVID19 if I have a positive test. The
correct interpretation is "The probability of COVID19 if my prior probability
of COVID19 is the one the one who is congruent with
$P(D)=\frac{28}{153}\approx{}0.183$

To make this more concrete. Imagine I go out and redo the experiment, but
instead of recruiting 124 healthy and 29 survivors, I recruit 124 healthy and 58
survivors. Then the table I would expect to receive (ignoring random variation) would be one with $\times{}2$ in the "Cases" column as follows: 

|              | Cases | Healthy | Total |
|--------------|-------|---------|-------|
| IgG positive | 54    | 1       | 55    |
| IgG negative | 4     | 123     | 127   |
| Total        | 58    | 124     | 182   |

Now all the joined probabilities have changed, but how does the 4 conditional probabilities pan out? (stop and think)


$$
P(D\mid{}\theta) = \frac{P(\theta,D)}{P(\theta)}
            = \frac{\frac{54}{182}}{\frac{58}{182}}
            = \frac{54}{58} = \underline{\underline{\frac{27}{29}}}
$$
$$
P(\neg{}D\mid{}\neg\theta) = \frac{P(\neg\theta,\neg{}D)}{P(\neg\theta)}
                      = \frac{\frac{123}{182}}{\frac{124}{182}}
                      = \underline{\underline{\frac{123}{124}}}
$$

Sensitive and Specificity are unchanged, which is why we use these to evaluate tests.

$$
P(D\mid{}\theta) = \frac{P(\theta,D)}{P(D)}
            = \frac{\frac{54}{182}}{\frac{55}{182}}
            = \frac{54}{55}\ne\frac{27}{28}
$$
$$
P(\neg{}D\mid{}\neg\theta) = \frac{P(\neg\theta,\neg{}D)}{P(\neg\theta)}
                      = \frac{\frac{123}{182}}{\frac{127}{182}} 
                      = \frac{123}{127}\ne\frac{123}{125}
$$

Positive and Negative predictive value have changed, which is why we do not use
these to evaluate tests!

Hey... Wait a minute! While I now understand why $P(D\mid{}\theta)$ and
$P(\neg{}D\mid{}\neg\theta)$ are what I should use for evaluating tests, my primary
goal was not to figure out how good the tests were, but what I should believe
after taking a test, "given a test results, what is the probability of
COVID19", or formally $P(\theta\mid{}D)$ or $P(\theta\mid{}\neg{}D)$... Well, stay tuned
for the next blog post.
