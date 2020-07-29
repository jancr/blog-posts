# Good Bayesian - Part 2 

## Priors and Bayes Theorem

In this section we will talk about priors, what we believe prior to seeing the
data. Sometimes Priors are very rigours, and based on previous research,
and sometimes they are more diffuse.

As in [the previous post:](../../blog/good-bayesian-2/)

$$
\begin{aligned}
	\theta &= \text{ What we care about} \rightarrow\text{ COVID19} \\
	D      &= \text{ What we measure (the data)} \rightarrow\text{ Test Result}
\end{aligned}
$$

The symbol "$\neg$" means not, thus $\theta$ means has had COVID19,
$\neg\theta$ means has not had COVID19, and $P(\theta)$ means the probability
of COVID19, since $P(\theta)$ does not contain $D$, we call it the prior, as it
is the prior probability of $\theta$ before we see the data.

## COVID19 Prior ($P(\theta)$)
In March COVID19 tests were conserved for health care providers. My wife's
coworker came down sick after coming home from a COVID19 hot zone, and
subsequently other colleges got sick, along with my entire family. While we
were recovering a partner of another coworker got sick and tested positive. So
we were part of a chain were many people got sick, but only one person a few
step removed from us were ever tested (and was positive).

What is the probability that my wife had COVID19?, my wife
had all the symptoms of a moderate case of COVID19. Because of the positive
test, the symptoms, and the contagion of the disease, I estimated that there
were 85% chance that we had COVID19. 

While 85% is somewhat arbitrary, it was the best inference I personally could
make based on the then available "data", and in any case, it is better than
picking 3% (the number of people in Denmark with antibodies). 

<!-- The following is a mess -->

## COVID19 Posterior ($P(\theta\mid{}D$)
In the previous post we learned that

$$
P(D\mid{}\theta) = \frac{P(\theta,D)}{P(\theta)}
$$

Thus, by reorganizing

$$
P(\theta,D) = P(D\mid{}\theta)P(\theta) 
$$

And, the same holds if we flip the variables

$$
P(\theta,D) = P(\theta\mid{}D)P(D) 
$$

Obviously $P(\theta,D)=P(\theta,D)$ So

$$
P(\theta\mid{}D)P(D) = P(D\mid{}\theta)P(\theta)
$$

And by dividing $P(D)$ away, we have Bayes Theorem.

$$
P(\theta\mid{}D)= \frac{P(D\mid{}\theta)P(\theta)}{P(D)}
$$

In common parlance, the 4 parts of Bayes Theorem are called:

$$
posterior = \frac{likleyhood\times{}prior}{data}
$$

The Sensitivity ($P(D\mid{}\theta)$) and Specificity ($P(\neg{}D\mid{}\neg\theta)$) are
likelihoods which tells us how likely it is to observe the two datums $D$ and
$\neg{}D$. The posterior is the probability of having had COVID19 given a test
result, now all we need it the innocent little $D$. How do we calculate that?

## The Generative Process ($P(D)$)

How was the data generated?, Here the example is one positive test.
Because our variable is dichotomous, there are two
ways to generate that data:

$$
P(D) = P(D,\theta) + P(D,\neg\theta)
$$

Either you have a positive test and have COVID19, or you have a positive test
and you do not have COVID19. We do not have these joined probabilities at hand,
but we know to convert between joined and conditional probabilities:

$$
P(D) = P(D\mid{}\theta)P(\theta) + P(D\mid{}\neg\theta)P(\neg\theta)
$$

The above is very close to what we have, but we do not have $P(\neg\theta)$ and
we do not have $P(D\mid{}\neg\theta)$, we do however know that probabilities have to
sum to 1, thus we know that:

$$
P(\theta) + P(\neg\theta) = 1
$$

$$
P(D\mid{}\neg\theta) + P(\neg{}D\mid{}\neg\theta) = 1
$$

Knowing this we can reformulate the data generation process in terms of the 3 variables we have:

$$
P(D) = P(D\mid{}\theta)P(\theta) + (1 - P(\neg{}D\mid{}\neg\theta))(1 - P(\theta))
$$


## Posterior
What we have is a prior ($P(\theta)$) and, two likelihoods: $P(D\mid{}\theta)$ and,
$P(\neg{}D\mid{}\neg\theta)$.

So my wife took an antibody test, and it was negative. Thus we have to
reformulate Bayes Theorem and the generative process in terms of $P(\neg{}D)$

First we describe how the data could be generated, the data here being the
Negative test ($\neg{}D$):

$$
\begin{aligned}
    P(\neg{}D) &= P(\neg{}D,\neg\theta) &+& P(\neg{}D,\theta) \\
               &= P(\neg{}D\mid{}\neg\theta)\neg{}P(\theta) &+& P(\neg{}D\mid{}\theta)P(\theta) \\
               &= P(\neg{}D\mid{}\neg\theta)(1 - P(\theta)) &+& (1 - P(D\mid{}\theta))P(\theta) 
\end{aligned}
$$

Then the posterior (the probability of having COVID19 given a negative test) is:

$$
\begin{aligned}
    P(\neg\theta\mid{}\neg{}D) &= \frac{P(\neg{}D\mid{}\neg\theta)(1 - P(\theta))}{P(^\neg{}D)} \\
    P(\theta\mid{}\neg{}D) &= 1 - \frac{P(\neg{}D\mid{}\neg\theta)(1 - P(\theta))}{P(^\neg{}D)}
\end{aligned}
$$

Let's plug in the numbers for $P(\neg{}D)$

$$
\begin{aligned}
    P(\neg{}D) &= P(\neg{}D\mid{}\neg\theta)(1 - P(\theta)) + (1 - P(D\mid{}\theta))P(\theta) \\
            &= \frac{123}{124}(1 - 0.85) + (1 - \frac{27}{28})\times{}0.85 \\
            &\approx 0.0149 + 0.030 \approx 0.179
\end{aligned}
$$

And then for the posterior:

$$
P(\theta\mid{}\neg{}D) = 1 - \frac{P(\neg{}D\mid{}\neg\theta)(1 - P(\theta))}{P(\neg{}D)}
                       = 1 - \frac{\frac{123}{124}(1 - 0.85)}{0.179} \approx 0.170
$$

## Conclusion
So there is a 17% chance that my wife has had COVID19.

Is this the best inference we could do with the available
data?. Can we now proclaim ourselves enlightened Bayesian. **NO!**, what we
have done is calculating a point estimates... True Bayesians think in
distributions, There is some uncertainly associated with the Data used to
calculate specificity and sensitivity, and there are also some uncertainly
around my point estimate of $P(\theta)=0.85$. This is what the next few blog
posts will be about. 

The uncertainly associated with the test data is relatively easy to take into
account, and is at the heart of Bayesian modeling, and will be the topic of the
next post.
