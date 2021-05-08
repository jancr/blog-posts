# a visual explanation of Bayesian updating

As a teaser here is the visual version of Bayesian updating:

![svg](figures/bayesian_updating/update2.svg)

But in order to understand that figure we need to go through the prior and likelihood!

You find me standing in a basketball court ready to shoot some hoops. What do
you believe about my performance before I take a shot?. There are no good Null
hypothesis here unless you happen to have a lot of knowledge about the average
human basket ball performance!, and even so, why do you care whether I am
significant different from the average?, You can fall back to the [new
statistics](https://journals.sagepub.com/doi/full/10.1177/0956797613504966)
which is almost as good as the Bayesian approach, it but does not answer what you
should believe before I take a shot.

The Beta distribution is a popular prior for binary events, when the two parameter ($\alpha$ and $\beta$) are equal to 1, it is uniform. Since you my dear reader have no concept about my basket skills you assume a $\theta$ comes from a $Beta(1, 1)$  distribution, formally:

$$p(\theta) \sim Beta(1, 1)$$

Where $\theta$ is my probability of scoring, the distribution looks like this:

![svg](figures/bayesian_updating/prior.svg)

Completely Uniform, a great prior when you are totally oblivious.

**I take a shot and miss ($z=0$)**, the likelihood of a miss looks like this:

![svg](figures/bayesian_updating/likelihood.svg)

(if you are extra currious, you can brush up on the math behind all the binary distributions [here](https://www.badprior.com/blog/dichotomous-distributions/))

Notice that:

 *  $p(z=0\mid\theta=0)=1$, the likelihood that I always miss is 1
 *  $p(z=0\mid\theta=0.5)=0.5$, the likelihood that I miss half the time is 0.5
 *  $p(z=0\mid\theta=1)=0$, the likelihood that I always hit is 0, which is
    obvious as I can't score all the time if I just missed.

Notice that these likelihoods and not probabilities, but how likely the data are for different values of $\theta$, so it is twice as likely:

$$\frac{p(z=0\mid\theta=0)}{p(z=0\mid\theta=0.5)}=\frac{1}{0.5}=2$$

That the data $z=0$ was generated by $\theta=0$ compared to $\theta=0.5$.

## Bayesian Updating Math

Here is Bayes theorem for the Bernoulli distribution with a Beta prior, where
the parameter $z$ is 1 when I score and 0 otherwise:
 
$$p(\theta|z) = \frac{p(z\mid{}\theta)p(\theta)}{p(z)}$$

For technical reason $p(z)$, the probability of the data, is difficult to
calculate, it is however 'just a normalization constant' because it does not
depend on $\theta$ which is my scoring probability, thus we can simply drop it
and get an unnormalized posterior:

$$p(\theta|z) \propto p(z\mid{}\theta)p(\theta)$$

An normalized posterior is simply a density function that does not sum to 1,
which means when we plot it it looks 'correct' except we have screwed up the numbers on
the y axis.

## Visual Bayesian Updating

So now we have a 'square' prior $p(\theta) \sim Beta(1, 1)$ and we have a triangle likelihood $p(z=0\mid\theta)$, if we multiply them together we get the unnormalized posterior, so we do:

$$p(\theta|z) \propto p(z\mid{}\theta)p(\theta)$$

Which intuitively can be taught of as: the square makes everything equally likely, so the likelihood will dominate the posterior, or in dodgy math:

$$posterior \propto square \times triangle \propto triangle$$

Here is the Figure:

![svg](figures/bayesian_updating/update1.svg)

Try to put your finger on the figure check that $\theta=0.5$ is 1 for the
square and 0.5 for the triangle and is thus $1\times{}0.5=0.5$ in the
unnormalized posterior

**I shoot again and score!**

Now we use the previous posterior as the new prior, but because we score we get an 'opposite triangle' which is the likelihood of $p(z=1\mid\theta)$

Again we multiply the prior triangle by the likelihood triangle and get a blob
centered on 0.5 as the posterior:

![svg](figures/bayesian_updating/update2.svg)

Notice how the posterior is peaked at $\theta=0.5$, this is because the two
triangles at the center have an unnormalized posterior density of
$0.5\times{}0.5=0.25$ where at edges such as $\theta=0.9$ they have
$0.9\times{}0.1=0.09$

**I shoot again and sore!**

So now again the previous blob posterior is our new prior, which we multiply by
the 'I scored triangle' resulting in a blob that has a mode above 0.5, which
makes sense as I made 2/3 shots:

![svg](figures/bayesian_updating/update3.svg)

While this may seem like a cute toy example it's a totally valid way of solving
a Bayesian posterior, and is the way all most popular bayesian books (Gelman,
Kruschke and McElreath) introduce the concept!

## Bayesian Updating using Conjugation 

In the case of the Bernoulli events we can actually solve the posterior easily
because the Beta is conjugated to the Bernoulli, conjugation is simply fancy statistics speak for it having a simple mathematical form, and that form is also a Beta distribution, thus you can update the beta distribution using this simple rule:

$Beta(\alpha + z, \beta + 1 - z)$

So we Started with a prior with $\alpha=\beta=1$

$Beta(1, 1)$

Then we got a miss, z=0

$Beta(1, 2)$

Then we got a hit, z=1

$Beta(2, 2)$

Then we got a miss, z=1

$Beta(3, 2)$

We can plot the $Beta(3, 2)$ posterior

![svg](figures/bayesian_updating/beta_posterior.svg)

Notice how the this posterior has the exact same shape as the one we got via
updating, the only different is the numbers on the y-axis.

(Hi, if you made it this far please comment, if there were something that was not well explained, I care more about my statistics communication skills than my ego, so negative feedback is very welcome)
