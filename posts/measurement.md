# Measurement

You have 3 items ($x_1$, $x_2$ and $x_3$), of unknown mass, it is imperative
you get as precise measurements as possible. You have been granted access to
use the alien artifact 3 times, which has a measurement error of $\epsilon\sim
N(0, 1ng)$

How do you best proceed?, How do you get as precise estimates of $x_i$ as
possible?, think before you proceed.

The above "riddle" is of course, weird, because rarely scales are expensive to
use, however most people have an intuitive understanding of how they work, so I
am using it as a standing for any scientific readout.

## TLDR (no math intuitive solution)
If you measure them in pairs of two, then you can recover the original mass, by
adding the two measurements which contains the item of interest, and
subtracting the one measurement that do not, that reduces the overall error
because you now have 3 half measurements, each of those 3 measurements only
have a quarter of squared error, adding these errors up gives $\frac{3}{4}$ squared
error, which is less than 1, which you get from measuring them individually.

## Naive Solution
The Naive solution, is of course to weight each item one time, that would give us

$$
\begin{aligned}
	x_1 = a+\epsilon \\
	x_2 = b+\epsilon \\
	x_3 = c+\epsilon
\end{aligned}
$$

Where $a$, $b$ and $c$ are the measurements, and their associated uncertainly.

## Weighing two at a time
How can we do better, the official solution to the riddle is to measure each
item twice. Putting that in a matrix, we can have each row of the matrix be one
equation, corresponding to one measurement ($a$, $b$ or $c$):

$$
\begin{aligned}
	X = 
	\begin{bmatrix}
		x_{1} & x_{2} & 0     \\
		0     & x_{2} & x_{3} \\
		x_{1} & 0     & x_{3} \\
	\end{bmatrix}
	= 
	\begin{bmatrix}
		a \\ b \\ c \\
	\end{bmatrix}
\end{aligned}
$$

Then we factor the $x_i$ into it's own vector, to make the matrix simpler:

$$
\begin{aligned}
	X = 
	\begin{bmatrix}
		1 & 1 & 0 \\
		0 & 1 & 1 \\
		1 & 0 & 1 \\
	\end{bmatrix}
	\begin{bmatrix}
		x_1 \\ x_2 \\ x_3 \\
	\end{bmatrix} = 
	\begin{bmatrix}
		a \\ b \\ c \\
	\end{bmatrix} 
\end{aligned}
$$

Now we can 'solve' this matrix by creating a triangle matrix, one where the
lower left triangle is all zeros. The first row is always 'triangular', and so
is the second as it starts with a zero, the third row needs to be converted to
$[0, 0, 1]$, which is achieved by subtracting the first row, adding the
second, and dividing by two:

$$
\begin{aligned}
	\begin{bmatrix}
		1 & 1 & 0 \\
		0 & 1 & 1 \\
		0 & 0 & 1 \\
	\end{bmatrix}
	\begin{bmatrix}
		x_1 \\ x_2 \\ x_3 \\
	\end{bmatrix}
	= 
	\begin{bmatrix}
		a \\ b \\ \frac{c - a + b}{2} \\
	\end{bmatrix}
\end{aligned}
$$

Solving the linear equations now is trivial, because the difference
between row $i$ and row $i+1$ is 0 or 1 in column $i$. So the matrix can be
converted to the $I$ by iteratively subtracting the previous row:

$$
\begin{aligned}
	\begin{bmatrix}
		1 & 1 & 0 \\
		0 & 1 & 0 \\
		0 & 0 & 1 \\
	\end{bmatrix}
	\begin{bmatrix}
		x_1 \\ x_2 \\ x_3 \\
	\end{bmatrix}
	&= 
	\begin{bmatrix}
		a \\ 
		\frac{a+b-c}{2} \\
		\frac{-a + b + c}{2} \\
	\end{bmatrix}
	\\
	\begin{bmatrix}
		1 & 0 & 0 \\
		0 & 1 & 0 \\
		0 & 0 & 1 \\
	\end{bmatrix}
	\begin{bmatrix}
		x_1 \\ x_2 \\ x_3 \\
	\end{bmatrix}
	&= 
	\begin{bmatrix}
		\frac{a-b+c}{2} \\
		\frac{a+b-c}{2} \\
		\frac{-a + b + c}{2} \\
	\end{bmatrix}
	\\
	\begin{bmatrix}
		x_1 \\ x_2 \\ x_3 \\
	\end{bmatrix}
	&= 
	\begin{bmatrix}
		\frac{a-b+c}{2} \\
		\frac{a+b-c}{2} \\
		\frac{-a + b + c}{2} \\
	\end{bmatrix}
\end{aligned}
$$

## Evaluation of naive vs two at a time
Finally, the Linear algebra side quest has ended. Let's investigate what this
weighing two at a time has bought us. 

Intuitively two thing can happen:

1. The error has increased because variances are additive:
    * $T = X + Y$ 
    * $\sigma_{T}^2 = \sigma_{X}^2 + \sigma_{Y}^2$
2. The error has decreased because the standard error of the mean is defined as:
    * $\epsilon = \frac{\sigma^2}{\sqrt{N}}$

We can now calculate $\epsilon_2$, for our two at a time trick, by 
exploiting the fact that variances are additives (even if two variables are
subtracted from each other). Because all $x_i$ are a linear combination of all 3
measurements: $a$, $b$ and $c$, they will have the same variance
($\epsilon_2$). 
<!-- Because $Var(a) = Var(-a) = \epsilon$ -->

$x_1$ can be rewritten as:

$$
x_1 = \frac{a-b+c}{2} = \frac{a}{2} + \frac{-b}{2} + \frac{c}{2}
$$

The error for $a$, $b$ and $c$ is $\epsilon=1$. Let's calculate $\epsilon_2$,
the error of the two at a time strategy.

$$
\begin{aligned}
    \epsilon_2^2 &= Var(x_1) \\
                 &= Var(\frac{a}{2}) + Var(\frac{-b}{2}) + Var(\frac{c}{2}) \\
                 &= \sum_{i=1}^{3} Var(\frac{\epsilon}{2}) \\
                 &= \frac{3\epsilon^2}{2^2} = \frac{3\epsilon^2}{4} \\
    \epsilon_2   &= \sqrt{\frac{3\epsilon^2}{4}} = \frac{\sqrt{3}\epsilon}{2} 
                    \approx{}0.866\epsilon 
\end{aligned}
$$

<!-- $$ -->
<!-- \begin{aligned} -->
<!--     x_1 = \frac{a-b+c}{2} = \frac{a}{2} + \frac{-b}{2} + \frac{c}{2} -->
<!--     \epsilon_2^2 &= Var(x_1) = \frac{a}{2} + \frac{-b}{2} + \frac{c}{2} -->
<!--  -->
<!--     \epsilon_2^2 &= Var(x_1) = Var(x_2) = Var(x_3) \\ -->
<!--                  &= \sum_{i=1}^3\frac{1^2}{2^2} = \frac{3}{4} \\ -->
<!--     \epsilon_2   &= \sqrt{\frac{3}{4}} = \frac{\sqrt{3}}{2} \approx{}0.866  -->
<!-- \end{aligned} -->
<!-- $$ -->

Again yielding $\epsilon_2=0.87\epsilon$
Which is almost half as good as cheating and performing the naive experiment
twice:

$$
\begin{aligned}
    \epsilon_{twice} &= \frac{\epsilon}{\sqrt{2}} \\
                     &= 0.71\epsilon
\end{aligned}
$$


## Optional: Sampling to sanity check the $\epsilon_2$

Before I figured out how to calculate $\epsilon_2$, I found it by sampling.

Intuitively the error on $x_a$, $x_b$ and $x_c$ should follow the same
distribution, let's call it $\epsilon_2$ all the $\epsilon$ follow a $N(0,1)$
distribution, and because the normal distribution is symmetrical around it's
mean:

$$
N(0,1)=-N(0,1)
$$

We can (while sampling from it) perform the following simplification.

$$
\begin{aligned}
    \epsilon_2 &=\pm\frac{a_{error}}{2} \pm\frac{b_{error}}{2} \pm\frac{c_{error}}{2} \\ 
               &= \frac{\epsilon'}{2} + \frac{\epsilon'}{2} + \frac{\epsilon'}{2} \\ 
               &= \frac{1}{2}\sum_{i=1}^{3}\epsilon'  \\
\end{aligned}
$$

Where $\epsilon'$ is a random variable drawn from from the error distribution
$N(0, \epsilon)$

<!-- Which means the $\epsilon_2$ estimate will be true for all $x_i$. -->

We estimate $\epsilon_2$ by 1. Sampling $3\times{}10^6$ times from a normal
distribution with $N(0, 1)$, and then reshape it to $(10^6, 3)$. 2. We sum over
the last axis and divide by 2 ($\frac{1}{2}\sum_{i=1}^{3}\epsilon$). Then we
have $10^6$ error estimates.

```
import scipy as sp
import scipy.stats
data = sp.stats.norm(0, 1).rvs(3*N).reshape(N, 3)
mu = data.sum(1) / 2
```

Since the error distribution was centered on 0, the variance of the error is:

$$
\begin{aligned}
    \epsilon_2^2 &= \frac{1}{N}\sum_{i=1}^{N} (\mu_i - 0)^2 \\
    \epsilon_2   &= \sqrt{\epsilon_2^2}
\end{aligned}
$$

Which is equivalent to the following python:

```
epsilon2_var = ((mu - 0) ** 2).sum() / N  # 0.75
epsilon2_sd = epsilon2_sd ** 0.5  # 0.87
```

Again yielding $\epsilon_2=0.87\epsilon$


## Optional: Solutions worse than the Naive
One may be tempted to think that if weighing two at a time is good because you
got 8 measurements, then maybe 9 measurements are better

Thus, instead of

$$
\begin{aligned}
	X = 
	\begin{bmatrix}
		1 & 1 & 0 \\
		0 & 1 & 1 \\
		1 & 0 & 1 \\
	\end{bmatrix}
	\begin{bmatrix}
		x_1 \\ x_2 \\ x_3 \\
	\end{bmatrix} = 
	\begin{bmatrix}
		a \\ b \\ c \\
	\end{bmatrix} 
\end{aligned}
$$

We do:

$$
\begin{aligned}
	X = 
	\begin{bmatrix}
		1 & 1 & 1 \\
		0 & 1 & 1 \\
		1 & 0 & 1 \\
	\end{bmatrix}
	\begin{bmatrix}
		x_1 \\ x_2 \\ x_3 \\
	\end{bmatrix} = 
	\begin{bmatrix}
		a \\ b \\ c \\
	\end{bmatrix} 
\end{aligned}
$$

That unfortunately do not check out as one might expect

$$
\begin{aligned}
	\begin{bmatrix}
		1 & 1 & 1 \\
		0 & 1 & 1 \\
		1 & 0 & 1 \\
	\end{bmatrix}
	\begin{bmatrix}
		x_1 \\ x_2 \\ x_3 \\
	\end{bmatrix} = 
	\begin{bmatrix}
		a \\ b \\ c \\
	\end{bmatrix} 
    \\
	\begin{bmatrix}
		1 & 0 & 0 \\
		0 & 1 & 0 \\
		0 & 0 & 1 \\
	\end{bmatrix}
	\begin{bmatrix}
		x_1 \\ x_2 \\ x_3 \\
	\end{bmatrix} = 
	\begin{bmatrix}
		a - b \\ a - c \\ b + c - a \\
	\end{bmatrix} 
\end{aligned}
$$

So with two at a time, the variance became half of all 3 measurements, resulting
in $\frac{3}{4}$, now we only have addition, no devision, so the variance of
$x_1$ and $x_2$ is 

$$
\epsilon_{1\text{ or }2}^2=\sum_{i=1}^{2}1^2=2
$$ 

The variance of $x_3$ is even worse

$$
\epsilon_{3}^2=\sum_{i=1}^{3}1^2=3
$$


