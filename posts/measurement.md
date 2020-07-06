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

## Naive Solution
The Naive solution, is of course to weight each item one time, that would give us

$$
\begin{aligned}
	x_1 = a+\epsilon_a \\
	x_2 = b+\epsilon_b \\
	x_3 = c+\epsilon_c
\end{aligned}
$$

Where $a$, $b$ and $c$ are the measurements, and their associated uncertainly.

## Weighing two at a time
How can we do better, the official solution to the riddle is to measure each
item twice. Putting that in a matrix, we can have each row of the matrix be one
equation, corresponding to one measurement ($a$, $b$ or $c$):

$$
\begin{aligned}
	X_{3,3} = 
	\begin{bmatrix}
		x_{1} & x_{2} & 0     \\
		0     & x_{2} & x_{3} \\
		0     & 0     & x_{3} \\
	\end{bmatrix}
	= 
	\begin{bmatrix}
		a \\ b \\ c \\
	\end{bmatrix} +
	\begin{bmatrix}
		\epsilon_a \\ \epsilon_b \\ \epsilon_c \\
	\end{bmatrix}
\end{aligned}
$$

Then we factor the $x_i$ into it's own vector, to make the matrix simpler:

$$
\begin{aligned}
	X_{3,3} = 
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
	\end{bmatrix} + \begin{bmatrix}
		\epsilon_a \\ \epsilon_b \\ \epsilon_c \\
	\end{bmatrix} \\
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
	\end{bmatrix} + \begin{bmatrix}
		\epsilon_a \\ 
		\epsilon_b \\
		\frac{\epsilon_c - \epsilon_a + \epsilon_b}{2} \\
	\end{bmatrix} \\
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
	\end{bmatrix} + \begin{bmatrix}
		\epsilon_a \\ 
		\frac{\epsilon_a+\epsilon_b-\epsilon_c}{2} \\
		\frac{-\epsilon_a + \epsilon_b + \epsilon_c}{2} \\
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
	\end{bmatrix} + \begin{bmatrix}
		\frac{\epsilon_a-\epsilon_b+\epsilon_c}{2} \\
		\frac{\epsilon_a+\epsilon_b-\epsilon_c}{2} \\
		\frac{-\epsilon_a + \epsilon_b + \epsilon_c}{2} \\
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
	\end{bmatrix} + \begin{bmatrix}
		\frac{\epsilon_a-\epsilon_b+\epsilon_c}{2} \\
		\frac{\epsilon_a+\epsilon_b-\epsilon_c}{2} \\
		\frac{-\epsilon_a + \epsilon_b + \epsilon_c}{2} \\
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

I do not know how to calculate the exact variance of the weighting twice
strategy. But, I know from the above matrix how the error terms are added
together. So we can simply sample from the error distribution, and compare to
the naive solution.

Intuitively the error on $x_a$, $x_b$ and $x_c$ should follow the same
distribution, let's call it $\epsilon_2$ all the $\epsilon$ follow a $N(0,1)$ distribution, and because the normal distribution is symmetrical around it's mean:

$$
N(0,1)=-N(0,1)
$$

We can (while sampling from it) perform the following simplification.

$$
\begin{aligned}
	\epsilon_2 &= \frac{1}{2}\sum_{i=1}^{3}\epsilon  \\
               &\approx \frac{\epsilon_a-\epsilon_b+\epsilon_c}{2} \\
               &\approx \frac{\epsilon_a+\epsilon_b-\epsilon_c}{2} \\
               &\approx \frac{-\epsilon_a + \epsilon_b + \epsilon_c}{2}
\end{aligned}
$$

Which means the $\epsilon_2$ estimate will be true for all $x_i$.

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

Since the error distribution was centered on 0, the standard error is:

$$
\epsilon_2 = \frac{1}{N}\sum_{i=1}^{N} (\mu - 0)^2
$$

Which is equivalent to the following python:

```
epsilon2 = ((mu - 0) ** 2).sum() / N  # 0.75
```

Resulting in a $\epsilon_2=0.75\epsilon$, so a 25% error reduction.

So the weighing two at a time is almost as good as weighting every item twice
($\frac{1}{\sqrt{2}}=0.71$), which would require using the weight 6 times!
