\# Section 2 — Lorentzian (Cauchy) inverse transform sampling



Given (HWHM Γ=1):

\\\[

p(x)=\\frac{\\Gamma/\\pi}{\\Gamma^2+x^2}\\quad \\Rightarrow \\quad p(x)=\\frac{1}{\\pi(1+x^2)}

\\]



\## CDF

\\\[

F(x)=\\int\_{-\\infty}^{x}\\frac{1}{\\pi(1+t^2)}dt

= \\frac{1}{\\pi}\\left\[\\arctan(t)\\right]\_{-\\infty}^{x}

= \\frac12 + \\frac{1}{\\pi}\\arctan(x)

\\]



\## Inverse CDF

Let \\(u\\in(0,1)\\):

\\\[

u = \\frac12+\\frac{1}{\\pi}\\arctan(x)

\\Rightarrow \\arctan(x)=\\pi\\left(u-\\frac12\\right)

\\Rightarrow x=\\tan\\left(\\pi\\left(u-\\frac12\\right)\\right)

\\]



Equivalent form used in code:

\\\[

x=\\cot(\\pi u)=\\frac{1}{\\tan(\\pi u)}

\\]

This produces the same Cauchy/Lorentzian distribution.



\## Correctness checks used

1\) Quantiles: \\(Q\_{0.25}\\approx -1\\), median \\(\\approx 0\\), \\(Q\_{0.75}\\approx +1\\)  

2\) CDF-uniform: \\(F(x)\\) should be approximately Uniform(0,1) with mean \\(\\approx 0.5\\) and variance \\(\\approx 1/12\\).

