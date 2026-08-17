# “One size fits all” analytic solutions to the Grad-Shafranov equation

**Antoine J. Cerfon and Jeffrey P. Freidberg**

Plasma Science and Fusion Center, Massachusetts Institute of Technology, 167 Albany Street, Cambridge, Massachusetts 02139, USA

Received 8 December 2009; accepted 1 February 2010; published online 9 March 2010.

*Physics of Plasmas* **17**, 032502 (2010). DOI: [10.1063/1.3328818](https://doi.org/10.1063/1.3328818)

Electronic mail: acerfon@mit.edu.

## Abstract

An extended analytic solution to the Grad-Shafranov equation using Solov’ev profiles is presented. The solution describes standard tokamaks, spherical tokamaks, spheromaks, and field reversed configurations. It allows arbitrary aspect ratio, elongation, and triangularity as well as a plasma surface that can be smooth or possess a double or single null divertor X-point. The solution can also be used to evaluate the equilibrium beta limit in a tokamak and spherical tokamak in which a separatrix moves onto the inner surface of the plasma.

© 2010 American Institute of Physics.

## I. Introduction

Analytic solutions of the Grad-Shafranov (GS) equation [1,2] are useful for studying equilibrium, stability, and transport properties of toroidally axisymmetric fusion devices. They are also useful for benchmarking magnetohydrodynamics (MHD) equilibrium codes for the more difficult situation where the usual large aspect ratio asymptotic expansion is not justified [e.g., the spherical tokamak (ST)].

In 1968, Solov’ev [3] proposed simple pressure and poloidal current profiles which convert the GS equation into a linear, inhomogeneous partial differential equation, much simpler to solve analytically. Despite their simplicity, and the fact that the current density is finite, not zero, at the plasma edge, these profiles still retain much of the crucial physics that describes each configuration of interest, and have, therefore, been extensively studied, particularly for STs [4-7]. The analytic solutions of the GS equation investigated in these papers have been used in the study of plasma shaping effects on equilibrium [8] and transport [9,10] properties.

A general property of the analytic solutions is that they contain only a very few terms, thereby making them attractive from a theoretical analysis point of view. Even so, a crucial point to keep in mind is that while the solutions exactly satisfy the GS equation, one is not free to specify a desired shape for the plasma surface on which to impose boundary conditions. One simply has to take whatever the surface turns out to be after optimizing over the small number of terms kept in the solution. Specifically, this minioptimization results in limits on the class of equilibria that can be accurately described. For instance, Ref. [4] focuses solely on low-$\beta$ equilibria, where the toroidal field is a vacuum field. It thus cannot describe the equilibrium $\beta$ limit. The solution presented in Ref. [5] can describe the equilibrium $\beta$ limit but only for small triangularities. It is ill behaved for moderate to large triangularities. In Refs. [6] and [7] the solutions allow for an inboard separatrix for a wider range of triangularities, but appear to be overconstrained in that the shape of the plasma (elongation and triangularity) depends on the choice of the location of the poloidal field null. Often trial and error is required to choose certain free coefficients that appear in the optimization in order to obtain an equilibrium with certain desired qualitative properties. Rarely, if ever, are nontokamak configurations considered.

In this paper, we present an extended analytic solution to the GS equation with Solov’ev profiles which possesses sufficient freedom to describe a variety of magnetic configurations: the standard tokamak, the ST, the spheromak, and the field reversed configuration (FRC). The new solution possesses a finite number of terms but includes several additional terms not contained in previous analyses. Our solution is valid for arbitrary aspect ratio, elongation, and triangularity. It also allows a wide range of $\beta$: (1) $\beta=0$ force free equilibria, (2) $\beta_p\approx 1$ equilibria where the toroidal field is a vacuum field that could have the value zero, and (3) high-$\beta$ equilibria where a separatrix moves onto the inner plasma surface. Lastly, the solution allows the plasma surface to be either smooth or to possess a double or single null divertor X-point. Most importantly, no trial and error hunting is required. A simple, direct, noniterative, one-pass methodology always yields the desired equilibrium solution.

## II. Analytic solution of the Grad-Shafranov equation with Solov’ev profiles

For toroidally axisymmetric systems, the magnetic field $\mathbf B$ can be expressed as

$$
\mathbf B=\frac{F(\Psi)}{R}\mathbf e_\phi+\frac{1}{R}\nabla\Psi\times\mathbf e_\phi . \tag{1}
$$

Here $\phi$ is the ignorable angle in the usual cylindrical coordinate system $(R,\phi,Z)$, $2\pi\Psi(R,Z)$ is the poloidal flux, and $2\pi F(\Psi)=-I_p(\Psi)$ is the net poloidal current flowing in the plasma and the toroidal field coils. As is well known, the flux function satisfies the GS equation

$$
R\frac{\partial}{\partial R}\left(\frac{1}{R}\frac{\partial\Psi}{\partial R}\right)
+\frac{\partial^2\Psi}{\partial Z^2}
=-\mu_0R^2\frac{dp}{d\Psi}-F\frac{dF}{d\Psi}, \tag{2}
$$

where $p=p(\Psi)$ is the plasma pressure. Both $p$ and $F$ are free functions of $\Psi$ which, along with the boundary conditions, determine the nature of the equilibrium.

The GS equation can be put in a nondimensional form through the normalization $R=R_0x$, $Z=R_0y$, and $\Psi=\Psi_0\psi$, where $R_0$ is the major radius of the plasma and $\Psi_0$ is an arbitrary constant. Equation (2) becomes

$$
x\frac{\partial}{\partial x}\left(\frac{1}{x}\frac{\partial\psi}{\partial x}\right)
+\frac{\partial^2\psi}{\partial y^2}
=-\frac{\mu_0R_0^4}{\Psi_0^2}x^2\frac{dp}{d\psi}
-\frac{R_0^2}{\Psi_0^2}F\frac{dF}{d\psi}. \tag{3}
$$

The well-known choices for $p$ and $F$ corresponding to the Solov’ev profiles are given by

$$
-\frac{\mu_0R_0^4}{\Psi_0^2}\frac{dp}{d\psi}=C,
\qquad
-\frac{R_0^2}{\Psi_0^2}F\frac{dF}{d\psi}=A, \tag{4}
$$

where $A$ and $C$ are constants. Since $\Psi_0$ is an arbitrary constant, one can, without loss in generality, choose it such that $A+C=1$. (The special case $A+C=0$ cannot occur for physical equilibria since it corresponds to a situation beyond the equilibrium limit where the separatrix moves onto the inner plasma surface.) This is formally equivalent to the rescaling $\Psi_0^2\rightarrow(A+C)\Psi_0^2$. Under these conditions, the GS equation with Solov’ev profiles can be written in the following dimensionless form:

$$
x\frac{\partial}{\partial x}\left(\frac{1}{x}\frac{\partial\psi}{\partial x}\right)
+\frac{\partial^2\psi}{\partial y^2}
=(1-A)x^2+A. \tag{5}
$$

The choice of $A$ defines the $\beta$ regime of interest for the configuration under consideration.

The solution to Eq. (5) is of the form $\psi(x,y)=\psi_P(x,y)+\psi_H(x,y)$, where $\psi_P$ is the particular solution and $\psi_H$ is the homogeneous solution. The particular solution can be written as

$$
\psi_P(x,y)=\frac{x^4}{8}+A\left(\frac{1}{2}x^2\ln x-\frac{x^4}{8}\right). \tag{6}
$$

The homogeneous solution satisfies

$$
x\frac{\partial}{\partial x}\left(\frac{1}{x}\frac{\partial\psi_H}{\partial x}\right)
+\frac{\partial^2\psi_H}{\partial y^2}=0. \tag{7}
$$

A general arbitrary degree polynomial solution to this equation for plasmas with up-down symmetry has been derived by Zheng *et al.* in Ref. [5]. For our purposes we need only a finite number of terms in the possible infinite sum of polynomials. Our approach is to write the solution as a series of polynomials with increasing exponents. We truncate the series such that the highest degree polynomials appearing are $R^6$ and $Z^6$. Previous studies have truncated the series at $R^4$ and $Z^4$. The full solution for up-down symmetric $\psi$, including the most general polynomial solution for $\psi_H$ satisfying Eq. (7) and consistent with our truncation criterion, is given by

$$
\begin{aligned}
\psi(x,y)={}&\frac{x^4}{8}+A\left(\frac{1}{2}x^2\ln x-\frac{x^4}{8}\right)
+c_1\psi_1+c_2\psi_2+c_3\psi_3+c_4\psi_4+c_5\psi_5+c_6\psi_6+c_7\psi_7,\\
\psi_1={}&1,\\
\psi_2={}&x^2,\\
\psi_3={}&y^2-x^2\ln x,\\
\psi_4={}&x^4-4x^2y^2,\\
\psi_5={}&2y^4-9y^2x^2+3x^4\ln x-12x^2y^2\ln x,\\
\psi_6={}&x^6-12x^4y^2+8x^2y^4,\\
\psi_7={}&8y^6-140y^4x^2+75y^2x^4-15x^6\ln x
+180x^4y^2\ln x-120x^2y^4\ln x.
\end{aligned} \tag{8}
$$

Equation (8) is the desired exact solution to the GS equation that describes all the configurations of interest that possess up-down symmetry. The unknown constants $c_n$ are determined from as yet unspecified boundary constraints on $\psi$. We note that the formulation can be extended to configurations which are up-down asymmetric. This formulation is described in Sec. IX. However, for simplicity the immediate discussion and examples are focused on the up-down symmetric case. Thus, our next task is to determine the unknown $c_n$ appearing in Eq. (8).

## III. The boundary constraints

Assume for the moment that the constant $A$ is specified (we show shortly how to choose $A$ for various configurations). There are then seven unknown $c_n$ to be determined. Note that, as stated, with a finite number of free constants it is not possible to specify the entire continuous shape of the desired plasma boundary. This would require an infinite number of free constants. We can only match seven properties of the surface since that is the number of free constants available.

Consider first the case where the plasma surface is smooth. A good choice for these properties is to match the function and its first and second derivatives at three test points: the inner equatorial point, the outer equatorial point, and the high point (see Fig. 1 for the geometry). While this might appear to require nine free constants (i.e., three conditions at each of the three points), two are redundant because of the up-down symmetry.

Although it is intuitively clear how to specify the function and its first derivative at each test point, the choice for the second derivative is less obvious. To specify the second derivatives we make use of a well-known analytic model for a smooth, elongated “D” shaped cross section, which accurately describes all the configurations of interest. The boundary of this cross section is given by the parametric equations

$$
\begin{aligned}
x&=1+\epsilon\cos(\tau+\alpha\sin\tau),\\
y&=\epsilon\kappa\sin\tau,
\end{aligned} \tag{9}
$$

where $\tau$ is a parameter covering the range $0\leq\tau\leq2\pi$. Also, $\epsilon=a/R_0$ is the inverse aspect ratio, $\kappa$ is the elongation, and $\sin\alpha=\delta$ is the triangularity. For convex plasma surfaces the triangularity is limited to the range $\delta\leq\sin(1)\approx0.841$.

Using these parametric equations it is straightforward to evaluate the desired second derivatives at each of the three test points. We have found that even with only three test points the outer flux surface resulting from our analytic solution for $\psi$ is smooth and remarkably close to the surface given by Eq. (9) over the entire range of geometric parameters.

The seven geometric constraints are given below, assuming that the free additive constant associated with the flux function is chosen so that $\psi=0$ on the plasma surface. This implies that $\psi<0$ in the plasma:

$$
\begin{aligned}
\psi(1+\epsilon,0)&=0 &&\text{outer equatorial point},\\
\psi(1-\epsilon,0)&=0 &&\text{inner equatorial point},\\
\psi(1-\delta\epsilon,\kappa\epsilon)&=0 &&\text{high point},\\
\psi_x(1-\delta\epsilon,\kappa\epsilon)&=0 &&\text{high-point maximum},\\
\psi_{yy}(1+\epsilon,0)&=-N_1\psi_x(1+\epsilon,0) &&\text{outer-point curvature},\\
\psi_{yy}(1-\epsilon,0)&=-N_2\psi_x(1-\epsilon,0) &&\text{inner-point curvature},\\
\psi_{xx}(1-\delta\epsilon,\kappa\epsilon)&=-N_3\psi_y(1-\delta\epsilon,\kappa\epsilon) &&\text{high-point curvature}.
\end{aligned} \tag{10}
$$

The coefficients $N_j$ are easily found from the model surface specified by Eq. (9) and can be written as

$$
\begin{aligned}
N_1&=\left[\frac{d^2x}{dy^2}\right]_{\tau=0}=-\frac{(1+\alpha)^2}{\epsilon\kappa^2},\\
N_2&=\left[\frac{d^2x}{dy^2}\right]_{\tau=\pi}=\frac{(1-\alpha)^2}{\epsilon\kappa^2},\\
N_3&=\left[\frac{d^2y}{dx^2}\right]_{\tau=\pi/2}=-\frac{\kappa}{\epsilon\cos^2\alpha}.
\end{aligned} \tag{11}
$$

For a given value of $A$ the conditions given by Eq. (10) reduce to a set of seven linear inhomogeneous algebraic equations for the unknown $c_n$. This is a trivial numerical problem.

A similar formulation applies to the situation where the plasma surface has a double null divertor X-point. Here, we can imagine that the smooth model surface actually corresponds to the 95% flux surface. The location of the X-point usually occurs slightly higher and slightly closer to the inboard side of the plasma. Specifically we assume a 10% shift so that $x_{\mathrm{sep}}=1-1.1\delta\epsilon$ and $y_{\mathrm{sep}}=1.1\kappa\epsilon$. In terms of the boundary constraints, there is effectively only one change. At the X-point we can no longer impose the second derivative curvature constraint but instead require that both the tangential and normal magnetic field vanish. The conditions at the inboard and outboard equatorial points are left unchanged. The end result is that if one seeks an equilibrium solution where the plasma surface corresponds to a double null divertor and the 95% surface has an approximate elongation $\kappa$ and triangularity $\delta$, then the constraint conditions determining the $c_n$ are given by

$$
\begin{aligned}
\psi(1+\epsilon,0)&=0 &&\text{outer equatorial point},\\
\psi(1-\epsilon,0)&=0 &&\text{inner equatorial point},\\
\psi(x_{\mathrm{sep}},y_{\mathrm{sep}})&=0 &&\text{high point},\\
\psi_x(x_{\mathrm{sep}},y_{\mathrm{sep}})&=0 &&B_{\mathrm{normal}}=0\text{ at the high point},\\
\psi_y(x_{\mathrm{sep}},y_{\mathrm{sep}})&=0 &&B_{\mathrm{tangential}}=0\text{ at the high point},\\
\psi_{yy}(1+\epsilon,0)&=-N_1\psi_x(1+\epsilon,0) &&\text{outer-point curvature},\\
\psi_{yy}(1-\epsilon,0)&=-N_2\psi_x(1-\epsilon,0) &&\text{inner-point curvature}.
\end{aligned} \tag{12}
$$

Hereafter, we assume that the $c_n$ have been determined. The next step in the analysis is to evaluate the critical figures of merit describing the plasma equilibrium. This is the goal of Sec. IV.

*Figure 1 (image omitted). Definition of the geometric parameters.*

## IV. The plasma figures of merit

There are four figures of merit that are often used to describe the basic properties of Solov’ev MHD equilibria. These are defined as follows:

$$
\begin{aligned}
\text{Total plasma beta}\qquad
\beta&=\frac{2\mu_0\langle p\rangle}{B_0^2+\overline{B}_p^{,2}},\\
\text{Toroidal plasma beta}\qquad
\beta_t&=\frac{2\mu_0\langle p\rangle}{B_0^2},\\
\text{Poloidal plasma beta}\qquad
\beta_p&=\frac{2\mu_0\langle p\rangle}{\overline{B}_p^{,2}},\\
\text{Kink safety factor}\qquad
q^*&=\frac{\epsilon B_0}{\overline{B}_p}.
\end{aligned} \tag{13}
$$

The parameter $B_0$ is the vacuum toroidal field at $R=R_0$. The quantity $\overline{B}_p$ is the average poloidal magnetic field on the plasma surface:

$$
\overline{B}_p
=\frac{\displaystyle\oint B_p\,dl_p}{\displaystyle\oint dl_p}
=\frac{\displaystyle\int\mu_0J_\phi\,dS_\phi}{\displaystyle\oint dl_p}
=\frac{\mu_0I}{R_0C_p}, \tag{14}
$$

where $C_p$ is the normalized poloidal circumference of the plasma surface,

$$
C_p=\frac{1}{R_0}\oint dl_p
=2\int_{1-\epsilon}^{1+\epsilon}\left[1+\left(\frac{dy}{dx}\right)^2\right]^{1/2}dx. \tag{15}
$$

Lastly, $\langle p\rangle$ is the volume averaged pressure,

$$
\langle p\rangle=\frac{\displaystyle\int p\,d\mathbf r}{\displaystyle\int d\mathbf r}. \tag{16}
$$

The goal now is to derive explicit expressions for the figures of merit in terms of $\psi$, $A$, and the geometric parameters $\epsilon$, $\kappa$, and $\delta$. To do this we need the quantities $p$ and $F^2=R^2B_\phi^2$, which are obtained by integrating Eq. (4) and using the fact that $\psi=0$ on the plasma surface:

$$
\begin{aligned}
p(x,y)&=-\frac{\Psi_0^2}{\mu_0R_0^4}(1-A)\psi,\\
B_\phi^2(x,y)&=\frac{R_0^2}{R^2}\left(B_0^2-\frac{2\Psi_0^2}{R_0^4}A\psi\right).
\end{aligned} \tag{17}
$$

When evaluating the figures of merit the normalized quantity $\Psi_0/(aR_0B_0)$ often appears in the results. It is convenient to replace this quantity by $q^*$ which, after a short calculation, can be written as

$$
\frac{1}{q^*}
=-\left(\frac{\Psi_0}{aR_0B_0}\right)\frac{1}{C_p}
\int\frac{dx\,dy}{x}\left[A+(1-A)x^2\right]. \tag{18}
$$

The implication is that when describing MHD equilibria there are certain natural combinations of the figures of merit that appear which depend only on the geometry and the, for now, free parameter $A$. This is convenient for determining general scaling relations.

Using this insight the desired form of the figures of merit is given by

$$
\begin{aligned}
\beta_p(\epsilon,\kappa,\delta,A)
&=-2(1-A)\frac{C_p^2}{V}
\left[\int\psi x\,dx\,dy\right]
\left\{\int\frac{dx\,dy}{x}\left[A+(1-A)x^2\right]\right\}^{-2},\\
\beta_t&=\frac{\epsilon^2\beta_p}{(q^*)^2},\\
\beta&=\frac{\epsilon^2\beta_p}{(q^*)^2+\epsilon^2},
\end{aligned} \tag{19}
$$

where

$$
V=\frac{1}{2\pi R_0^3}\int d\mathbf r=\int x\,dx\,dy \tag{20}
$$

is the normalized plasma volume.

The analysis is now complete and ready to be applied to the magnetic configurations of interest.

## V. ITER

A relatively simple case, which serves as a point of reference, is the International Thermonuclear Experimental Reactor (ITER) tokamak [11]. The baseline design [12] has the following parameters: $\epsilon=0.32$, $\kappa=1.7$, and $\delta=0.33$. The vacuum toroidal magnetic field at $R=R_0$ is $B_0=5.3\ \mathrm{T}$ while the plasma current is $I=15\ \mathrm{MA}$. Using the model surface given by Eq. (9) yields a normalized circumference $C_p=2.79$ and a normalized volume $V=0.53$. These are approximate values used to estimate a value for $q^*=1.57$. When evaluating the figures of merit the actual values of $C_p$ and $V$ from our Solov’ev equilibrium are used. A wide range of beta values is possible for ITER. Choosing $A=-0.155$ yields $\beta_t=0.05$, which is the baseline value.

The flux surfaces for the ITER example, assuming the smooth boundary constraints, are illustrated in Fig. 2. Observe that the shape of the surfaces and the magnetic axis shift are quite plausible as compared with full numerical solutions to the GS equation.

*Figure 2 (image omitted). ITER-like equilibrium ($\epsilon=0.32$, $\kappa=1.7$, and $\delta=0.33$).*

## VI. The spherical tokamak

The ST is a much more challenging configuration to model because of the finite aspect ratio. To show the range of possible ST equilibria we consider the flux surfaces for three qualitatively different regimes of operation. These different regimes are characterized by different values of the free constant $A$.

The first regime corresponds to force free equilibria which, by definition, is equivalent to zero pressure. From Eq. (4) this requires $A=1$. In the second regime of interest we assume that $B_\phi$, even with plasma, remains a vacuum toroidal field: that is, the free function $F(\psi)=R_0B_0=\mathrm{const}$. Again, referring to Eq. (4) we see that this requires $A=0$. The last regime to consider corresponds to the equilibrium beta limit where a separatrix moves onto the inner plasma surface. In this case $A$ is determined by the condition

$$
\psi_x(1-\epsilon,0)=0. \tag{21}
$$

Equation (21) is to be added to the geometric boundary constraints given by Eq. (10). The problem now requires the solution of eight (rather than seven) linear algebraic equations with the unknowns corresponding to the seven $c_n$ plus $A$, still a trivial computational problem.

The flux surfaces for these three cases, assuming the smooth boundary constraints, are illustrated in Fig. 3 for typical parameters corresponding to the National Spherical Torus Experiment (NSTX) [13-15]: $\epsilon=0.78$, $\kappa=2$, $\delta=0.35$, and $q^*=2$. Again, the surfaces appear quite plausible with the magnetic axis moving further out as beta increases. For these cases the figures of merit are summarized in Table I.

*Figure 3 (image omitted). (a) Force-free NSTX-like equilibrium ($\epsilon=0.78$, $\kappa=2$, and $\delta=0.35$). (b) Low-$\beta$ NSTX-like equilibrium with the same geometry. (c) Equilibrium-$\beta$-limit NSTX-like equilibrium with the same geometry. Note the separatrix on the inner surface of the plasma.*

**Table I. Figures of merit for ST equilibria.**

| Figure of merit | Force free | Vacuum $B_\phi$ | Equilibrium limit |
|---|---:|---:|---:|
| $\beta_p$ | 0 | 1.07 | 4.20 |
| $\beta_t$ | 0 | 0.16 | 0.64 |
| $\beta$ | 0 | 0.14 | 0.55 |
| Axis shift $\Delta/a$ | 0.11 | 0.34 | 0.43 |

It is of interest to further examine the properties of the ST at the equilibrium limit. There are many ways to do this and one possible example is as follows. Consider a ST in which the inverse aspect ratio is fixed. For NSTX this value is $\epsilon=0.78$. The triangularity, for the sake of simplicity, is also held fixed at a typical NSTX value: $\delta=0.35$. The kink safety factor is set to $q^*=2$ to provide MHD stability against external kink modes. The goal now is to see how the value of beta at the equilibrium limit varies with the elongation $\kappa$.

It is straightforward to use the analytic solution to plot a curve of $\beta$ versus $\kappa$. This curve is illustrated in Fig. 4. Note that at $\kappa=1$ the critical beta is $\beta=0.38$. For larger $\kappa$ the critical beta increases and at $\kappa=2$, $\beta=0.55$.

*Figure 4 (image omitted). $\beta$ versus $\kappa$ at the equilibrium $\beta$ limit with $\epsilon$, $\delta$, and $q^*$ held fixed ($\epsilon=0.78$, $\delta=0.35$, and $q^*=2$).*

The last example of interest for the ST demonstrates that the analytical solution can be used to create a double null divertor. In this case we redo the intermediate case where $A=0$ using the divertor constraints given by Eq. (12). The resulting flux surfaces are illustrated in Fig. 5. Note that the solution has no difficulty generating a reasonable double null divertor equilibrium.

*Figure 5 (image omitted). Low-$\beta$ NSTX-like equilibrium with double null divertor ($\epsilon=0.78$, $\kappa=2$, and $\delta=0.35$).*

## VII. The spheromak

The calculation of the spheromak flux surfaces closely parallels that of the ST. What is different is the evaluation of the figures of merit. Consider first the flux surfaces. Two interesting cases to consider are as follows. First, according to Taylor’s theory of relaxation [16] the plasma should naturally evolve to a low beta force free state corresponding to $A=1$. A set of flux surfaces for this case using the smooth surface constraints is illustrated in Fig. 6(a) for typical spheromak parameters: $\epsilon=0.95$, $\kappa=1$, and $\delta=0.2$. They look reasonable, and obviously $\beta=0$ since the plasma is force free.

The second case of interest recognizes that theoretically the spheromak also exhibits an equilibrium beta limit when the separatrix moves onto the inner plasma surface. This would not violate Taylor’s theory since the plasma beta can be finite if it is externally heated. As for the ST the value of $A$ for this case is determined by requiring that $\psi_x(1-\epsilon,0)=0$. In terms of the corresponding figures of merit note that by definition $B_\phi=0$ on the plasma surface since there is no toroidal field magnet. This implies that $q^*=0$ for a spheromak. The conclusion is that the critical beta at the equilibrium limit can be written as

$$
\beta=\beta_p=-2(1-A)\frac{C_p^2}{V}
\left[\int\psi x\,dx\,dy\right]
\left\{\int\frac{dx\,dy}{x}\left[A+(1-A)x^2\right]\right\}^{-2}. \tag{22}
$$

The flux surfaces for this case are illustrated in Fig. 6(b), again assuming $\epsilon=0.95$, $\kappa=1$, and $\delta=0.2$. Note the larger shift in the magnetic axis as compared with the force free case. The value of beta at the equilibrium limit is given by $\beta=2.20$.

*Figure 6 (image omitted). (a) Spheromak equilibrium ($\epsilon=0.95$, $\kappa=1$, $\delta=0.2$, and $\beta=0$). (b) Equilibrium-$\beta$-limit spheromak equilibrium ($\epsilon=0.95$, $\kappa=1$, $\delta=0.2$, and $\beta=\beta_p=2.20$). Note the separatrix on the inner surface of the plasma.*

## VIII. The field reversed configuration

The final configuration of interest corresponds to the FRC. Here the plasma is very elongated (i.e., $\kappa\sim10$) and has zero toroidal field (i.e., $B_0=0$), implying that $A=0$. Therefore, $q^*=0$ and $\beta=\beta_p$. Ideally a FRC has $\epsilon=1$ and $\delta=1$.

There are two ways to calculate the flux surfaces. The first method makes use of the solution already derived using the smooth surface constraints and approximates the ideal FRC by choosing $\epsilon=0.99$ and $\delta=0.7$. Recall that $\delta<0.841$ for a convex plasma surface. The flux surfaces for the highly elongated case $\kappa=10$ are illustrated in Fig. 7(a). Observe that this is a reasonably good representation of a FRC. The value of beta is found to be $\beta=1.20$.

The second way to calculate the flux surfaces is to explicitly make the plasma surface a separatrix. In this case $R=0$ is the center line of the plasma, thereby guaranteeing that $\epsilon=1$ and $\delta=1$. To do this we must replace the model surface given by Eq. (9) with one that is compatible with a separatrix. A convenient choice is a half-ellipse,

$$
\begin{aligned}
x&=2\cos\tau,\\
y&=\kappa\sin\tau,
\end{aligned}
\qquad -\frac{\pi}{2}\leq\tau\leq\frac{\pi}{2}. \tag{23}
$$

The solution for the flux surfaces is again given by Eq. (8), but in this case certain coefficients are automatically zero in order for $R=0$ to correspond to the inner boundary of the flux surface: $\psi(0,y)=0$. Specifically, $c_3=c_5=c_7=0$. The remaining nontrivial surface constraints are now given by

$$
\begin{aligned}
\psi(2,0)&=0 &&\text{outer equatorial point},\\
\psi(0,\kappa)&=0 &&\text{high point},\\
\psi_{yy}(2,0)&=-N_1\psi_x(2,0) &&\text{outer-point curvature},\\
\psi_{xx}(0,\kappa)&=-N_3\psi_y(0,\kappa) &&\text{high-point curvature}.
\end{aligned} \tag{24}
$$

For a half-ellipse the parameters $N_1$ and $N_3$ are easily evaluated:

$$
N_1=-\frac{2}{\kappa^2},
\qquad
N_3=-\frac{\kappa}{4}. \tag{25}
$$

The flux surfaces for the second method are plotted in Fig. 7(b) for $\kappa=10$. The separatrix bounding the plasma is apparent. The value of beta is found to be $\beta=1.05$, which is not too different from that obtained using the first method.

*Figure 7 (image omitted). (a) FRC equilibrium obtained with the first method (see Sec. VIII), $\epsilon=0.99$, $\kappa=10$, and $\delta=0.7$. (b) FRC equilibrium obtained with the second method (see Sec. VIII), $\epsilon=1$, $\kappa=10$, and $\delta=1$.*

## IX. Up-down asymmetric formulation

In this section we show how the up-down symmetric formulation can be generalized to include the up-down asymmetric case. Of particular interest is a configuration with a single null divertor.

For up-down asymmetric configurations we assume that the reference surface of interest can be modeled parametrically as follows: $x=x(\tau)$, $y=y(\tau)$. In normalized units the inner and outer equatorial points are still located at $x=1-\epsilon$, $y=0$ and $x=1+\epsilon$, $y=0$, respectively. The upper portion of the surface is smooth and has a maximum at $x=1-\delta\epsilon$, $y=\kappa\epsilon$. The lower portion of the surface is assumed to have a single null X-point located at $x=x_{\mathrm{sep}}$ and $y=y_{\mathrm{sep}}<0$. The model surface can be specified either analytically or numerically.

Under these assumptions the appropriate analytic solution to the GS equation is now given by

$$
\begin{aligned}
\psi(x,y)={}&\frac{x^4}{8}+A\left(\frac{1}{2}x^2\ln x-\frac{x^4}{8}\right)
+c_1\psi_1+c_2\psi_2+c_3\psi_3+c_4\psi_4+c_5\psi_5+c_6\psi_6\\
&+c_7\psi_7+c_8\psi_8+c_9\psi_9+c_{10}\psi_{10}+c_{11}\psi_{11}+c_{12}\psi_{12}.
\end{aligned} \tag{26}
$$

The functions $\psi_1$ to $\psi_7$ have already been defined in Sec. II by Eq. (8). The new functions $\psi_8$ to $\psi_{12}$ have odd symmetry in $y$, thereby allowing up-down asymmetric solutions. These terms can be written as

$$
\begin{aligned}
\psi_8&=y,\\
\psi_9&=yx^2,\\
\psi_{10}&=y^3-3yx^2\ln x,\\
\psi_{11}&=3yx^4-4y^3x^2,\\
\psi_{12}&=8y^5-45yx^4-80y^3x^2\ln x+60yx^4\ln x.
\end{aligned} \tag{27}
$$

There are now 12 unknown coefficients. Following the procedure in the main text there are 12 constraint relations (keeping in mind that the up-down symmetry conditions right at the inner and outer equatorial points no longer automatically apply). A good choice for the boundary constraints corresponding to a single null divertor is given by

$$
\begin{aligned}
\psi(1+\epsilon,0)&=0 &&\text{outer equatorial point},\\
\psi(1-\epsilon,0)&=0 &&\text{inner equatorial point},\\
\psi(1-\delta\epsilon,\kappa\epsilon)&=0 &&\text{upper high point},\\
\psi(x_{\mathrm{sep}},y_{\mathrm{sep}})&=0 &&\text{lower X-point},\\
\psi_y(1+\epsilon,0)&=0 &&\text{outer-point up-down symmetry},\\
\psi_y(1-\epsilon,0)&=0 &&\text{inner-point up-down symmetry},\\
\psi_x(1-\delta\epsilon,\kappa\epsilon)&=0 &&\text{upper high-point maximum},\\
\psi_x(x_{\mathrm{sep}},y_{\mathrm{sep}})&=0 &&B_y=0\text{ at the lower X-point},\\
\psi_y(x_{\mathrm{sep}},y_{\mathrm{sep}})&=0 &&B_x=0\text{ at the lower X-point},\\
\psi_{yy}(1+\epsilon,0)&=-N_1\psi_x(1+\epsilon,0) &&\text{outer-point curvature},\\
\psi_{yy}(1-\epsilon,0)&=-N_2\psi_x(1-\epsilon,0) &&\text{inner-point curvature},\\
\psi_{xx}(1-\delta\epsilon,\kappa\epsilon)&=-N_3\psi_y(1-\delta\epsilon,\kappa\epsilon) &&\text{high-point curvature}.
\end{aligned} \tag{28}
$$

A simple practical choice for the $N_j$ that works well is based on the model surface described by Eq. (9). We assume initially that the configuration is up-down symmetric with $\kappa$ and $\delta$ corresponding to the smooth upper portion of the surface. This assumption then leads to values for the $N_j$ given by Eq. (11). The location of the lower X-point is then chosen, as in Sec. III: $x_{\mathrm{sep}}=1-1.1\delta\epsilon$ and $y_{\mathrm{sep}}=-1.1\kappa\epsilon$.

The calculation of the unknown $c_n$ is still a linear algebraic problem, although now involving 12 unknowns. Still, this is trivial computationally. Equations (26)-(28) represent the formulation of the up-down asymmetric problem.

To demonstrate the procedure we show the results for two examples. The first corresponds to ITER which is characterized by the following parameters: $\epsilon=0.32$, $\kappa=1.7$, $\delta=0.33$, $x_{\mathrm{sep}}=0.88$, $y_{\mathrm{sep}}=-0.60$, and $q^*=1.57$. The value of $A$ is chosen as $A=-0.155$, which leads to a value of beta given by $\beta_t=0.05$. The second example corresponds to a high beta ST. Here, we use NSTX values for the geometry: $\epsilon=0.78$, $\kappa=2$, $\delta=0.35$, $x_{\mathrm{sep}}=0.70$, $y_{\mathrm{sep}}=-1.71$, and $q^*=2$. For this case $A$ is chosen to correspond to a high value of beta but still below the equilibrium limit. Specifically we choose

$$
A=-\frac{(1-\epsilon)^2}{\epsilon(2-\epsilon)}=-0.05,
$$

which is the condition for the toroidal current density to vanish at the inner midplane and leads to $\beta=0.16$. The flux surfaces for these two examples are illustrated in Figs. 8(a) and 8(b). Observe that the surfaces for both examples appear quite reasonable, thereby demonstrating the effectiveness of the procedure to model single null divertor configurations.

*Figure 8 (image omitted). (a) Lower single null ITER-like equilibrium ($\epsilon=0.32$, $\kappa=1.7$, and $\delta=0.33$). (b) Lower single null NSTX-like equilibrium ($\epsilon=0.78$, $\kappa=2$, and $\delta=0.35$).*

## X. Conclusions

We have presented an analytic solution to the GS equation for Solov’ev profiles which substantially extends the range of validity compared with previously derived solutions. By including additional terms in the usual polynomial expansion and requiring a correspondingly larger set of fitting boundary conditions, we obtain solutions for a wide range of geometric parameters ($\epsilon$, $\kappa$, and $\delta$) and figures of merit ($\beta$ and $q^*$). This has enabled us to model, with a single solution, the standard tokamak, the ST, the spheromak, and FRC.

## References

1. H. Grad and H. Rubin, *Proceedings of the Second United Nations Conference on the Peaceful Uses of Atomic Energy* (United Nations, Geneva, 1958), Vol. 31, p. 190.
2. V. D. Shafranov, Zh. Eksp. Teor. Fiz. **33**, 710 (1957) [Sov. Phys. JETP **6**, 545 (1958)].
3. L. S. Solov’ev, Zh. Eksp. Teor. Fiz. **53**, 626 (1967) [Sov. Phys. JETP **26**, 400 (1968)].
4. J. P. Freidberg, *Ideal Magnetohydrodynamics* (Plenum, New York, 1985), pp. 162-167.
5. S. B. Zheng, A. J. Wootton, and E. R. Solano, Phys. Plasmas **3**, 1176 (1996).
6. R. H. Weening, Phys. Plasmas **7**, 3654 (2000).
7. B. Shi, Phys. Plasmas **12**, 122504 (2005).
8. R. Srinivasan, K. Avinash, and P. K. Kaw, Phys. Plasmas **8**, 4483 (2001).
9. Y. Xiao and P. J. Catto, Phys. Plasmas **13**, 082307 (2006).
10. B. Shi, Plasma Phys. Controlled Fusion **49**, 2019 (2007).
11. R. Aymar, V. Chuyanov, M. Huguet, R. Parker, and Y. Shimomura, *Proceedings of the 16th International Conference on Fusion Energy, Montreal, 1996* (International Atomic Energy Agency, Trieste, 1997), Vol. 1, p. 3.
12. R. Aymar, P. Barabaschi, and Y. Shimomura, Plasma Phys. Controlled Fusion **44**, 519 (2002).
13. M. Ono, S. M. Kaye, Y.-K. M. Peng, G. Barnes, W. Blanchard, M. D. Carter, J. Chrzanowski, L. Dudek, R. Ewig, D. Gates, R. E. Hatcher, T. Jarboe, S. C. Jardin, D. Johnson, R. Kaita, M. Kalish, C. E. Kessel, H. W. Kugel, R. Maingi, R. Majeski, J. Manickam, B. McCormack, J. Menard, D. Mueller, B. A. Nelson, B. E. Nelson, C. Neumeyer, G. Oliaro, F. Paoletti, R. Parsells, E. Perry, N. Pomphrey, S. Ramakrishnan, R. Raman, G. Rewoldt, J. Robinson, A. L. Roquemore, P. Ryan, S. Sabbagh, D. Swain, E. J. Synakowski, M. Viola, M. Williams, J. R. Wilson, and NSTX Team, Nucl. Fusion **40**, 557 (2000).
14. S. M. Kaye, M. G. Bell, R. E. Bell, J. Bialek, T. Bigelow, M. Bitter, P. Bonoli, D. Darrow, P. Efthimion, J. Ferron, E. Fredrickson, D. Gates, L. Grisham, J. Hosea, D. Johnson, R. Kaita, S. Kubota, H. Kugel, B. LeBlanc, R. Maingi, J. Manickam, T. K. Mau, R. J. Maqueda, E. Mazzucato, J. Menard, D. Mueller, B. Nelson, N. Nishino, M. Ono, F. Paoletti, S. Paul, Y.-K. M. Peng, C. K. Phillips, R. Raman, P. Ryan, S. A. Sabbagh, M. Schaffer, C. H. Skinner, D. Stutman, D. Swain, E. Synakowski, Y. Takase, J. Wilgen, J. R. Wilson, W. Zhu, S. Zweben, A. Bers, M. Carter, B. Deng, C. Domier, E. Doyle, M. Finkenthal, K. Hill, T. Jarboe, S. Jardin, H. Ji, L. Lao, K. C. Lee, N. Luhmann, R. Majeski, S. Medley, H. Park, T. Peebles, R. I. Pinsker, G. Porter, A. Ram, M. Rensink, T. Rognlien, D. Stotler, B. Stratton, G. Taylor, W. Wampler, G. A. Wurden, X. Q. Xu, and L. Zeng, Phys. Plasmas **8**, 1977 (2001).
15. S. A. Sabbagh, S. M. Kaye, J. Menard, F. Paoletti, M. Bell, R. E. Bell, J. M. Bialek, M. Bitter, E. D. Fredrickson, D. A. Gates, A. H. Glasser, H. Kugel, L. L. Lao, B. P. LeBlanc, R. Maingi, R. J. Maqueda, E. Mazzucato, G. A. Wurden, W. Zhu, and NSTX Research Team, Nucl. Fusion **41**, 1601 (2001).
16. J. B. Taylor, Rev. Mod. Phys. **58**, 741 (1986).
