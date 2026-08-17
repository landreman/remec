# Analytical tokamak equilibrium for shaped plasmas

S. B. Zheng,[^a] A. J. Wootton, and Emilia R. Solano  
*Fusion Research Center, University of Texas at Austin, Austin, Texas 78712*

Received 14 August 1995; accepted 1 November 1995.

## Abstract

A general analytical solution of the Grad-Shafranov equation is presented. Specific functional forms of pressure and plasma current are used; the solution allows arbitrary plasma size, aspect ratio, elongation, triangularity, current, and poloidal beta, without imposing undue constraints amongst those variables.

Copyright 1996 American Institute of Physics. [S1070-664X(96)03302-0]

---

Analytical solutions of the Grad-Shafranov[^1][^2] equation are very useful for theoretical studies of plasma equilibrium, transport, and magnetohydrodynamic (MHD) stability. The well-known Solov'ev equilibrium[^3] has been extensively used for such studies, and also as a benchmark of numerical codes that attempt to find more general solutions. However, the Solov'ev equilibrium solutions typically studied[^3][^4] are overconstrained, either in shape (elliptical) or in plasma current (which is commonly determined by the choice of poloidal beta, $\beta_{\mathrm{pol}}$).

In an axisymmetric system a convenient representation of the magnetic field is

$$
\mathbf{B}=F\boldsymbol{\nabla}\phi+\boldsymbol{\nabla}\Psi\times\boldsymbol{\nabla}\phi . \tag{1}
$$

Here $\phi$ is the ignorable angle in the cylindrical coordinate system $(R,\phi,Z)$, and $F$ and $\Psi$ are axisymmetric scalar functions. The function $F$ is a flux function, associated with the poloidal current in the system, while $\Psi$ is the poloidal flux divided by $2\pi$. MKS units are used throughout this paper.

The Grad-Shafranov[^1][^2] equation can be written as follows:

$$
R\frac{\partial}{\partial R}
\left(\frac{1}{R}\frac{\partial\Psi}{\partial R}\right)
+\frac{\partial^2\Psi}{\partial Z^2}
=-\mu_0 R J_\phi
=-\mu_0R^2\frac{\partial P}{\partial\Psi}
-F\frac{\partial F}{\partial\Psi}. \tag{2}
$$

Here $J_\phi$ is the toroidal plasma current density; $P$ is the thermal pressure, which is a flux function. Note that the toroidal field is not determined by the Grad-Shafranov equation. Its choice will only rescale the values of the safety factor $q$. It will not change the shape of the flux surfaces, or the shape of the profiles of pressure, current density, $q$, etc.

The simplest solution to Eq. (2) can be found by assuming that

$$
-\mu_0\frac{\partial P}{\partial\Psi}=A_1,
\qquad
F\frac{\partial F}{\partial\Psi}=A_2, \tag{3}
$$

with $A_1$ and $A_2$ constant. This obviously reduces the set of possible current-density profile shapes to $J_\phi\propto RA_1-A_2/R$. In particular, this parametrization of the current density does not provide enough flexibility to represent very hollow profiles.

With the restrictions given by Eq. (3), the Grad-Shafranov equation is reduced to the form

$$
R\frac{\partial}{\partial R}
\left(\frac{1}{R}\frac{\partial\Psi}{\partial R}\right)
+\frac{\partial^2\Psi}{\partial Z^2}
=R^2A_1-A_2, \tag{4}
$$

with solution

$$
\Psi=\Psi_0+\frac{A_1}{8}R^4-\frac{A_2}{2}Z^2, \tag{5}
$$

where $\Psi_0$ is a solution of the homogeneous equation

$$
R\frac{\partial}{\partial R}
\left(\frac{1}{R}\frac{\partial\Psi_0}{\partial R}\right)
+\frac{\partial^2\Psi_0}{\partial Z^2}=0. \tag{6}
$$

The usual solution is based on the expansion

$$
\Psi_0=\sum_{n=0,2,\ldots}f_n(R)Z^n, \tag{7}
$$

with each expansion term verifying the equation

$$
R\frac{d}{dR}\left(\frac{1}{R}\frac{df_n(R)}{dR}\right)
=-(n+1)(n+2)f_{n+2},
\qquad n=0,2,\ldots, \tag{8}
$$

which is frequently truncated by assuming $f_n(R)=0$ for $n\geq 3$. It is common to assume up-down symmetry and use only the even expansion terms.

A more convenient expression for $\Psi_0$ is given by a rearranged series expansion as follows:

$$
\Psi_0=
\sum_{n=0,2,\ldots}\sum_{k=0}^{n/2}
G(n,k,R)Z^{n-2k}. \tag{9}
$$

Here the functions $G(n,k,R)$ satisfy the following equations:

$$
R\frac{\partial}{\partial R}
\left(\frac{1}{R}\frac{\partial G(n,0,R)}{\partial R}\right)=0, \tag{10}
$$

and

$$
R\frac{\partial}{\partial R}
\left(\frac{1}{R}\frac{\partial G(n,k,R)}{\partial R}\right)
=-(n-2k+1)(n-2k+2)G(n,k-1,R),
\qquad \frac{n}{2}\geq k>0, \tag{11}
$$

whose solution is given by

$$
G(n,k,R)=g_{n1}G_1(n,k,R)+g_{n2}G_2(n,k,R), \tag{12}
$$

with

$$
G_1(n,0,R)=1,
$$

$$
\begin{aligned}
G_1(n,k>0,R)
&=(-1)^k\frac{n!}{(n-2k)!}
\frac{R^{2k}}{2^{2k}k!(k-1)!}\\
&\quad\times\left(2\ln(R)+\frac{1}{k}-2\sum_{j=1}^{k}\frac{1}{j}\right),
\end{aligned} \tag{13}
$$

and

$$
G_2(n,k,R)=(-1)^k\frac{n!}{(n-2k)!}
\frac{R^{2k+2}}{2^{2k}k!(k+1)!}.
$$

The constants $g_{n1}$, $g_{n2}$, $A_1$, and $A_2$ are determined by imposing external constraints. The advantage of this latter representation lies in the explicit form of the general solution for all $n,k$ presented in Eqs. (12) and (13).

If the plasma is assumed to be up-down symmetric, its shape can be described by four parameters: the equatorial innermost and outermost points, $R_i$ and $R_o$, and the coordinates of the highest point, $(R_t,Z_t)$; or, equivalently, the major radius $R_0=(R_i+R_o)/2$, minor radius $a=(R_o-R_i)/2$, elongation $\kappa=Z_t/a$, and triangularity $\delta=(R_0-R_t)/a$. Specified values of plasma current and pressure at the magnetic axis (or, equivalently, $\beta_{\mathrm{pol}}$) provide two additional constraints. From Eq. (13), the simplest solution with that much freedom is given by

$$
\begin{aligned}
\Psi={}&c_1+c_2R^2+c_3(R^4-4R^2Z^2)
+c_4\left[R^2\ln(R)-Z^2\right]\\
&+\frac{A_1}{8}R^4-\frac{A_2}{2}Z^2,
\end{aligned} \tag{14}
$$

which is equivalent to choosing $f_n=0$ for $n\geq4$ in the traditional expansion given in Eq. (7) (with even terms only).

With Eq. (14), the boundary conditions become

$$
c_1+c_2R_i^2+c_3R_i^4+c_4R_i^2\ln(R_i)
=-\frac{A_1}{8}R_i^4, \tag{15}
$$

$$
c_1+c_2R_o^2+c_3R_o^4+c_4R_o^2\ln(R_o)
=-\frac{A_1}{8}R_o^4, \tag{16}
$$

$$
\begin{aligned}
c_1+c_2R_t^2+c_3R_t^2(R_t^2-4Z_t^2)
+c_4\left[R_t^2\ln(R_t)-Z_t^2\right]
=-\frac{A_1}{8}R_t^4+\frac{A_2}{2}Z_t^2,
\end{aligned} \tag{17}
$$

$$
2c_2+4c_3(R_t^2-2Z_t^2)
+c_4\left[2\ln(R_t)+1\right]
=-\frac{A_1}{2}R_t^2, \tag{18}
$$

$$
I_p=\int J_\phi\,dR\,dZ
=-\int\frac{R^2A_1-A_2}{\mu_0R}\,dR\,dZ, \tag{19}
$$

and

$$
\beta_{\mathrm{pol}}
=\frac{8\pi}{\mu_0}\frac{\displaystyle\int P\,dR\,dZ}{I_p^2}
=-\frac{8\pi A_1}{\mu_0^2I_p^2}\int\Psi\,dR\,dZ. \tag{20}
$$

In previous studies of spherical tokamak equilibria,[^4] only three shape coefficients are used (the coefficient $c_4$ is set to zero). In that case the plasma is described by its minor radius, aspect ratio $A=R_0/a$, and elongation, without allowing a choice for triangularity. Furthermore, the constant $A_2$ is set to zero for simplicity. This has the inconvenient consequence of overconstraining the plasma current or $\beta_{\mathrm{pol}}$, as can be seen by comparing Eqs. (19) and (20).

Multiplying the coefficients $(c_1,\ldots,c_4,A_1,A_2)$ by a constant $\alpha_I$ produces a new solution with the same shape and poloidal beta, but with a rescaled value of the plasma current, $I_p'=\alpha_I I_p$. In the same manner, the spatial dimension can be rescaled. Defining

$$
R'=\alpha_LR,
\qquad
Z'=\alpha_LZ,
$$

the new function $\Psi'(R',Z')$ is defined by

$$
\begin{aligned}
\Psi'(R',Z')
&=\Psi'(\alpha_LR,\alpha_LZ)=\Psi(R,Z)\\
&=d_1+d_2(R')^2
+d_3\left[(R')^4-4(R')^2(Z')^2\right]\\
&\quad+d_4\left[(R')^2\ln(R')-(Z')^2\right]
+\frac{A_1'}{8}(R')^4-\frac{A_2'}{2}(Z')^2.
\end{aligned} \tag{21}
$$

Here

$$
\begin{gathered}
d_1=c_1,
\qquad
d_2=\frac{c_2-c_4\ln(\alpha_L)}{\alpha_L^2},
\qquad
d_3=\frac{c_3}{\alpha_L^4},\\
d_4=\frac{c_4}{\alpha_L^2},
\qquad
A_1'=\frac{A_1}{\alpha_L^4},
\qquad
A_2'=\frac{A_2}{\alpha_L^2}.
\end{gathered}
$$

The function $\Psi'(R',Z')$ satisfies Eq. (4), with coefficients $A_1'$ and $A_2'$ replacing $A_1$ and $A_2$, respectively. The corresponding plasma current is given by

$$
I_p'
=-\int\frac{(R')^2A_1'-A_2'}{\mu_0R'}\,dR'\,dZ'
=\frac{I_p}{\alpha_L}, \tag{22}
$$

and the poloidal beta by

$$
\beta_{\mathrm{pol}}'
=-\frac{8\pi A_1'}{(\mu_0I_p')^2}
\int\Psi'\,dR'\,dZ'
=\beta_{\mathrm{pol}}. \tag{23}
$$

Hence, given the coefficients $c_{10}$, $c_{20}$, $c_{30}$, $c_{40}$, $A_{10}$, and $A_{20}$ for a normalized plasma current $I_p=1\ \mathrm{A}$ and major radius $R_{00}=1\ \mathrm{m}$, the coefficients for a plasma with the same shape and poloidal beta (but different $I_p$ and size) are calculated by similarity:

$$
\begin{aligned}
c_1&=(I_pR_0)c_{10},
&c_2&=\frac{I_p}{R_0}\left[c_{20}-c_{40}\ln(R_0)\right],\\
c_3&=\frac{I_p}{R_0^3}c_{30},
&c_4&=\frac{I_p}{R_0}c_{40},\\
A_1&=\frac{I_p}{R_0^3}A_{10},
&A_2&=\frac{I_p}{R_0}A_{20}.
\end{aligned}
$$

It is simplest to first solve for a plasma with unit current and unit major radius and use the scaling relations described above to find the final desired equilibrium. However, even in this simplest case only numerical solutions to Eqs. (15)-(20) have been found. The coefficients can be computed numerically, given a desired plasma description. For example, a low-aspect-ratio plasma corresponding to the University Spherical Tokamak Experiment (USTX) point design[^5] is shown in Fig. 1. The plasma is characterized by $I_p=1\ \mathrm{MA}$, $R_0=0.70\ \mathrm{m}$, $a=0.49\ \mathrm{m}$, $\kappa=1.7$, $\delta=0.125$, and $\beta_{\mathrm{pol}}=0.4$.

**Figure 1 (image omitted).** Flux contours for a spherical tokamak plasma, with $a=0.49\ \mathrm{m}$, $R_0=0.70\ \mathrm{m}$, $\kappa=1.7$, $\beta_{\mathrm{pol}}=0.40$, and $I_p=1\ \mathrm{MA}$.

A general analytical solution of the Grad-Shafranov equation, of the Solov'ev type, is presented. It is shown that, given the usual parametric description of the plasma (shape and size, $\beta_{\mathrm{pol}}$, and $I_p$), an equilibrium can be computed with enough freedom to independently control pressure and plasma current, for arbitrary choices of plasma size, aspect ratio, elongation, and triangularity. The computation can be performed numerically, or a precomputed fitted solution can be used.

## Acknowledgments

This research has been supported by U.S. Department of Energy Grant Nos. DE-FG03-94ER54241 and DE-FG03-95ER54296.

## References

[^a]: Permanent address: Institute of Physics, Chinese Academy of Sciences, Beijing, People's Republic of China.

[^1]: H. Grad and H. Rubin, in *Proceedings of the Second United Nations Conference on the Peaceful Uses of Atomic Energy* (United Nations, Geneva, 1958), Vol. 31, p. 190.

[^2]: V. D. Shafranov, *Sov. Phys. JETP* **6**, 545 (1958); *Zh. Eksp. Teor. Fiz.* **33**, 710 (1957).

[^3]: L. S. Solov'ev, *Sov. Phys. JETP* **26**, 400 (1968); *Zh. Eksp. Teor. Fiz.* **53**, 626 (1967).

[^4]: J. P. Freidberg, *Ideal Magnetohydrodynamics* (Plenum, New York, 1985), pp. 162-167.

[^5]: See AIP Document No. PAPS PHPAE-03-1176-138 for 138 pages of *USTX-The University Spherical Tokamak Experiment*, Fusion Research Center Report No. 468, The University of Texas at Austin, Austin, 1995, by S. C. McCool *et al.* Order by PAPS number and journal reference from American Institute of Physics, Physics Auxiliary Publication Service, Carolyn Gehlbach, 500 Sunnyside Boulevard, Woodbury, NY 11797-2999. Fax: 516-576-2223; email: janis@aip.org. The price is $1.50 for each microfiche (98 pages) or $5.00 for photocopies of up to 30 pages, and $0.15 for each additional page over 30 pages. Airmail additional. Make checks payable to the American Institute of Physics.

---

## Editorial note on equation verification

The mathematical transcription was checked against the rendered source pages and for internal consistency. Four apparent typographical errors in the printed article were corrected:

1. The sentence introducing Eq. (3) refers to Eq. (1) in the source; it should refer to the Grad-Shafranov equation, Eq. (2).
2. In Eq. (21), the printed $d_3$ term omits the factor $4$ multiplying $(R')^2(Z')^2$.
3. In Eq. (21), the printed $A_2'$ term has a plus sign; consistency with Eqs. (5), (14), and the stated scaling $A_2'=A_2/\alpha_L^2$ requires a minus sign.
4. In Eqs. (19) and (22), the source typesets $R$ and $R'$ beneath a fraction bar outside the respective integral. Because these are integration variables, the factors $1/R$ and $1/R'$ have been placed inside the integrands, as also required by Eq. (2).

The two corrections to Eq. (21) follow directly by substituting $R=R'/\alpha_L$ and $Z=Z'/\alpha_L$ into Eq. (14).
