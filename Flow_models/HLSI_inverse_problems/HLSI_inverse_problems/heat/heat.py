\documentclass[11pt]{article}
\usepackage{amssymb}
\usepackage{amsmath}
\usepackage{mathrsfs}
\usepackage{bbm}
\usepackage{tikz}
\usepackage{float}
\usepackage{accents}
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{graphicx}
\usepackage{verbatim}
\usepackage{centernot}
\usepackage{float}
\usepackage[margin=1.2in]{geometry}
\newcommand{\bs}{{\bigskip}}
\newcommand{\floor}[1]{\lfloor #1 \rfloor}
\newcommand{\ceiling}[1]{\lceil #1 \rceil}
\newcommand{\ms}{{\medskip}}
\newcommand{\sms}{{\smallskip}}
\newcommand{\N}{\mathbb{N}}
\renewcommand{\baselinestretch}{1.5}
\newcommand{\indi}{\mathbbm{1}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\I}{\mathbb{I}}
\newcommand{\sigF}{\mathcal{F}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\F}{\mathbb{F}}

\newcommand{\B}{\mathbb{B}}
\newcommand{\dd}[2]{\frac{d{#1}}{d{#2}}}
\newcommand{\pfrac}[2]{\frac{\partial{#1}}{\partial{#2}}}
\newcommand{\lrfloor}[1]{\lfloor{#1} \rfloor }
\newcommand{\spann}[1]{\langle {#1} \rangle}
\newcommand{\lrceil}[1]{\lceil{#1} \rceil }
\newcommand{\fancyF}{\mathscr{F}}
\newcommand{\borelB}{\mathcal{B}}
\newcommand{\Var}{\text{Var}}
\newcommand{\inner}[2]{\langle{#1},{#2}\rangle}
\newcommand{\sinner}[1]{\langle{#1},{#1} \rangle}
\renewcommand{\epsilon}{\varepsilon}
\renewcommand{\overrightarrow}{\vec}
\newcommand{\tb}{\textbf}
\newcommand{\bfrac}[2]{{\displaystyle{\frac{#1}{#2}}}}
\newcommand{\bcup}{{\bigcup\limits}}
\newcommand{\bcap}{{\bigcap\limits}}
\newcommand{\eval}[2]{\Big|_{#1}^{#2}}
\newcommand{\ceil}[1]{\left\lceil #1 \right\rceil}
\newcommand{\nimply}{\centernot\Rightarrow}
\newcommand{\ar}{\Rightarrow}
\newcommand{\norm}[2]{\| #1 \|_{#2}}
\newcommand{\snorm}[1]{{\norm{#1}{}}}
\newcommand{\probp}{{\mathbb{P}}}
\newcommand{\tr}{\text{tr}}
\newcommand{\enorm}[1]{\| #1 \|}
\newcommand{\goto}{\rightarrow}
\newcommand{\bint}[2]{{\displaystyle{\int_{#1}^{#2}}}}
\newcommand{\cint}[1]{\displaystyle{\oint_{#1}}}
\newcommand{\nogoto}{\centernot\rightarrow}
\renewcommand{\baselinestretch}{1.5}
\newcommand{\bsum}[2]{{\displaystyle{\sum_{#1}^{#2}}}}
\newcommand{\bprod}[2]{{\displaystyle{\prod_{#1}^{#2}}}}
\newcommand{\botimes}[2]{{\displaystyle{\bigotimes_{#1}^{#2}}}}
\newcommand{\func}[3]{#1: #2\rightarrow#3}
\newcommand{\sfunc}[2]{#1: #2\rightarrow#2}
\newcommand{\cexp}[2]{\E[#1 \mid #2]}
\usepackage{venndiagram}
\newcommand{\Lim}[1]{\raisebox{0.5ex}{\scalebox{0.8}{$\displaystyle \lim_{#1}\;$}}}
\newcommand{\Limn}{\Lim{n \in \N}}
\newcommand{\Cross}{\mathbin{\tikz [x=1.4ex,y=1.4ex,line width=.2ex] \draw (0,0) -- (1,1) (0,1) -- (1,0);}}
\title{ CSE 393P HW 3}
\author{Alois D'uston,  ald4435}
\setlength{\parindent}{0pt}
\date{}

\begin{document}

\maketitle

\section*{Problem 1}

Let
\[
F_{\mathrm{LS}}(u)=\frac12\int_{\Omega}(u-d)^2\,dx,
\qquad
R_{\mathrm{TN}}(u)=\frac{\beta}{2}\int_{\Omega}|\nabla u|^2\,dx,
\]
and
\[
R^{\delta}_{\mathrm{TV}}(u)=\beta\int_{\Omega}\bigl(|\nabla u|^2+\delta\bigr)^{1/2}\,dx.
\]
Thus
\[
F_{\mathrm{TN}}(u)=F_{\mathrm{LS}}(u)+R_{\mathrm{TN}}(u),
\qquad
F^{\delta}_{\mathrm{TV}}(u)=F_{\mathrm{LS}}(u)+R^{\delta}_{\mathrm{TV}}(u).
\]
Since the boundary condition is homogeneous Neumann, the natural admissible space is
\[
U=H^1(\Omega).
\]
I will use $\hat u$ for a variation (test function) and $\tilde u$ for the Newton step.

\subsection*{(a) First-order necessary conditions: weak and strong forms}

\textbf{Tikhonov regularization.} For a variation $u+\varepsilon \hat u$,
\[
\left.\frac{d}{d\varepsilon}F_{\mathrm{TN}}(u+\varepsilon \hat u)\right|_{\varepsilon=0}
=
\int_{\Omega}(u-d)\hat u\,dx
+
\beta\int_{\Omega}\nabla u\cdot \nabla \hat u\,dx.
\]
Hence the first-order necessary condition is:

\emph{Weak form:} find $u\in H^1(\Omega)$ such that for all $\hat u\in H^1(\Omega)$,
\[
\int_{\Omega}(u-d)\hat u\,dx
+
\beta\int_{\Omega}\nabla u\cdot \nabla \hat u\,dx=0.
\]

Integrating by parts,
\[
\beta\int_{\Omega}\nabla u\cdot \nabla \hat u\,dx
=
-\beta\int_{\Omega}(\Delta u)\hat u\,dx
+\beta\int_{\partial\Omega}(\nabla u\cdot n)\hat u\,ds.
\]
Using the prescribed boundary condition $\nabla u\cdot n=0$ on $\partial\Omega$, we get

\emph{Strong form:}
\[
-\beta\Delta u + u = d \qquad \text{in }\Omega,
\]
with boundary condition
\[
\nabla u\cdot n=0 \qquad \text{on }\partial\Omega.
\]

\textbf{Regularized total variation.} Let
\[
q(u):=|\nabla u|^2+\delta.
\]
Then
\[
\left.\frac{d}{d\varepsilon}F^{\delta}_{\mathrm{TV}}(u+\varepsilon \hat u)\right|_{\varepsilon=0}
=
\int_{\Omega}(u-d)\hat u\,dx
+
\beta\int_{\Omega}\frac{\nabla u\cdot \nabla \hat u}{\sqrt{|\nabla u|^2+\delta}}\,dx.
\]
So the optimality condition is:

\emph{Weak form:} find $u\in H^1(\Omega)$ such that for all $\hat u\in H^1(\Omega)$,
\[
\int_{\Omega}(u-d)\hat u\,dx
+
\beta\int_{\Omega}\frac{\nabla u\cdot \nabla \hat u}{\sqrt{|\nabla u|^2+\delta}}\,dx=0.
\]

Integrating by parts,
\[
\beta\int_{\Omega}\frac{\nabla u\cdot \nabla \hat u}{\sqrt{|\nabla u|^2+\delta}}\,dx
=
-\beta\int_{\Omega}\operatorname{div}\!\left(\frac{\nabla u}{\sqrt{|\nabla u|^2+\delta}}\right)\hat u\,dx
+
\beta\int_{\partial\Omega}\frac{\nabla u\cdot n}{\sqrt{|\nabla u|^2+\delta}}\hat u\,ds.
\]
Thus the strong form is
\[
-\beta\operatorname{div}\!\left(\frac{\nabla u}{\sqrt{|\nabla u|^2+\delta}}\right)+u=d
\qquad \text{in }\Omega,
\]
with natural Neumann boundary condition
\[
\frac{\nabla u\cdot n}{\sqrt{|\nabla u|^2+\delta}}=0
\qquad \text{on }\partial\Omega.
\]
Since $\sqrt{|\nabla u|^2+\delta}>0$, this is equivalent to
\[
\nabla u\cdot n=0 \qquad \text{on }\partial\Omega.
\]

\subsection*{(b) Why $R_{\mathrm{TV}}$ is not differentiable at $\nabla u=0$, but $R^{\delta}_{\mathrm{TV}}$ is}

The TV integrand is
\[
\phi(g)=|g|=(g\cdot g)^{1/2}, \qquad g\in \R^2.
\]
For $g\neq 0$,
\[
D\phi(g)[h]=\frac{g\cdot h}{|g|}.
\]
At $g=0$, if a derivative existed, it would have to be a linear map $L$ satisfying
\[
L(h)=\lim_{\varepsilon\to 0}\frac{|\varepsilon h|-0}{\varepsilon}.
\]
But for $\varepsilon>0$ this quotient is $|h|$, while for $\varepsilon<0$ it is $-|h|$. Hence the two-sided derivative does not exist. Equivalently, the directional dependence $h\mapsto |h|$ is not linear, so there is no Fr\'echet derivative at $g=0$.

Therefore $R_{\mathrm{TV}}(u)=\beta\int_{\Omega}|\nabla u|\,dx$ is not differentiable at points where $\nabla u=0$.

For the regularized TV integrand,
\[
\phi_{\delta}(g)=\sqrt{|g|^2+\delta},
\]
we have for every $g\in \R^2$,
\[
D\phi_{\delta}(g)[h]=\frac{g\cdot h}{\sqrt{|g|^2+\delta}}.
\]
In particular, at $g=0$,
\[
D\phi_{\delta}(0)[h]=0.
\]
So $\phi_{\delta}$ is smooth for every $\delta>0$, and hence $R^{\delta}_{\mathrm{TV}}$ is differentiable.

\subsection*{(c) Infinite-dimensional Newton step for $F^{\delta}_{\mathrm{TV}}$}

Define the gradient functional
\[
G(u)[\hat u]
:=
\int_{\Omega}(u-d)\hat u\,dx
+
\beta\int_{\Omega}\frac{\nabla u\cdot \nabla \hat u}{\sqrt{|\nabla u|^2+\delta}}\,dx.
\]
The Newton step $\tilde u$ solves
\[
D G(u)[\tilde u](\hat u)=-G(u)[\hat u]
\qquad \text{for all }\hat u\in H^1(\Omega).
\]

Let $g=\nabla u$ and $w=\nabla \tilde u$. Then
\[
\frac{d}{d\varepsilon}\left(\frac{g+\varepsilon w}{\sqrt{|g+\varepsilon w|^2+\delta}}\right)\Bigg|_{\varepsilon=0}
=
\frac{w}{\sqrt{|g|^2+\delta}}
-
\frac{(g\cdot w)g}{(|g|^2+\delta)^{3/2}}.
\]
Using the identity $(a\cdot b)c=(ca^T)b$, this can be written as
\[
\left(
\frac{1}{\sqrt{|g|^2+\delta}}I
-
\frac{gg^T}{(|g|^2+\delta)^{3/2}}
\right)w.
\]
Therefore the second variation is
\[
D^2F^{\delta}_{\mathrm{TV}}(u)[\tilde u,\hat u]
=
\int_{\Omega}\tilde u\,\hat u\,dx
+
\int_{\Omega}(A(u)\nabla \tilde u)\cdot \nabla \hat u\,dx,
\]
where
\[
A(u)
:=
\beta\left(
\frac{1}{\sqrt{|\nabla u|^2+\delta}}I
-
\frac{\nabla u\,\nabla u^T}{(|\nabla u|^2+\delta)^{3/2}}
\right).
\]
Equivalently,
\[
A(u)=\frac{\beta}{(|\nabla u|^2+\delta)^{3/2}}
\left((|\nabla u|^2+\delta)I-\nabla u\,\nabla u^T\right).
\]

Hence the Newton step is:

\emph{Weak form:} find $\tilde u\in H^1(\Omega)$ such that for all $\hat u\in H^1(\Omega)$,
\[
\int_{\Omega}\tilde u\,\hat u\,dx
+
\int_{\Omega}(A(u)\nabla \tilde u)\cdot \nabla \hat u\,dx
=
-
\int_{\Omega}(u-d)\hat u\,dx
-
\beta\int_{\Omega}\frac{\nabla u\cdot \nabla \hat u}{\sqrt{|\nabla u|^2+\delta}}\,dx.
\]

Integrating by parts, this becomes

\emph{Strong form:}
\[
-\operatorname{div}(A(u)\nabla \tilde u)+\tilde u
=
-\left[u-d-\beta\operatorname{div}\!\left(\frac{\nabla u}{\sqrt{|\nabla u|^2+\delta}}\right)\right]
\qquad \text{in }\Omega,
\]
with boundary condition
\[
A(u)\nabla \tilde u\cdot n=0
\qquad \text{on }\partial\Omega.
\]
That is,
\[
-\operatorname{div}(A(u)\nabla \tilde u)+\tilde u=-\mathcal{R}(u),
\]
where $\mathcal{R}(u)$ is the strong-form residual of the Euler--Lagrange equation.

\subsection*{(d) Eigenvalues/eigenvectors of $A(u)$ and why TV preserves edges}

Let $g=\nabla u$. Then
\[
A(u)=\beta\left(\frac{1}{\sqrt{|g|^2+\delta}}I-\frac{gg^T}{(|g|^2+\delta)^{3/2}}\right).
\]
There are two natural directions:

\begin{itemize}
    \item the direction parallel to the gradient, $v_{\parallel}=g$ (assuming $g\neq 0$),
    \item any direction perpendicular to the gradient, $v_{\perp}\perp g$.
\end{itemize}

For the parallel direction,
\[
A(u)g
=
\beta\left(\frac{1}{\sqrt{|g|^2+\delta}}-\frac{|g|^2}{(|g|^2+\delta)^{3/2}}\right)g
=
\beta\frac{\delta}{(|g|^2+\delta)^{3/2}}g.
\]
So
\[
\lambda_{\parallel}=\beta\frac{\delta}{(|\nabla u|^2+\delta)^{3/2}},
\qquad
v_{\parallel}=\nabla u.
\]

For any $v_{\perp}$ with $g\cdot v_{\perp}=0$,
\[
A(u)v_{\perp}
=
\beta\frac{1}{\sqrt{|g|^2+\delta}}v_{\perp},
\]
so
\[
\lambda_{\perp}=\beta\frac{1}{\sqrt{|\nabla u|^2+\delta}},
\qquad
v_{\perp}\perp \nabla u.
\]

Now interpret these directions geometrically. Near a sharp image edge, $|\nabla u|$ is large, and the gradient points \emph{across} the edge (normal to the level sets). Thus:
\[
\lambda_{\parallel}
=\beta\frac{\delta}{(|\nabla u|^2+\delta)^{3/2}}
\ll
\lambda_{\perp}
=\beta\frac{1}{\sqrt{|\nabla u|^2+\delta}}.
\]
So diffusion across the edge is strongly suppressed, while diffusion along the edge remains much larger. This means a Newton step for TV smoothing will tend to smooth oscillations \emph{along} the edge without significantly blurring the jump \emph{across} the edge.

By contrast, for Tikhonov regularization the diffusion tensor is simply
\[
A_{\mathrm{TN}}=\beta I,
\]
so the same amount of smoothing acts in every direction. In particular, it smooths just as strongly across an edge as along it, which blurs sharp interfaces.

\subsection*{(e) Large $\delta$ limit and singularity at $\delta=0$}

For large $\delta$,
\[
\sqrt{|\nabla u|^2+\delta}
=
\sqrt{\delta}\,\sqrt{1+\frac{|\nabla u|^2}{\delta}}
\approx
\sqrt{\delta}+\frac{|\nabla u|^2}{2\sqrt{\delta}}+O\!\left(\frac{|\nabla u|^4}{\delta^{3/2}}\right).
\]
Therefore
\[
R^{\delta}_{\mathrm{TV}}(u)
=
\beta\int_{\Omega}\sqrt{|\nabla u|^2+\delta}\,dx
\approx
\beta\sqrt{\delta}\,|\Omega|
+
\frac{\beta}{2\sqrt{\delta}}\int_{\Omega}|\nabla u|^2\,dx.
\]
Up to the irrelevant constant $\beta\sqrt{\delta}\,|\Omega|$, this is exactly a Tikhonov regularization term with effective parameter
\[
\beta_{\mathrm{eff}}=\frac{\beta}{\sqrt{\delta}}.
\]
So for sufficiently large $\delta$, the TV regularization behaves like TN regularization.

Now consider the Hessian tensor when $\delta=0$ and $\nabla u\neq 0$:
\[
A_0(u)
=
\beta\left(\frac{1}{|\nabla u|}I-\frac{\nabla u\,\nabla u^T}{|\nabla u|^3}\right).
\]
Its eigenvalues are obtained from the formulas above with $\delta=0$:
\[
\lambda_{\parallel}=0,
\qquad
\lambda_{\perp}=\frac{\beta}{|\nabla u|}.
\]
Hence the Hessian is singular. In fact, the entire diffusion in the gradient direction disappears. At points where $\nabla u=0$, the expression is not even defined. This shows why taking $\delta$ too small leads to ill-conditioning, while taking $\delta$ too large destroys the edge-preserving behavior and makes the method behave like Tikhonov regularization.

\subsection*{(f) Optional: discretize-then-optimize vs optimize-then-discretize}

Let $V_h\subset H^1(\Omega)$ be a finite element space with basis $\{\varphi_i\}_{i=1}^N$, and write
\[
u_h=\sum_{i=1}^N U_i\varphi_i.
\]
If we first discretize the functional, then $F^{\delta}_{\mathrm{TV}}$ becomes a finite-dimensional function of the coefficient vector $U\in\R^N$. Its gradient components are exactly the weak-form residuals obtained by testing with each basis function $\varphi_i$, and its Hessian entries are exactly the bilinear form
\[
D^2F^{\delta}_{\mathrm{TV}}(u_h)[\varphi_j,\varphi_i].
\]
Thus the finite-dimensional Newton step solves the linear system obtained by assembling the weak Newton form from part (c).

Conversely, if we first derive the infinite-dimensional weak Newton step and then restrict both $\tilde u$ and $\hat u$ to $V_h$, we obtain the same assembled matrix and right-hand side. Therefore the two procedures produce the same finite element equations. In this setting, discretize-then-optimize and optimize-then-discretize are equivalent.





\section*{Problem 2}

Done $\checkmark$

\section*{Problem 3(a)}

We consider the anisotropic Poisson problem
\[
-\nabla \cdot (A\nabla u) = f \quad \text{in } \Omega,
\qquad
u = u_0 \quad \text{on } \Gamma = \partial \Omega,
\]
where $A(x) \in \R^{2\times 2}$ is symmetric positive definite for each $x \in \Omega$.

Let
\[
V := \{v \in H^1(\Omega) : v|_{\Gamma} = 0\}
\]
be the space of test functions, and let the admissible set be
\[
U := \{w \in H^1(\Omega) : w|_{\Gamma} = u_0\}.
\]

To derive the weak form, multiply the strong form by a test function $\hat u \in V$ and integrate over $\Omega$:
\[
\int_\Omega \bigl(-\nabla \cdot (A\nabla u)\bigr)\,\hat u\,dx
= \int_\Omega f\,\hat u\,dx.
\]
Applying integration by parts gives
\[
\int_\Omega A\nabla u \cdot \nabla \hat u\,dx
- \int_{\Gamma} \hat u\,(A\nabla u \cdot n)\,ds
= \int_\Omega f\,\hat u\,dx.
\]
Since $\hat u|_{\Gamma}=0$, the boundary term vanishes, so the weak form is:
\[
\boxed{
\text{Find } u \in U \text{ such that }
\int_\Omega A\nabla u \cdot \nabla \hat u\,dx
= \int_\Omega f\,\hat u\,dx
\quad \text{for all } \hat u \in V.
}
\]

Equivalently, bringing everything to the left-hand side,
\[
\int_\Omega A\nabla u \cdot \nabla \hat u\,dx
- \int_\Omega f\,\hat u\,dx = 0
\qquad \forall \hat u \in V.
\]

Now we derive the corresponding energy functional. Consider
\[
J(u) := \frac12 \int_\Omega \nabla u \cdot A\nabla u\,dx - \int_\Omega fu\,dx,
\qquad u \in U.
\]
Because $A$ is symmetric positive definite, the quadratic term is coercive, so this is the natural anisotropic Dirichlet energy minus the work of the source.

To verify that its minimizer solves the weak form, take a variation $u + \epsilon \hat u$ with $\hat u \in V$:
\[
\frac{d}{d\epsilon} J(u+\epsilon \hat u)\Big|_{\epsilon=0}
= \frac12 \frac{d}{d\epsilon}
\int_\Omega \nabla(u+\epsilon \hat u)\cdot A\nabla(u+\epsilon \hat u)\,dx\Big|_{\epsilon=0}
- \frac{d}{d\epsilon}\int_\Omega f(u+\epsilon \hat u)\,dx\Big|_{\epsilon=0}.
\]
Using symmetry of $A$,
\[
\frac{d}{d\epsilon} J(u+\epsilon \hat u)\Big|_{\epsilon=0}
= \int_\Omega A\nabla u \cdot \nabla \hat u\,dx - \int_\Omega f\hat u\,dx.
\]
Thus the first-order optimality condition $\delta J(u;\hat u)=0$ for all $\hat u \in V$ is exactly the weak form above.

Therefore, the energy functional minimized by the solution of $(1)$ is
\[
\boxed{
J(u) = \frac12 \int_\Omega \nabla u \cdot A\nabla u\,dx - \int_\Omega fu\,dx,
\qquad u \in U = \{w \in H^1(\Omega): w|_{\Gamma}=u_0\}.
}
\]

\section*{Problem 3(b)}

Using quadratic finite elements on the mesh loaded from \texttt{circle.xml}, I solved the anisotropic
Poisson problem for the two conductivity tensors
\[
A_1 = \begin{pmatrix} 10 & 0 \\ 0 & 10 \end{pmatrix},
\qquad
A_2 = \begin{pmatrix} 1 & -5 \\ -5 & 100 \end{pmatrix},
\]
with source term
\[
f(x,y) = \exp\bigl(-100(x^2+y^2)\bigr),
\qquad
u_0 = 0 \text{ on } \Gamma.
\]
For $A_1$, the solution is radially symmetric, as expected from the radial source and isotropic
conductivity. For $A_2$, the solution is distorted by the anisotropy of the tensor, so the spreading is
stronger in preferred directions. The difference plot $u_{A_2}-u_{A_1}$ highlights this change.

\begin{figure}[H]
    \centering
    \includegraphics[width=\textwidth]{figs/problem3_results.png}
    \caption{Finite element solutions of the anisotropic Poisson problem on the unit disk for the two
    conductivity tensors $A_1$ and $A_2$, together with the difference $u_{A_2}-u_{A_1}$.}
    \label{fig:problem3b}
\end{figure}

The computed extrema were:

{\footnotesize
\begin{verbatim}
A1 solution:
  min = 0.0
  max = 0.0012956269729364432

A2 solution:
  min = 0.0
  max = 0.0006365860132383044

Difference u_A2 - u_A1:
  min = -0.000777516422055416
  max = 8.897542992577181e-06
\end{verbatim}
}

These numerical results are consistent with the plots. The solution corresponding to the isotropic tensor
$A_1$ attains a larger peak value, while the anisotropic tensor $A_2$ spreads the response more strongly
along preferred directions and therefore lowers the maximum value. The difference field is negative over
most of the domain, as shown by the substantially negative minimum of $u_{A_2}-u_{A_1}$, with only a
very small positive maximum. This indicates that the anisotropic conductivity in $A_2$ redistributes the
solution away from the isotropic profile produced by $A_1$.

The FEniCS code used for this computation is shown below.

{\footnotesize
\begin{verbatim}
# Solve Problem 3(b) with quadratic finite elements and compare A1 vs A2

V = dl.FunctionSpace(mesh, "Lagrange", 2)
bc = dl.DirichletBC(V, u_0, "on_boundary")

def solve_problem(A, name):
    u = dl.TrialFunction(V)
    v = dl.TestFunction(V)

    a = ufl.inner(A * ufl.grad(u), ufl.grad(v)) * dl.dx
    L = f * v * dl.dx

    uh = dl.Function(V, name=name)
    dl.solve(a == L, uh, bc)
    return uh

u_A1 = solve_problem(A1, "u_A1")
u_A2 = solve_problem(A2, "u_A2")

# Difference field
u_diff = dl.project(u_A2 - u_A1, V)

# Basic summary statistics
vals1 = u_A1.vector().get_local()
vals2 = u_A2.vector().get_local()
vals_diff = u_diff.vector().get_local()

print("A1 solution:")
print("  min =", vals1.min())
print("  max =", vals1.max())
print()
print("A2 solution:")
print("  min =", vals2.min())
print("  max =", vals2.max())
print()
print("Difference u_A2 - u_A1:")
print("  min =", vals_diff.min())
print("  max =", vals_diff.max())

# Use common color scale for the two solutions
sol_min = min(vals1.min(), vals2.min())
sol_max = max(vals1.max(), vals2.max())

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
p1 = dl.plot(u_A1)
p1.set_clim(sol_min, sol_max)
plt.colorbar(p1)
plt.title("Solution with A1 = [[10, 0], [0, 10]]")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 3, 2)
p2 = dl.plot(u_A2)
p2.set_clim(sol_min, sol_max)
plt.colorbar(p2)
plt.title("Solution with A2 = [[1, -5], [-5, 100]]")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(1, 3, 3)
p3 = dl.plot(u_diff)
plt.colorbar(p3)
plt.title("Difference u_A2 - u_A1")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.savefig("problem3_results.png", dpi=300, bbox_inches="tight")
plt.show()
\end{verbatim}
}

\end{document}
