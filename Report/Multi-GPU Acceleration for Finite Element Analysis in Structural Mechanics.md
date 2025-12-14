# Multi-GPU Acceleration for Finite Element Analysis in Structural Mechanics

   

## 1. Introduction

The increasing complexity of engineering systems and the demand for accurate numerical predictions have made computational simulation an indispensable tool in modern engineering and applied sciences. Among numerical techniques, the **Finite Element Method (FEM)** stands out as one of the most robust and versatile approaches for solving boundary-value problems governed by partial differential equations (PDEs).

Despite its conceptual elegance and widespread adoption, FEM is computationally expensive. As model fidelity increases—through mesh refinement or complex geometries—the size of the resulting algebraic systems grows rapidly, leading to significant computational costs.

Recent advances in **Graphics Processing Units (GPUs)** have enabled massive data-parallel execution, making them particularly suitable for accelerating FEM workloads. Within this context, this project explores the development of a **GPU-accelerated FEM solver**, progressing from a CPU baseline to single-GPU and multi-GPU execution.

   

## 2. The Finite Element Method (FEM)

### 2.1 Conceptual Foundations

The Finite Element Method is a numerical discretisation technique that approximates the solution of PDEs by decomposing a continuous domain into smaller subdomains, known as **elements**, interconnected at **nodes**. The unknown field variables are approximated using polynomial **shape functions**.

Through a variational formulation, the governing equations are transformed into a system of algebraic equations of the form:

\[
\mathbf{K}\mathbf{u} = \mathbf{f}
\]

where \(\mathbf{K}\) is the global stiffness matrix, \(\mathbf{u}\) the vector of nodal unknowns, and \(\mathbf{f}\) the load vector.

   

## 3. Types of Problems Addressed by FEM

FEM can be applied to a wide range of physical problems, including:

- Linear static structural problems  
- Dynamic and modal analysis  
- Thermo-mechanical and multi-physics problems  
- Potential problems governed by the Poisson or Laplace equations  

In this work, particular emphasis is placed on **potential problems**, which are mathematically well-posed and computationally suitable for performance benchmarking.

   

## 4. Computational Motivation for GPU and Multi-GPU Acceleration

FEM simulations are dominated by sparse linear algebra operations, especially **sparse matrix–vector multiplications (SpMV)** and iterative solvers. These operations exhibit a high degree of data parallelism, making them well suited for execution on GPUs under the **SIMT** model.

While single-GPU acceleration provides substantial speedups, large-scale problems and parametric studies benefit from **multi-GPU execution**, enabling increased scalability and reduced time-to-solution.

   

## 5. Case Study: Potential Flow in a Channel with Geometric Constrictions

### 5.1 Physical Problem Description

The specific case study analysed in this work consists of a two-dimensional potential flow of an incompressible and inviscid fluid inside a channel with geometric constrictions formed by circular obstacles. The channel is assumed to be infinite in the out-of-plane direction, allowing for a 2D formulation.

A uniform inflow velocity is imposed at the inlet, and the objective is to determine:

- the velocity potential distribution,
- the velocity field,
- the pressure distribution,
- the maximum and minimum values of velocity and pressure and their locations.

This configuration is representative of internal flows in ducts and channels and provides a demanding benchmark due to the presence of strong velocity gradients near the constrictions.

### 5.2 Mathematical Formulation

The flow is described using the velocity potential \(\phi(x,y)\), such that:

\[
u = \frac{\partial \phi}{\partial x}, \quad
v = \frac{\partial \phi}{\partial y}
\]

Assuming incompressibility and irrotational flow, the governing equation reduces to the **Laplace equation:

\[
\nabla^2 \phi = 0 \quad \text{in } \Omega
\]



The spatial discretisation is performed using the Galerkin finite element method, leading to a linear system:

\[
\mathbf{K}\boldsymbol{\phi} = \mathbf{F}
\]

   

### 5.3 Boundary Conditions

The problem involves mixed boundary conditions, defined as follows:

#### Inlet Boundary (Neumann)

At the inlet section of the channel, a uniform normal velocity is imposed:

\[
\frac{\partial \phi}{\partial n} = V_0 = 2.5 \; \text{m/s}
\]

This corresponds to a natural (Neumann) boundary condition.

#### Outlet Boundary (Dirichlet)

At the outlet sections, the velocity potential is set to zero:

\[
\phi = 0
\]

This essential (Dirichlet) boundary condition defines a reference level for the potential.

#### Wall Boundaries

Along all solid boundaries of the domain, including the circular constrictions, a no-penetration condition is enforced. This condition states that the normal component of the velocity vanishes at the walls, which in terms of the velocity potential can be written as:

\[
\frac{\partial \phi}{\partial n} = 0
\]

This boundary condition guarantees that the fluid remains confined within the channel and that no mass flux occurs across the solid surfaces.

### 5.4 Mesh Refinement Strategy

To accurately capture high-gradient regions near the geometric constrictions, a locally refined mesh is employed in the central region of the channel, while coarser elements are used elsewhere.

Mesh refinement leads to a significant increase in the number of degrees of freedom, making this case particularly suitable for evaluating the performance and scalability of GPU and multi-GPU FEM implementations.

   

## 6. Relevance to High Performance Graphical Computing

This project aligns with the objectives of High Performance Graphical Computing by:

- applying GPU acceleration to a realistic FEM problem,
- analysing solver scalability under mesh refinement,
- comparing CPU, single-GPU, and multi-GPU implementations,
- integrating numerical accuracy with performance benchmarking.

By combining a classical potential flow problem with modern parallel computing techniques, this work bridges traditional computational mechanics and contemporary high-performance hardware architectures.

   
