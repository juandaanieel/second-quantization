---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Tutorial

Welcome to the `second_quantization` package tutorial! This section contains comprehensive examples demonstrating all the functionality of the package through practical physics applications.

```{toctree}
:maxdepth: 1
:caption: Tutorial Examples
basic_pmm.md

```

## Tutorial Overview

In this tutorial we will use the package to explore the poor man's Kitaev chain model, a simplified system that can host Majorana-like modes using conventional superconducting quantum dots.
The goal is to demonstrate how to use the `second_quantization` package for symbolic operator algebra, Hamiltonian construction, and spectroscopic analysis of quantum many-body systems.
This tutorial consists of three main parts:

- **Model definition and numerical analysis**: We will define the model Hamiltonian using symbolic fermionic operators, construct the corresponding matrix representation, and analyze its spectrum.
- **Analysis of Majorana modes**: We will build the symbolic Majorana operators in the system and analyze their properties using the package's built-in functions.
- **Perturbation theory with `pymablock`**: We will demonstrate how to integrate our package with the `pymablock` package to perform symbolic perturbation theory calculations.
