# The Thermal Sunyaev-Zel’dovich (tSZ) Power Spectrum 

This repo contains the code for computing the tSZ power spectrum for dark matter halos.  
The thermal Sunyaev–Zel’dovich effect arises when CMB photons inverse-Compton scatter off hot electrons in the intracluster medium (ICM) of galaxy clusters. The tSZ spectrum is a plot showing the frequency dependence of the thermal Sunyaev–Zel’dovich (tSZ) effect on the CMB.

## Data needed
1. The dark matter lightcone particle shells at all time steps (or redshifts)
2. The dark matter halos at all time steps

The dark matter lightcone particle shells are needed to compute the $M_{500}$, which is the total mass enclosed within $R_{500}$, the radius at which the mean enclosed density equals 500 times the critical density of the Universe. The dark matter halo locations and masses can then be used to compute the tSZ power spectrum.


## Compton-y Map (Orthographic Projection)

![Compton-y Map](Images/y_map_orthographic.png)


## Thermal Sunyaev-Zel’dovich (tSZ) Power Spectrum

![tSZ Power Spectrum](Images/tSZ_spectrum.png)


---

## About

This repository contains routines for analyzing cosmological simulations and observational data, including:

- Compton y parameter map generation
- tSZ spectrum computation
