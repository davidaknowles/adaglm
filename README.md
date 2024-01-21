## adaglm

Fits a negative binomial (NB) generalized linear model (GLM) using Adam. Since Adam supports minibatching `adaglm` can run on very large datasets (millions of datapoints). By default the fitting is done in two steps: 1) Fit a Poisson GLM. 2) Starting with coefficients initialized from the Poisson GLM, fit a NB GLM jointly optimizing the coefficients and concentration.

See `vignettes/negbin_glm.Rmd` for example usage. 
