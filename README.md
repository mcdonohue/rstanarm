[<img src="https://raw.githubusercontent.com/stan-dev/logos/master/logo_tm.png" width=100 alt="Stan Logo"/>](http://mc-stan.org)

# Latent Time Joint Mixed Effect Models (LTJMM)

[![Build Status](https://travis-ci.org/mcdonoue/rstanarm.svg?branch=master)](https://travis-ci.org/mcdonoue/rstanarm)[![CRAN\_Status\_Badge](http://www.r-pkg.org/badges/version/rstanarm?color=blue)](http://cran.r-project.org/package=rstanarm)[![Downloads](http://cranlogs.r-pkg.org/badges/rstanarm?color=blue)](http://cran.rstudio.com/package=rstanarm)

### Latent Time Joint Mixed Effect Models (LTJMM) via rstanarm

This fork of the [rstanarm](https://github.com/stan-dev/rstanarm) package includes the following modifications:

1. **stan_mvmer** models are extended from 5 to 20 longitudinal submodels
2. **stan_ljtmm** extends **stan_mvmer** to accommodate a group-specific (individual-specific) latent time parameters which are shared within group across submodels.

The LTJMM is described in [Li, et al. (2017)](https://doi.org/10.1177/0962280217737566).

### Installation

To install from GitHub, first make sure that you can install the **rstan**
package and C++ toolchain by following these
[instructions](https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started).
Once **rstan** is successfully installed, you can install **rstanarm** from
GitHub using the **remotes** package by executing the following in R:

```r
Sys.setenv(MAKEFLAGS = "-j4") # change 4 to however many cores you can/want to use to parallelize install 
Sys.setenv("R_REMOTES_NO_ERRORS_FROM_WARNINGS" = "true")
remotes::install_github("mcdonohue/rstanarm", INSTALL_opts = "--no-multiarch", 
  force = TRUE, build_vignettes = FALSE)
```

## Reference

* Li, D., Iddi, S., Thompson, W. K., Donohue, M. C., for ADNI. (2017). Bayesian latent time joint mixed effect models for multicohort longitudinal data. *Statistical methods in medical research*. [https://doi.org/10.1177/0962280217737566](https://doi.org/10.1177/0962280217737566)