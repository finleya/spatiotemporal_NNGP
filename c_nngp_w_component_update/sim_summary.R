rm(list=ls())
library(coda)
library(tidyverse)
library(MBA)
library(fields)

sep.type <- "sep"

load(paste0("../sim_data/",sep.type,"/params.rds"))
beta.0 <- params["beta.0"]
beta.1 <- params["beta.1"]
alpha <- params["alpha"]
gamma <- params["gamma"]
kappa <- params["kappa"]
sigma.sq <- params["sigma.sq"]
tau.sq <- params["tau.sq"]

w <- read.table(paste0("../sim_data/",sep.type,"/w"))[,1]
y <- read.table(paste0("../sim_data/",sep.type,"/y"))[,1]
p <- 2

## Check MCMC samples.
beta.s <- mcmc(matrix(scan("chain-1-beta"), ncol = p, byrow = FALSE))
colnames(beta.s) <- paste0("beta.",1:p)

summary(beta.s)

w.s <- read_table("chain-1-w", col_names = FALSE)
w.mu <- apply(w.s, 1, median)
plot(w, w.mu)
lines(-10:10, -10:10)

alpha.s <- mcmc(matrix(scan("chain-1-alpha"), ncol = 1, byrow = FALSE))
summary(alpha.s)
plot(alpha.s)

gamma.s <- mcmc(matrix(scan("chain-1-gamma"), ncol = 1, byrow = FALSE))
summary(gamma.s)
plot(gamma.s)

kappa.s <- mcmc(matrix(scan("chain-1-kappa"), ncol = 1, byrow = FALSE))
summary(kappa.s)
plot(kappa.s)

sigma.sq.s <- mcmc(matrix(scan("chain-1-sigmaSq"), ncol = 1, byrow = FALSE))
summary(sigma.sq.s)
plot(sigma.sq.s)
sigma.sq

tau.sq.s <- mcmc(matrix(scan("chain-1-tauSq"), ncol = 1, byrow = FALSE))
summary(tau.sq.s)
tau.sq
