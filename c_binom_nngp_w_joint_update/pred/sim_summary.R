rm(list=ls())
library(coda)
library(tidyverse)
library(MBA)
library(fields)

w <- read.table(paste0("../sim_data/w.ho"))[,1]
y <- read.table(paste0("../sim_data/y.ho"))[,1]

w.s <- read_table("ppd-samps-w0", col_names = FALSE)
w.mu <- apply(w.s, 1, mean)
plot(w, w.mu)
lines(-10:10, -10:10)

p.s <- as.matrix(read_table("ppd-samps-p0", col_names = FALSE))

y.s <- matrix(rbinom(length(p.s), prob = p.s, size = 1), nrow = nrow(p.s))

y.mu <- apply(y.s, 1, mean)

table(y, round(y.mu))
