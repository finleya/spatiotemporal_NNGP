rm(list = ls())

dyn.load("../../mk_nn_index/nn.so")
source("../../mk_nn_index/nn.R")
source("../../mk_nn_index/util.R")
source("../../mk_nn_index/mk_nn_index.R")

## dyn.load("../libs/cNSCovOMP.so")
## source("../libs/cNSCovOMP.R")

library(rgl)
library(dplyr)
library(MBA)
library(fields)
library(viridis)
library(GpGp)

rmvn <- function(n, mu=0, V = matrix(1)){
  p <- length(mu)
  if(any(is.na(match(dim(V),p))))
    stop("Dimension problem!")
  D <- chol(V)
  t(matrix(rnorm(n*p), ncol=p)%*%D + rep(mu,rep(n,p)))
}

## Make some coordinates and order them for NNGP.
set.seed(1)

coords <- expand.grid(seq(0,1,0.05), seq(0,1,0.05), seq(0,1,0.05))
coords <- tibble(x = coords[,1], y = coords[,2], z = coords[,3]) %>% arrange(z, x)
coords <- coords %>% as.matrix()

## For these data, the space between time points at a given location is larger than spacing between spatial locations. As a result, most neighbors to a given location are in the same time. So to get a few temporal neighbors I scale the time dimension a bit. This is only used for creating the neighbor set.
#coords.scaled <- coords

#min.sp <- 0.05
#min.t <- 0.1

#coords.scaled[,3] <- min.sp/min.t*coords[,3]

ord <- order_maxmin(coords, space_time = TRUE)
coords <- coords[ord,]

## Make the various neighbor vectors.
neighbor.indx <- mk_nn_index(coords, n.neighbors = 15)

## Check out the ordering and neighbors, just for fun.
n.indx <- neighbor.indx$n.indx #see spNNGP manual for description of n.indx.

i <- nrow(coords)
spheres3d(coords[-i,], col="gray", radius=0.001)
spheres3d(coords[i,,drop=FALSE], col="blue", radius=0.01)
spheres3d(coords[n.indx[[i]],,drop=FALSE], col="red", radius=0.01)

## Get the various indexes ready to read into the c code.
nn.indx <- neighbor.indx$nn.indx
nn.indx.lu <- neighbor.indx$nn.indx.lu
u.indx <- neighbor.indx$u.indx
u.indx.lu <- neighbor.indx$u.indx.lu
ui.indx <- neighbor.indx$ui.indx

## Write the various neighbor vectors out for c code.
system("rm nn.indx.* ui.indx.* u.indx.*")

write.table(nn.indx, paste0("nn.indx.",length(nn.indx)), row.names=F, col.names=F)
write.table(nn.indx.lu, paste0("nn.indx.lu.",length(nn.indx.lu)), row.names=F, col.names=F)
write.table(u.indx, paste0("u.indx.",length(u.indx)), row.names=F, col.names=F)
write.table(u.indx.lu, paste0("u.indx.lu.",length(u.indx.lu)), row.names=F, col.names=F)
write.table(ui.indx, paste0("ui.indx.",length(ui.indx)), row.names=F, col.names=F)

## Make some data.
n <- nrow(coords)
n

x <- cbind(1, rnorm(n))

beta <- as.matrix(c(1,5))

sigma.sq <- 5
tau.sq <- 0.1

## This alpha and gamma are the spatial and temporal ranges, respectively. See the code at the end of this script to help visualize the ranges (given the mixture term kappa).
alpha <- 3/0.5
gamma <- 3/0.5

## Makes the non-separable covariance matrix.
#C <- c.ns.cov.omp(coords, coords, sigma.sq = sigma.sq, tau.sq = 0,
#                  alpha = alpha, gamma = gamma, kappa = kappa, inv = FALSE, n.omp.threads = 20)$C

C <- sigma.sq*exp(-alpha*as.matrix(dist(coords[,3])))*exp(-gamma*as.matrix(dist(coords[,1:2])))

w <- rmvn(1, rep(0, n), C)

## Check them out.
colors <- viridis(100)[as.numeric(cut(w, breaks = 100))]
spheres3d(coords, col=colors, radius=0.05)
axes3d() 

y <- rnorm(n, x%*%beta + w, sqrt(tau.sq))

options(scipen = 100, digits = 4)
write.table(y, "y", row.names=F, col.names=F, sep="\t")
write.table(x, "X", row.names=F, col.names=F, sep="\t")
write.table(w, "w", row.names=F, col.names=F, sep="\t")
write.table(coords, "coords", row.names=F, col.names=F, sep="\t")

params <- c("beta.0" = beta[1], "beta.1" = beta[2],
            "alpha" = alpha, "gamma" = gamma, 
            "sigma.sq" = sigma.sq, "tau.sq" = tau.sq)

save(params, file = "params.rds")


## Some code useful for visualizing the space-time correlation surface.
ns.cor <- function(v){
    h <- v[1] #space, gamma
    u <- v[2] #time, alpha
    exp(-alpha*u)*exp(-gamma*h)
}

v <- expand.grid(h = seq(0, 1, by = 0.01), u = seq(0, 1, by = 0.01))

cor <- apply(v, 1, ns.cor)

surf <- mba.surf(cbind(v, cor), no.X = 100, no.Y = 100)$xyz.est
image.plot(surf, xlab = "Space", ylab = "time", zlim = c(0,1))
contour(surf, add = TRUE)
