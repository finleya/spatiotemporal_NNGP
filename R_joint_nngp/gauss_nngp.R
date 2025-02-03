rm(list=ls())
dyn.load("../mk_nn_index/nn.so")
source("../mk_nn_index/nn.R")
source("../mk_nn_index/util.R")
library(fields)
library(viridis)
library(Matrix)
library(coda)
library(GpGp)
library(RhpcBLASctl)

## For the nonseparable model.

#Sys.setenv(OPENBLAS_NUM_THREADS = 10)
#Sys.setenv(OMP_NUM_THREADS = 10)

omp_set_num_threads(5)

dyn.load("../libs/cNSCovOMP.so")
source("../libs/cNSCovOMP.R")

## Some useful functions.
rmvn <- function(n, mu=0, V = matrix(1)){
  p <- length(mu)
  if(any(is.na(match(dim(V),p))))
    stop("Dimension problem!")
  D <- chol(V)
  t(matrix(rnorm(n*p), ncol=p)%*%D + rep(mu,rep(n,p)))
}

rinvgamma <- function(n, shape, scale){
    1/rgamma(n, shape = shape, rate = scale)
}

logit <- function(theta, a, b){log((theta-a)/(b-theta))}

logit.inv <- function(z, a, b){b-(b-a)/(1+exp(z))}

## Get data.
set.seed(1)

y <- read.table("../sim_data/non_sep/y")[,1]
x <- as.matrix(read.table("../sim_data/non_sep/X"))
coords <- as.matrix(read.table("../sim_data/non_sep/coords"))
w <- read.table("../sim_data/non_sep/w")[,1]
n <- nrow(coords)
p <- ncol(x)
  
## Sampler.
n.iter <- 500

beta.samples <- matrix(0, n.iter, p)
w.samples <- matrix(0, n, n.iter)
sigma.sq.samples <- rep(0, n.iter)
alpha.samples <- rep(0, n.iter)
gamma.samples <- rep(0, n.iter)
kappa.samples <- rep(0, n.iter)
tau.sq.samples <- rep(0, n.iter)

## Priors.
sigma.sq.b <- 5
tau.sq.b <- 1
alpha.a <- 1
alpha.b <- 10
gamma.a <- 1
gamma.b <- 10
kappa.a <- 0
kappa.b <- 1

## Tuning.
alpha.tuning <- 0.01
gamma.tuning <- 0.01
kappa.tuning <- 0

## NNGP stuff.
m <- 15

A <- matrix(0, n, n)
D <- rep(0, n)
nn <- mkNNIndx(coords, m)
nn.indx <- mk.n.indx.list(nn$nnIndx, n, m)

## Starting and other stuff.
xx <- t(x)%*%x
beta.s <- coef(lm(y ~ x-1))
w.s <- y - x%*%beta.s
sigma.sq.s <- 1
tau.sq.s <- 1
alpha.s <- 5
gamma.s <- 5
kappa.s <- 0.5

batch.iter <- 0
batch.length <- 5
batch.accept <- 0
I <- Diagonal(n, x = 1)

## Collect samples.
for(s in 1:n.iter){

    ## Update beta.
    V <- chol2inv(chol(xx/tau.sq.s))
    v <- t(x)%*%(y - w.s)/tau.sq.s
    beta.s <- rmvn(1, V%*%v, V)

    ## Update w.
    A <- matrix(0, n, n)
    D <- rep(0, n)
    for(i in 1:(n-1)){

        a <- c.ns.cov.omp(coords[nn.indx[[i+1]],,drop=FALSE], coords[nn.indx[[i+1]],,drop=FALSE], sigma.sq = sigma.sq.s, tau.sq = 0,
                     alpha = alpha.s, gamma = gamma.s, kappa = kappa.s, inv = FALSE, n.omp.threads = 1)$C

        b <- c.ns.cov.omp(coords[nn.indx[[i+1]],,drop=FALSE], coords[i+1,,drop=FALSE], sigma.sq = sigma.sq.s, tau.sq = 0,
                     alpha = alpha.s, gamma = gamma.s, kappa = kappa.s, inv = FALSE, n.omp.threads = 1)$C
        
        A[i+1, nn.indx[[i+1]]] <- solve(a, b)

        a <- c.ns.cov.omp(coords[i+1,,drop=FALSE], coords[nn.indx[[i+1]],,drop=FALSE], sigma.sq = sigma.sq.s, tau.sq = 0,
                          alpha = alpha.s, gamma = gamma.s, kappa = kappa.s, inv = FALSE, n.omp.threads = 1)$C

        
        D[i+1] <- sigma.sq.s - a %*% A[i+1,nn.indx[[i+1]]]

        ## A[i+1, nn.indx[[i+1]]] <- solve(sigma.sq.s*exp(-phi.cand*rdist(coords[nn.indx[[i+1]],,drop=FALSE])),
        ##                                 sigma.sq.s*exp(-phi.cand*rdist(coords[nn.indx[[i+1]],,drop=FALSE], coords[i+1,,drop=FALSE])))
        
        ## D[i+1] <- sigma.sq.s - sigma.sq.s*exp(-phi.cand*rdist(coords[i+1,,drop=FALSE], coords[nn.indx[[i+1]],,drop=FALSE])) %*% A[i+1,nn.indx[[i+1]]]
    }

    D[1] <- sigma.sq.s
    A <- Matrix(A, sparse = TRUE)
    
    C.log.det <- sum(log(D)) ## Get the log det for updating phi later on.
    
    D.inv <- Diagonal(n, x = 1/D)
    
    C.inv <- t(I - A)%*%D.inv%*%(I - A)

    V <- C.inv + Diagonal(x = 1/tau.sq.s, n = n)
    v <- (y - x%*%beta.s)/tau.sq.s
    chl <- Cholesky(V, perm = TRUE)
    L <- expand1(chl, "L")
    P <- expand1(chl, "P1")
    w.s <- (t(P)%*%solve(t(L), solve(L, P%*%v) + rnorm(n)))[,1]
    
    ## Update phi.
    ## Current.
    current.ltd <- as.numeric(-0.5*C.log.det-0.5*(t(w.s)%*%C.inv%*%w.s) +
                              log(alpha.s - alpha.a) + log(alpha.b - alpha.s) +
                              log(gamma.s - gamma.a) + log(gamma.b - gamma.s) +
                              log(kappa.s - kappa.a) + log(kappa.b - kappa.s))
    
    ## Candidate.
    alpha.cand <- logit.inv(rnorm(1, logit(alpha.s, alpha.a, alpha.b), sqrt(alpha.tuning)), alpha.a, alpha.b)
    gamma.cand <- logit.inv(rnorm(1, logit(gamma.s, gamma.a, gamma.b), sqrt(gamma.tuning)), gamma.a, gamma.b)
    kappa.cand <- logit.inv(rnorm(1, logit(kappa.s, kappa.a, kappa.b), sqrt(kappa.tuning)), kappa.a, kappa.b)

    A <- matrix(0, n, n)
    D <- rep(0, n)
    for(i in 1:(n-1)){

        a <- c.ns.cov.omp(coords[nn.indx[[i+1]],,drop=FALSE], coords[nn.indx[[i+1]],,drop=FALSE], sigma.sq = sigma.sq.s, tau.sq = 0,
                     alpha = alpha.cand, gamma = gamma.cand, kappa = kappa.cand, inv = FALSE, n.omp.threads = 1)$C

        b <- c.ns.cov.omp(coords[nn.indx[[i+1]],,drop=FALSE], coords[i+1,,drop=FALSE], sigma.sq = sigma.sq.s, tau.sq = 0,
                     alpha = alpha.cand, gamma = gamma.cand, kappa = kappa.cand, inv = FALSE, n.omp.threads = 1)$C
        
        A[i+1, nn.indx[[i+1]]] <- solve(a, b)


        a <- c.ns.cov.omp(coords[i+1,,drop=FALSE], coords[nn.indx[[i+1]],,drop=FALSE], sigma.sq = sigma.sq.s, tau.sq = 0,
                          alpha = alpha.cand, gamma = gamma.cand, kappa = kappa.cand, inv = FALSE, n.omp.threads = 1)$C

        
        D[i+1] <- sigma.sq.s - a %*% A[i+1,nn.indx[[i+1]]]
        
        ## A[i+1, nn.indx[[i+1]]] <- solve(sigma.sq.s*exp(-phi.cand*rdist(coords[nn.indx[[i+1]],,drop=FALSE])),
        ##                                 sigma.sq.s*exp(-phi.cand*rdist(coords[nn.indx[[i+1]],,drop=FALSE], coords[i+1,,drop=FALSE])))
        
        ## D[i+1] <- sigma.sq.s - sigma.sq.s*exp(-phi.cand*rdist(coords[i+1,,drop=FALSE], coords[nn.indx[[i+1]],,drop=FALSE])) %*% A[i+1,nn.indx[[i+1]]]
    }

    D[1] <- sigma.sq.s

    A <- Matrix(A, sparse = TRUE)

    C.log.det.cand  <- sum(log(D)) ## Get the log det for updating phi later on.
    
    D.inv <- Diagonal(n, x = 1/D)
    
    C.inv.cand  <- t(I - A)%*%D.inv%*%(I - A)
    

    cand.ltd <- as.numeric(-0.5*C.log.det.cand-0.5*(t(w.s)%*%C.inv.cand%*%w.s) + 
                           log(alpha.cand - alpha.a) + log(alpha.b - alpha.cand) +
                           log(gamma.cand - gamma.a) + log(gamma.b - gamma.cand) +
                           log(kappa.cand - kappa.a) + log(kappa.b - kappa.cand))
    
    if(runif(1,0,1) < exp(cand.ltd-current.ltd)){
        alpha.s <- alpha.cand
        gamma.s <- gamma.cand
        kappa.s <- kappa.cand

        batch.accept <- batch.accept+1
    }

    ## Update sigma.sq.
    sigma.sq.s <- rinvgamma(1, 2 + n/2, sigma.sq.b + as.numeric(t(w.s)%*%(C.inv*sigma.sq.s)%*%w.s)/2)

    ## Update tau.sq.
    tau.sq.s <- rinvgamma(1, 2 + n/2, tau.sq.b + sum((y - x%*%beta.s - w.s)^2)/2)
    
    ## Save samples.
    beta.samples[s,] <- beta.s
    w.samples[,s] <- w.s
    sigma.sq.samples[s] <- sigma.sq.s
    alpha.samples[s] <- alpha.s
    gamma.samples[s] <- gamma.s
    kappa.samples[s] <- kappa.s
    tau.sq.samples[s] <- tau.sq.s

    ## Progress and reporting.
    batch.iter <- batch.iter + 1
    
    if(batch.iter == batch.length){
        print(paste("Complete:",round(100*s/n.iter)))
        print(paste("Metrop acceptance:", 100*batch.accept/batch.length))
        print("---------")
        batch.iter <- 0
        batch.accept <- 0
    }
    
}


burn.in <- 1
plot(mcmc(beta.samples), density=FALSE)

w.mu <- apply(w.samples[,burn.in:n.iter], 1, mean)
par(mfrow=c(2,2))
plot(w, w.mu)
plot(mcmc(sigma.sq.samples), density=FALSE)
plot(mcmc(tau.sq.samples[20:n.iter]), density=FALSE)
plot(mcmc(alpha.samples), density=FALSE)
plot(mcmc(gamma.samples), density=FALSE)
plot(mcmc(kappa.samples), density=FALSE)
