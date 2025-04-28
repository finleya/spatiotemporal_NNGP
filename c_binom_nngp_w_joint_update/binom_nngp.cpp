#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <limits>
#include <omp.h>
#include <suitesparse/cholmod.h>
using namespace std;

#include "../libs/kvpar.h"

#define MATHLIB_STANDALONE
#include <Rmath.h>
#include <Rinternals.h>

#include <time.h>
#include <sys/time.h>
#define CPUTIME (SuiteSparse_time ( ))

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //Handle error.
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

double get_cpu_time(){
    return (double)clock() / CLOCKS_PER_SEC;
}

extern "C" {
  extern void dgemm_(const char *transa, const char *transb,
		    const int *m, const int *n, const int *k,
		    const double *alpha, const double *a,
		    const int *lda, const double *b, const int *ldb,
		    const double *beta, double *c, const int *ldc);
   
  extern void  dcopy_(const int *n, const double *dx, const int *incx, double *dy, const int *incy);
  
  extern int dpotrf_(const char *uplo, int *n, double *a, int *lda, int *info);

  extern int dpotri_(const char *uplo, int *n, double *a, int *lda, int *info);

  extern void dsymm_(const char *side, const char *uplo, const int *m,
		     const int *n, const double *alpha,
		     const double *a, const int *lda,
		     const double *b, const int *ldb,
		     const double *beta, double *c, const int *ldc);

  extern double ddot_(int *n, double *x, int *incx, double *y, int *incy);

  extern void dgemv_(const char *trans, const int *m, const int *n, const double *alpha,
		     const double *a, const int *lda, const double *x, const int *incx,
		     const double *beta, double *y, const int *incy);
  
  extern void dsymv_(const char *uplo, const int *n, const double *alpha, const double *a, const int *lda,
		    const double *x, const int *incx, const double *beta, double *y, const int *incy);

  extern void daxpy_(const int *n, const double *alpha, const double *x, const int *incx, double *y, const int *incy);

  extern void dtrmv_(const char *uplo, const char *transa, const char *diag, const int *n,
		     const double *a, const int *lda, double *b, const int *incx);

}

///////////////////////////////////////////////////////
//rpg stuff

// Mathematical constants
#define MATH_PI        3.141592653589793238462643383279502884197169399375105820974
#define MATH_PI_2      1.570796326794896619231321691639751442098584699687552910487
#define MATH_2_PI      0.636619772367581343075535053490057448137838582961825794990
#define MATH_PI2       9.869604401089358618834490999876151135313699407240790626413
#define MATH_PI2_2     4.934802200544679309417245499938075567656849703620395313206
#define MATH_SQRT1_2   0.707106781186547524400844362104849039284835937688474036588
#define MATH_SQRT_PI_2 1.253314137315500251207882642405522626503493370304969158314
#define MATH_LOG_PI    1.144729885849400174143427351353058711647294812915311571513
#define MATH_LOG_2_PI  -0.45158270528945486472619522989488214357179467855505631739
#define MATH_LOG_PI_2  0.451582705289454864726195229894882143571794678555056317392

// Generate exponential distribution random variates
double exprnd(double mu){
  return -mu * (double)std::log(1.0 - (double)runif(0.0,1.0));
}

// Function a_n(x) defined in equations (12) and (13) of
// Bayesian inference for logistic models using Polya-Gamma latent variables
// Nicholas G. Polson, James G. Scott, Jesse Windle
// arXiv:1205.0310
//
// Also found in the PhD thesis of Windle (2013) in equations
// (2.14) and (2.15), page 24
double aterm(int n, double x, double t){
  double f = 0;
  if(x <= t) {
    f = MATH_LOG_PI + (double)std::log(n + 0.5) + 1.5*(MATH_LOG_2_PI- (double)std::log(x)) - 2*(n + 0.5)*(n + 0.5)/x;
  }
  else {
    f = MATH_LOG_PI + (double)std::log(n + 0.5) - x * MATH_PI2_2 * (n + 0.5)*(n + 0.5);
  }    
  return (double)exp(f);
}


// Generate inverse gaussian random variates
double randinvg(double mu){
  // sampling
  double u = rnorm(0.0,1.0);
  double V = u*u;
  double out = mu + 0.5*mu * ( mu*V - (double)std::sqrt(4.0*mu*V + mu*mu * V*V) );
  
  if(runif(0.0,1.0) > mu /(mu+out)) {    
    out = mu*mu / out; 
  }    
  return out;
}

// Sample truncated gamma random variates
// Ref: Chung, Y.: Simulation of truncated gamma variables 
// Korean Journal of Computational & Applied Mathematics, 1998, 5, 601-610
double truncgamma(){
  double c = MATH_PI_2;
  double X, gX;
  
  bool done = false;
  while(!done)
  {
    X = exprnd(1.0) * 2.0 + c;
    gX = MATH_SQRT_PI_2 / (double)std::sqrt(X);
    
    if(runif(0.0,1.0) <= gX) {
      done = true;
    }
  }
  
  return X;  
}

// Sample truncated inverse Gaussian random variates
// Algorithm 4 in the Windle (2013) PhD thesis, page 129
double tinvgauss(double z, double t){
  double X, u;
  double mu = 1.0/z;
  
  // Pick sampler
  if(mu > t) {
    // Sampler based on truncated gamma 
    // Algorithm 3 in the Windle (2013) PhD thesis, page 128
    while(1) {
	  u = runif(0.0, 1.0);
      X = 1.0 / truncgamma();
      
	  if ((double)std::log(u) < (-z*z*0.5*X)) {
        break;
      }
    }
  }  
  else {
    // Rejection sampler
    X = t + 1.0;
    while(X >= t) {
      X = randinvg(mu);
    }
  }    
  return X;
}

// Sample PG(1,z)
// Based on Algorithm 6 in PhD thesis of Jesse Bennett Windle, 2013
// URL: https://repositories.lib.utexas.edu/bitstream/handle/2152/21842/WINDLE-DISSERTATION-2013.pdf?sequence=1
double samplepg(double z){
  //  PG(b, z) = 0.25 * J*(b, z/2)
  z = (double)std::fabs((double)z) * 0.5;
  
  // Point on the intersection IL = [0, 4/ log 3] and IR = [(log 3)/pi^2, \infty)
  double t = MATH_2_PI;
  
  // Compute p, q and the ratio q / (q + p)
  // (derived from scratch; derivation is not in the original paper)
  double K = z*z/2.0 + MATH_PI2/8.0;
  double logA = (double)std::log(4.0) - MATH_LOG_PI - z;
  double logK = (double)std::log(K);
  double Kt = K * t;
  double w = (double)std::sqrt(MATH_PI_2);

  double logf1 = logA + pnorm(w*(t*z - 1),0.0,1.0,1,1) + logK + Kt;
  double logf2 = logA + 2*z + pnorm(-w*(t*z+1),0.0,1.0,1,1) + logK + Kt;
  double p_over_q = (double)std::exp(logf1) + (double)std::exp(logf2);
  double ratio = 1.0 / (1.0 + p_over_q); 

  double u, X;

  // Main sampling loop; page 130 of the Windle PhD thesis
  while(1) 
  {
    // Step 1: Sample X ? g(x|z)
    u = runif(0.0,1.0);
    if(u < ratio) {
      // truncated exponential
      X = t + exprnd(1.0)/K;
    }
    else {
      // truncated Inverse Gaussian
      X = tinvgauss(z, t);
    }

    // Step 2: Iteratively calculate Sn(X|z), starting at S1(X|z), until U ? Sn(X|z) for an odd n or U > Sn(X|z) for an even n
    int i = 1;
    double Sn = aterm(0, X, t);
    double U = runif(0.0,1.0) * Sn;
    int asgn = -1;
    bool even = false;

    while(1) 
    {
      Sn = Sn + asgn * aterm(i, X, t);
      
      // Accept if n is odd
      if(!even && (U <= Sn)) {
        X = X * 0.25;
        return X;
      }
      
      // Return to step 1 if n is even
      if(even && (U > Sn)) {
        break;
      }
      
      even = !even;
      asgn = -asgn;
      i++;
    }
  }
  return X;
}

double rpg(int n, double z){
  
  double sum = 0;
  int i;
 
  for(i = 0; i < n; i++){
    sum += samplepg(z);
  }
 
  return(sum);
}
///////////////////////////////////////////////////////


void show(int *a, int n);
void show(double *a, int n);
void show(int *a, int r, int c);
void show(double *a, int r, int c);
void zeros(double *a, int n);
void writeRMatrix(string outfile, double * a, int nrow, int ncol);
void writeRMatrix(string outfile, int * a, int nrow, int ncol);
void mvrnorm(double *des, double *mu, double *cholCov, int dim);
void covTransInv(double *z, double *v, int m);
void covTrans(double *v, double *z, int m);
void covTrans(vector<double> v, double *z, int m);
void covTransInvExpand(double *v, double *z, int m);
void covExpand(double *v, double *z, int m);
double logit(double theta, double a, double b);
double logitInv(double z, double a, double b);
double dist2(double &a1, double &a2, double &b1, double &b2);

template <typename T> string toString(T Number){
     ostringstream ss;
     ss << Number;
     return ss.str();
}

//Description: returns the index the value in v that equals i. 
int which(int i, int *v, int nv){
  for(int j = 0; j < nv; j++){
    if(i == v[j]){
      return(j);
    }
  }
  cout << "error in which: i=" << i <<" not found in v" << endl;
  exit(1);
}

void show(cholmod_sparse *A) {
 
    int nrow = A->nrow; 
    int ncol = A->ncol;
    int nzmax = A->nzmax;
    int *p = (int *)A->p; //Column pointers.
    int *i = (int *)A->i; //Row indices.
    double *x = (double *)A->x;
    int col, start, end, idx, row;
    double value;
    double *tmp = new double[nrow*ncol]; zeros(tmp, nrow*ncol);
    
   
    for(col = 0; col < ncol; ++col) {
        start = p[col];
        end = p[col + 1];

        for(idx = start; idx < end; ++idx) {
            row = i[idx];
            value = x[idx];
	    tmp[col*nrow+row] = value;
        }
    }
    show(tmp, nrow, ncol);
    delete[] tmp;
}

void addDiagCCS(cholmod_sparse *A, double x){
  //Assumes there are non-zero value on the diagonal and the matrix is square.

  int i, j;
  if(A->stype == 1){
#pragma omp parallel for
    for(i = 0; i < (int)A->ncol; i++){
      ((double *)A->x)[((int *)A->p)[i+1]-1] += x;
    }
  }else if(A->stype == -1){
#pragma omp parallel for
    for(i = 0; i < (int)A->ncol; i++){
      ((double *)A->x)[((int *)A->p)[i]] += x;
    }
  }else{
    for(i = 0; i < A->ncol; i++){
      j = 0;
      while(true){
	if(i == ((int *)A->i)[((int *)A->p)[i]+j]){
	  ((double *)A->x)[((int *)A->p)[i]+j] += x;
	  break;
	}
	j++;
      }
    } 
  }
  
}

void addDiagCCS(cholmod_sparse *A, double *x){
  //Assumes there are non-zero value on the diagonal and the matrix is square.

  int i, j;
  if(A->stype == 1){
#pragma omp parallel for
    for(i = 0; i < (int)A->ncol; i++){
      ((double *)A->x)[((int *)A->p)[i+1]-1] += x[i];
    }
  }else if(A->stype == -1){
#pragma omp parallel for
    for(i = 0; i < (int)A->ncol; i++){
      ((double *)A->x)[((int *)A->p)[i]] += x[i];
    }
  }else{
    for(i = 0; i < A->ncol; i++){
      j = 0;
      while(true){
	if(i == ((int *)A->i)[((int *)A->p)[i]+j]){
	  ((double *)A->x)[((int *)A->p)[i]+j] += x[i];
	  break;
	}
	j++;
      }
    } 
  }
  
}


double logDetFactor(cholmod_factor *L){
  //Assumes super L.

  double det = 0;
  int i, j;
  
  if(!L->is_super){cout << "factor must be super in logDetL" << endl; exit(1);}
  
  int *lpi = (int*)(L->pi), *lsup = (int*)(L->super);
  for(i = 0; i < L->nsuper; i++) { //Supernodal block i.
    int nrp1 = 1 + lpi[i+1] - lpi[i],
      nc = lsup[i + 1] - lsup[i];
    double *x = (double*)(L->x) + ((int*)(L->px))[i];
    
    for (j = 0; j < nc; j++) {
      det += 2 * log(fabs(x[j * nrp1]));
    }
  }

  return(det);
}

double stCor(double &h, double &u, double &alpha, double &gamma, double &kappa, int &corType){

  double r;
  
  if(corType == 0){
    r = 1.0/std::pow(alpha*std::pow(u, 2.0) + 1.0, kappa)*exp(-gamma*h/std::pow(alpha*std::pow(u, 2.0) + 1.0, kappa/2.0));
  }else if(corType == 1){
    r = exp(-alpha*h)*exp(-gamma*u);
  }else{
    cout << "c++ error: corType not defined." << endl;
  }

  return(r);
}

// double nsCor1(double &h, double &u, double &alpha, double &gamma, double &kappa){
//   double r = 1.0/std::pow(alpha*std::pow(u, 2.0) + 1.0, kappa)*exp(-gamma*h/std::pow(alpha*std::pow(u, 2.0) + 1.0, kappa/2.0));
       
//   return(r);
// }

// double sCor1(double &h, double &u, double &alpha, double &gamma){
//   double r = exp(-alpha*h)*exp(-gamma*u);
  
//   return(r);
// }

double updateFU(cholmod_common *cm, double *c, double *C, double *coords, double *B, 
		cholmod_sparse *F, cholmod_sparse *U, 
		double sigmaSq, double alpha, double gamma, double kappa, int n, int m, int *nnIndx, int *nnIndxLU, int corType){
  
  int i, k, j, l;
  int info = 0;
  int inc = 1;
  double one = 1.0;
  double zero = 0.0;
  char lower = 'L';
  double logDet = 0;
  double t, tf;
  int threadID = 0;
  int mm = m*m;
  double h, u;

#pragma omp parallel for private(k, l, info, threadID, h, u)
  for(i = 0; i < n; i++){
    threadID = omp_get_thread_num();
    if(i > 0){
      for(k = 0; k < nnIndxLU[n+i]; k++){
	h = dist2(coords[i], coords[n+i], coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]]);
	u = dist2(coords[2*n+i], zero, coords[2*n+nnIndx[nnIndxLU[i]+k]], zero);
	c[m*threadID+k] = sigmaSq*stCor(h, u, alpha, gamma, kappa, corType);
	for(l = 0; l <= k; l++){
	  h = dist2(coords[nnIndx[nnIndxLU[i]+k]], coords[n+nnIndx[nnIndxLU[i]+k]], coords[nnIndx[nnIndxLU[i]+l]], coords[n+nnIndx[nnIndxLU[i]+l]]);
	  u = dist2(coords[2*n+nnIndx[nnIndxLU[i]+k]], zero, coords[2*n+nnIndx[nnIndxLU[i]+l]], zero);
	  C[mm*threadID+l*nnIndxLU[n+i]+k] = sigmaSq*stCor(h, u, alpha, gamma, kappa, corType);
	}
      }
      dpotrf_(&lower, &nnIndxLU[n+i], &C[mm*threadID], &nnIndxLU[n+i], &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}
      dpotri_(&lower, &nnIndxLU[n+i], &C[mm*threadID], &nnIndxLU[n+i], &info); if(info != 0){cout << "c++ error: dpotri failed" << endl;}
      dsymv_(&lower, &nnIndxLU[n+i], &one, &C[mm*threadID], &nnIndxLU[n+i], &c[m*threadID], &inc, &zero, &B[nnIndxLU[i]], &inc);
      ((double *)F->x)[i] = 1.0/(sigmaSq - ddot_(&nnIndxLU[n+i], &B[nnIndxLU[i]], &inc, &c[m*threadID], &inc));
    }else{
      B[i] = 0;
      ((double *)F->x)[i] = 1.0/sigmaSq;
    }
  }
  
  //Update U.
  for(i = 0, k = 0; i < n; i++){
    for(j = 0; j <= nnIndxLU[n+i]; j++, k++){
      if(j == nnIndxLU[n+i]){
	((double *)U->x)[k] = 1.0;
      }else{
	((double *)U->x)[k] = -1.0*B[nnIndxLU[i]+nnIndxLU[n+i]-1-j];
      }
    }
  }

  logDet = 0;

  for(i = 0; i < n; i++){
    logDet += log(1/((double *)F->x)[i]);
  }

  return(logDet);
}

void dimCheck(string test, int i, int j){
  if(i != j)
    cout << test << " " << i << "!=" << j << endl;
}

int main(int argc, char **argv){
  int i, j, k, l, s;
  int info = 0;
  int inc = 1;
  double one = 1.0;
  double negOne = -1.0;
  double zero = 0.0;
  double *cholmod_one = new double[2]; cholmod_one[0] = 1; cholmod_one[1] = 0;
  double *cholmod_zero = new double[2]; cholmod_zero[0] = 0; cholmod_zero[1] = 0;
  char lower = 'L';
  char upper = 'U';
  char ntran = 'N';
  char ytran = 'T';
  char rside = 'R';
  char lside = 'L';
  
  string parfile;
  if(argc > 1)
    parfile = argv[1];
  else
    parfile = "pfile";
  
  kvpar par(parfile);
  
  bool debug = false;
  
  //Get stuff from the pfile.
  int nThreads; par.getVal("n.threads", nThreads);
  int seed; par.getVal("seed", seed);
  int nSamples; par.getVal("n.samples", nSamples);
  int nSamplesStart; par.getVal("n.samples.start", nSamplesStart);
  int nReport; par.getVal("n.report", nReport);
  string outFile; par.getVal("out.file", outFile);
  int permute; par.getVal("perm", permute);
  int cholStats; par.getVal("cholmod.stats", cholStats);
  int printLP; par.getVal("print.P&L", printLP);

  string corTypeString; par.getVal("st.cor.type", corTypeString);
  int corType;
  bool nonSep;
  if(corTypeString == "non_sep_1"){
    corType = 0;
    nonSep = true;
  }else if(corTypeString == "sep_1"){
    corType = 1;
    nonSep = false;
  }else{
    cout << "c++ error: st.cor.type not defined." << endl;
  }
  
  omp_set_num_threads(nThreads);
  
  //Set seed.
  set_seed(123,seed);
  
  //m number of nearest neighbors.
  //n number of locations.
  //p number of columns of X.
  int m; par.getVal("m", m);
  int n; par.getVal("n", n);
  int p; par.getVal("p", p);
  int pp = p*p;
  int np = n*p;

  //Data and starting values.
  double *X = NULL; par.getFile("X.file", X, i, j);
  double *y = NULL; par.getFile("y.file", y, i, j);
  double *coords = NULL; par.getFile("coords.file", coords, i, j);
  vector<double> betaStarting; par.getVal("beta.starting", betaStarting);
  double *beta = new double[p];
  for(i = 0; i < p; i++){
    beta[i] = betaStarting[i];
  }


  vector<double> betaMu; par.getVal("beta.mu", betaMu);
  double betaSigmaSq; par.getVal("beta.sigmaSq", betaSigmaSq);	

  int nTrial = 1;
  
  double sigmaSq; par.getVal("sigmaSq.starting", sigmaSq);
  double alpha; par.getVal("alpha.starting", alpha);
  double gamma; par.getVal("gamma.starting", gamma);

  //Priors and tuning.
  double sigmaSq_a; par.getVal("sigmaSq.a", sigmaSq_a);
  double sigmaSq_b; par.getVal("sigmaSq.b", sigmaSq_b);

  double alpha_a; par.getVal("alpha.a", alpha_a);
  double alpha_b; par.getVal("alpha.b", alpha_b);
  double alphaTuning; par.getVal("alpha.tuning", alphaTuning);
  
  double gamma_a; par.getVal("gamma.a", gamma_a);
  double gamma_b; par.getVal("gamma.b", gamma_b);
  double gammaTuning; par.getVal("gamma.tuning", gammaTuning);

  //Stuff for non-separable model (these won't be accessed for a separable model).
  double kappa = 0;
  double kappa_a = 0;
  double kappa_b = 0;
  double kappaTuning = 0;
  
  if(nonSep){
    kappa; par.getVal("kappa.starting", kappa);
    kappa_a; par.getVal("kappa.a", kappa_a);
    kappa_b; par.getVal("kappa.b", kappa_b);
    kappaTuning; par.getVal("kappa.tuning", kappaTuning);
  }
  
  int *nnIndx = NULL; par.getFile("nn.indx.file", nnIndx, i, j);
  int *nnIndxLU = NULL; par.getFile("nn.indx.lu.file", nnIndxLU, i, j);
  int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);

  //Other stuff.
  int mm = m*m;
  double *B = new double[nIndx];
  double *c = new double[m*nThreads];
  double *C = new double[mm*nThreads];
  
  //Return stuff.
  int nSamplesKeep = (nSamples-nSamplesStart);
  double *betaSamples = new double[p*nSamplesKeep];
  double *sigmaSqSamples = new double[nSamplesKeep];
  double *alphaSamples = new double[nSamplesKeep];
  double *gammaSamples = new double[nSamplesKeep];
  double *kappaSamples = new double[nSamplesKeep];
  double *wSamples = new double[n*nSamplesKeep];
  double *fittedSamples = new double[n*nSamplesKeep];
 
  double logPostCand, logPostCurrent, logDet, status = 0;
  double accept = 0;
  double batchAccept = 0;
  double alphaCand, gammaCand, kappaCand;

  //PG stuff.
  double *omega = new double[n];
  double *kappa0 = new double[n];
  double *yStr = new double[n];
  
  for(i = 0; i < n; i++){
    kappa0[i] = y[i] - static_cast<double>(nTrial)/2.0;
  }
  
  double *w = new double[n]; zeros(w, n);

  double *tmp_p = new double[p];
  double *tmp_p2 = new double[p];
  double *tmp_pp = new double[pp];
  double *tmp_n = new double[n];
  double *tmp_np = new double[n*p];
  double a, v, b, e, mu, var, aij;
  int jj, kk, ll;

  int sKeep = 0;

  //cholmod stuff.
  int ver[3];
  FILE *fp;
  cholmod_common Common, *cm;
  cm = &Common;
  cholmod_start(cm);
  cm->supernodal = CHOLMOD_SUPERNODAL;
  cm->final_ll = true; 
  
  if(permute){
    cout << "requesting permutation trials" << endl;
    cm->nmethods = 9; 
  }else{
    cm->nmethods = 1;
    cm->method[0].ordering = CHOLMOD_NATURAL;
  }
  
  if(cholStats){
    cholmod_version (ver) ;
    printf ("cholmod version %d.%d.%d\n", ver [0], ver [1], ver [2]) ;
    SuiteSparse_version (ver) ;
    printf ("SuiteSparse version %d.%d.%d\n", ver [0], ver [1], ver [2]) ;
  }

  //Note, U = t(I - A), F = 1/D.
  
  //Allocate and initialize U, U', and F.
  cholmod_sparse *U = cholmod_allocate_sparse(n, n, nIndx+n, 1, 1, 0, CHOLMOD_REAL, cm);
  cholmod_sparse *Ut = cholmod_allocate_sparse(n, n, nIndx+n, 1, 1, 0, CHOLMOD_REAL, cm);
  cholmod_factor *L, *LCand;
    
  ((int *)U->p)[n] = nIndx+n;
  for(i = 0, k = 0; i < n; i++){
    for(j = 0; j <= nnIndxLU[n+i]; j++, k++){
      if(j == nnIndxLU[n+i]){
  	((double *)U->x)[k] = 1.0;
  	((int *)U->i)[k] = i;
      }else{
  	((double *)U->x)[k] = 1.0;
  	((int *)U->i)[k] = nnIndx[nnIndxLU[i]+nnIndxLU[n+i]-1-j];
      }
    }
    if(i == 0){
      ((int *)U->p)[i] = 0;
    }else{
      ((int *)U->p)[i] = ((int *)U->p)[i-1]+nnIndxLU[n+i-1]+1;
    }
  }

  cholmod_sparse *F = cholmod_allocate_sparse(n, n, n, 1, 1, 0, CHOLMOD_REAL, cm);
  ((int *)F->p)[n] = n;
  for(i = 0; i < n; i++){
    ((int *)F->i)[i] = i;
    ((int *)F->p)[i] = i;
    ((double *)F->x)[i] = 1.0;
  }
  
  logDet = updateFU(cm, c, C, coords, B, F, U, sigmaSq, alpha, gamma, kappa, n, m, nnIndx, nnIndxLU, corType);
  
  //cout << logDet << endl;

  //UF^{-1}U'.
  cholmod_transpose_unsym(U, 1, NULL, NULL, 0, Ut, cm);
  cholmod_sparse *UF = cholmod_ssmult(U, F, 0, 1, 1, cm);
  cholmod_sparse *UFUt = cholmod_ssmult(UF, Ut, 1, 1, 1, cm);

  //Use Lcp as the persistent copy of the symbolic factorization.
  cholmod_factor *Lcp = cholmod_analyze(UFUt, cm);

  cholmod_print_factor(Lcp, "L", cm);
  //printf ("pnalyze: flop %g lnz %g\n", cm->fl, cm->lnz);
  cout << "permutation selected: " <<  cm->selected << " " << Lcp->ordering << " " << cm->method[0].ordering << endl;

  //Set the permutation matrix and permute y, X, and set up for v = y - X'beta.
  cholmod_sparse *P = cholmod_allocate_sparse(n, n, n, 1, 1, 0, CHOLMOD_REAL, cm);
  int *perm = (int*)Lcp->Perm;
  ((int*)P->p)[n] = n;
  
  for(i = 0; i < n; i++){
    ((double*)P->x)[i] = 1.0;
    ((int*)P->i)[i] = which(i, perm, n);
    ((int*)P->p)[i] = i;
  }

  //Set up some cholmod work vectors. 
  cholmod_dense *tmp_n1_cm = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, cm);
  cholmod_dense *tmp_n2_cm = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, cm);
  cholmod_dense *tmp_n3_cm; // Will be dynamically allocated and freed within the sampler.
  cholmod_dense *tmp_n4_cm; // Will be dynamically allocated and freed within the sampler.
  
  //One more test update of B, F, U, Ut, and L.  
  logDet = updateFU(cm, c, C, coords, B, F, U, sigmaSq, alpha, gamma, kappa, n, m, nnIndx, nnIndxLU, corType);
  //cout << logDet << endl;

  //Get UF^{-1}U, its factor, log det of (UF^{-1}U)^{-1}, and double check some stuff.
  cholmod_transpose_unsym(U, 1, NULL, NULL, 0, Ut, cm);
  UF = cholmod_ssmult(U, F, 0, 1, 1, cm);
  UFUt = cholmod_ssmult(UF, Ut, 1, 1, 1, cm);

  L = cholmod_copy_factor(Lcp, cm);
    
  cholmod_factorize(UFUt, L, cm);

  //cout << logDetFactor(L) << endl; //Should equal -logDet.
  
  if(printLP){
    cout << "writing L and P" << endl;
    
    string f = "P-m"+toString(m)+"-perm-selected-"+toString(cm->selected);
    fp = fopen(f.c_str(), "w+");
    cholmod_write_sparse(fp, P, 0, 0, cm);
    fclose(fp);
    
    cholmod_sparse *L_cm = cholmod_factor_to_sparse(L, cm);
    
    f = "L-m"+toString(m)+"-perm-selected-"+toString(cm->selected);
    fp = fopen(f.c_str(), "w+");
    cholmod_write_sparse(fp, L_cm, 0, 0, cm);
    fclose(fp);
    cholmod_free_sparse(&L_cm, cm);
    cout << "Writing L then exit because there is something messed up with xtype in cholmod" << endl;
    exit(1);
  }

  //Free cholmod stuff (it would be nice if these objects could be pre-allocated and avoid dynamic allocation in the sampler, my hunch is there's a way but this works for now).
  //U, Ut, and F are modified in place, so they should be okay.
  cholmod_free_sparse(&UF, cm);
  cholmod_free_sparse(&UFUt, cm);
  cholmod_free_factor(&L, cm);
  
  double wall_start_0 = get_wall_time();

  cout << "start sampling" << endl;
  for(s = 0; s < nSamples; s++){

    ///////////////
    //update augs
    ///////////////     
    for(i = 0; i < n; i++){ 
      omega[i] = rpg(nTrial, ddot_(&p, &X[i], &n, beta, &inc) + w[i]);
      yStr[i] = kappa0[i]/omega[i];
    }
    
    ///////////////
    //update beta 
    ///////////////
    for(i = 0; i < n; i++){
      tmp_n[i] = (yStr[i] - w[i])*omega[i];
    }
    dgemv_(&ytran, &n, &p, &one, X, &n, tmp_n, &inc, &zero, tmp_p, &inc); 	  
    
    for(i = 0; i < n; i++){
      for(j = 0; j < p; j++){
    	tmp_np[j*n+i] = X[j*n+i]*omega[i];
      }
    }
    
    dgemm_(&ytran, &ntran, &p, &p, &n, &one, X, &n, tmp_np, &n, &zero, tmp_pp, &p);

    //Add prior, assume N(mu_beta, sigma^2_beta * I).
    for(i = 0; i < p; i++){
      tmp_pp[i*p+i] += 1.0/betaSigmaSq;
      tmp_p[i] += (1.0/betaSigmaSq)*betaMu[i];
    }

    dpotrf_(&lower, &p, tmp_pp, &p, &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}
    dpotri_(&lower, &p, tmp_pp, &p, &info); if(info != 0){cout << "c++ error: dpotri failed" << endl;}
    dsymv_(&lower, &p, &one, tmp_pp, &p, tmp_p, &inc, &zero, tmp_p2, &inc);
    dpotrf_(&lower, &p, tmp_pp, &p, &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}
    mvrnorm(beta, tmp_p2, tmp_pp, p);
    
    ///////////////
    //update w 
    ///////////////
    //Get L for UFUt + diag(1/tau^2), i.e., C^{-1} + diag(1/tau^2).
    logDet = updateFU(cm, c, C, coords, B, F, U, sigmaSq, alpha, gamma, kappa, n, m, nnIndx, nnIndxLU, corType);
    
    cholmod_transpose_unsym(U, 1, NULL, NULL, 0, Ut, cm);
    UF = cholmod_ssmult(U, F, 0, 1, 1, cm);
    UFUt = cholmod_ssmult(UF, Ut, 1, 1, 1, cm);

    addDiagCCS(UFUt, omega);

    L = cholmod_copy_factor(Lcp, cm);    
    cholmod_factorize(UFUt, L, cm);

    dgemv_(&ntran, &n, &p, &one, X, &n, beta, &inc, &zero, tmp_n, &inc);
    for(i = 0; i < n; i++){
      ((double *)tmp_n1_cm->x)[i] = (yStr[i] - tmp_n[i])*omega[i]; //v
    }

    cholmod_sdmult(P, 0, cholmod_one, cholmod_zero, tmp_n1_cm, tmp_n2_cm, cm); //Pv
    tmp_n3_cm = cholmod_solve(CHOLMOD_L, L, tmp_n2_cm, cm);  //a = solve(L, Pv).
    
    for(i = 0; i < n; i++){
      ((double *)tmp_n3_cm->x)[i] += rnorm(0, 1);
    }

    tmp_n4_cm = cholmod_solve(CHOLMOD_Lt, L, tmp_n3_cm, cm); //b = solve(L', a + v), where v is n(0, 1).
    cholmod_sdmult(P, 1, cholmod_one, cholmod_zero, tmp_n4_cm, tmp_n1_cm, cm); //P'b

    dcopy_(&n, (double *)tmp_n1_cm->x, &inc, w, &inc);

    //cholmod_free_sparse(&UF, cm); //This is used once more for updating sigma^2, then freed.
    cholmod_free_sparse(&UFUt, cm);
    cholmod_free_factor(&L, cm);
    //tmp_n1_cm and tmp_n2_cm are pre-allocated and modified in place and should be okay.
    cholmod_free_dense(&tmp_n3_cm, cm);
    cholmod_free_dense(&tmp_n4_cm, cm);

    /////////////////////
    //update sigma^2
    /////////////////////
    UFUt = cholmod_ssmult(UF, Ut, 1, 1, 1, cm); //UF is from w above.

    cholmod_sdmult(UFUt, 0, cholmod_one, cholmod_zero, tmp_n1_cm, tmp_n2_cm, cm); //tmp_n1_cm is w from above.
    a = ddot_(&n, (double *)tmp_n2_cm->x, &inc, w, &inc);

    sigmaSq = 1.0/rgamma(sigmaSq_a+n/2.0, 1.0/(sigmaSq_b+0.5*a*sigmaSq));

    cholmod_free_sparse(&UF, cm); //Done with UF now.
    cholmod_free_sparse(&UFUt, cm);
 
    ///////////////
    //update theta
    ///////////////
    //Current (some savings could be had by reusing F and U if the proposal is rejected).
    logDet = updateFU(cm, c, C, coords, B, F, U, sigmaSq, alpha, gamma, kappa, n, m, nnIndx, nnIndxLU, corType);
    
    cholmod_transpose_unsym(U, 1, NULL, NULL, 0, Ut, cm);
    UF = cholmod_ssmult(U, F, 0, 1, 1, cm);
    UFUt = cholmod_ssmult(UF, Ut, 1, 1, 1, cm);

    cholmod_sdmult(UFUt, 0, cholmod_one, cholmod_zero, tmp_n1_cm, tmp_n2_cm, cm); //note, tmp_n1_cm is w from above.
    a = ddot_(&n, (double *)tmp_n2_cm->x, &inc, w, &inc);
    
    logPostCurrent = -0.5*logDet - 0.5*a;
    logPostCurrent += log(alpha - alpha_a) + log(alpha_b - alpha);
    logPostCurrent += log(gamma - gamma_a) + log(gamma_b - gamma);
    if(nonSep){
      logPostCurrent += log(kappa - kappa_a) + log(kappa_b - kappa);
    }
    
    cholmod_free_sparse(&UF, cm);
    cholmod_free_sparse(&UFUt, cm);
    
    //Candidate.
    alphaCand = logitInv(rnorm(logit(alpha, alpha_a, alpha_b), alphaTuning), alpha_a, alpha_b);
    gammaCand = logitInv(rnorm(logit(gamma, gamma_a, gamma_b), gammaTuning), gamma_a, gamma_b);
    if(nonSep){
      kappaCand = logitInv(rnorm(logit(kappa, kappa_a, kappa_b), kappaTuning), kappa_a, kappa_b);
    }
    
    logDet = updateFU(cm, c, C, coords, B, F, U, sigmaSq, alphaCand, gammaCand, kappaCand, n, m, nnIndx, nnIndxLU, corType);
    
    cholmod_transpose_unsym(U, 1, NULL, NULL, 0, Ut, cm);
    UF = cholmod_ssmult(U, F, 0, 1, 1, cm);
    UFUt = cholmod_ssmult(UF, Ut, 1, 1, 1, cm);

    cholmod_sdmult(UFUt, 0, cholmod_one, cholmod_zero, tmp_n1_cm, tmp_n2_cm, cm); //note, tmp_n1_cm is w from above.
    a = ddot_(&n, (double *)tmp_n2_cm->x, &inc, w, &inc);
    
    logPostCand = -0.5*logDet - 0.5*a;
    logPostCand += log(alphaCand - alpha_a) + log(alpha_b - alphaCand);
    logPostCand += log(gammaCand - gamma_a) + log(gamma_b - gammaCand);
    if(nonSep){
      logPostCand += log(kappaCand - kappa_a) + log(kappa_b - kappaCand);
    }
    
    cholmod_free_sparse(&UF, cm);
    cholmod_free_sparse(&UFUt, cm);
    
    if(runif(0.0,1.0) <= exp(logPostCand - logPostCurrent)){

      alpha = alphaCand;
      gamma = gammaCand;
      kappa = kappaCand;

      accept++;
      batchAccept++;
    }
    
   ///////////////
   //fit 
   ///////////////
   for(i = 0; i < n; i++){
     tmp_n[i] = 1.0/(1.0 + exp(-(ddot_(&p, &X[i], &n, beta, &inc) + w[i])));
   }
    
    if(s >= nSamplesStart){
      dcopy_(&p, beta, &inc, &betaSamples[sKeep*p], &inc);
      sigmaSqSamples[sKeep] = sigmaSq;
      alphaSamples[sKeep] = alpha;
      gammaSamples[sKeep] = gamma;
      kappaSamples[sKeep] = kappa;
      dcopy_(&n, w, &inc, &wSamples[sKeep*n], &inc);
      dcopy_(&n, tmp_n, &inc, &fittedSamples[sKeep*n], &inc);
      sKeep++;
    }
    
    /////////////////////////////////////////////
    //report
    /////////////////////////////////////////////
    if(status == nReport){
      cout << "percent complete: " << 100*s/nSamples << endl;     
      cout << "theta: " << 100.0*batchAccept/nReport << endl;
      cout << "beta: " << endl;
      show(beta, 1, p);
      cout << "sigmaSq: " << sigmaSq << endl;
      cout << "alpha: " << alpha << endl;
      cout << "gamma: " << gamma << endl;
      cout << "kappa: " << kappa << endl;
      cout << "---------------" << endl;
      batchAccept = 0;
      status = 0;
    }
    status++;
   
  }
  
  cholmod_free_factor(&Lcp, cm);

  cout << "Runtime: " << get_wall_time() - wall_start_0 << endl;
  
  writeRMatrix(outFile+"-beta", betaSamples, p, nSamplesKeep);
  writeRMatrix(outFile+"-alpha", alphaSamples, 1, nSamplesKeep);
  writeRMatrix(outFile+"-gamma", gammaSamples, 1, nSamplesKeep);
  if(nonSep){
    writeRMatrix(outFile+"-kappa", kappaSamples, 1, nSamplesKeep);
  }
  writeRMatrix(outFile+"-sigmaSq", sigmaSqSamples, 1, nSamplesKeep);
  writeRMatrix(outFile+"-w", wSamples, n, nSamplesKeep);
  writeRMatrix(outFile+"-fitted", fittedSamples, n, nSamplesKeep); 

  return(0);
}

void writeRMatrix(string outfile, double * a, int nrow, int ncol){

    ofstream file(outfile.c_str());
    if ( !file ) {
      cerr << "Data file could not be opened." << endl;
      exit(1);
    }
    
    for(int i = 0; i < nrow; i++){
      for(int j = 0; j < ncol-1; j++){
	file << setprecision(10) << fixed << a[j*nrow+i] << "\t";
      }
      file << setprecision(10) << fixed << a[(ncol-1)*nrow+i] << endl;    
    }
    file.close();
}


void writeRMatrix(string outfile, int* a, int nrow, int ncol){
  
  ofstream file(outfile.c_str());
  if ( !file ) {
    cerr << "Data file could not be opened." << endl;
    exit(1);
  }
  
  
  for(int i = 0; i < nrow; i++){
    for(int j = 0; j < ncol-1; j++){
      file << fixed << a[j*nrow+i] << "\t";
    }
    file << fixed << a[(ncol-1)*nrow+i] << endl;    
  }
  file.close();
}

void mvrnorm(double *des, double *mu, double *cholCov, int dim){
  
  int i;
  int inc = 1;
  double one = 1.0;
  double zero = 0.0;
  
  for(i = 0; i < dim; i++){
    des[i] = rnorm(0, 1);
  }
 
  dtrmv_("L", "N", "N", &dim, cholCov, &dim, des, &inc);
  daxpy_(&dim, &one, mu, &inc, des, &inc);
}

void show(double *a, int n){
  for(int i = 0; i < n; i++)
    cout << setprecision(20) << fixed << a[i] << endl;
}


void show(int *a, int n){
  for(int i = 0; i < n; i++)
    cout << fixed << a[i] << endl;
}


void zeros(double *a, int n){
  for(int i = 0; i < n; i++)
    a[i] = 0.0;
}


void show(double *a, int r, int c){

  for(int i = 0; i < r; i++){
    for(int j = 0; j < c; j++){
      cout << fixed << a[j*r+i] << "\t";
    }
    cout << endl;
  }
}


void show(int *a, int r, int c){

  for(int i = 0; i < r; i++){
    for(int j = 0; j < c; j++){

      cout << fixed << a[j*r+i] << "\t";
    }
    cout << endl;
  }
}

void covTransInv(double *z, double *v, int m){
  int i, j, k;

  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      v[k] = z[k];
      if(i == j)
	v[k] = exp(z[k]);
    }
  }

}

void covTrans(double *v, double *z, int m){
  int i, j, k;

  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      z[k] = v[k];
      if(i == j)
	z[k] = log(v[k]);
    }
  }

}

void covTrans(vector<double> v, double *z, int m){
  int i, j, k;

  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      z[k] = v[k];
      if(i == j)
	z[k] = log(v[k]);
    }
  }

}

void covTransInvExpand(double *v, double *z, int m){
  int i, j, k;
  
  zeros(z, m*m);
  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      z[i*m+j] = v[k];
      if(i == j)
	z[i*m+j] = exp(z[i*m+j]);
    }
  }
  
}

void covExpand(double *v, double *z, int m){
  int i, j, k;
  
  zeros(z, m*m);
  for(i = 0, k = 0; i < m; i++){
    for(j = i; j < m; j++, k++){
      z[i*m+j] = v[k];
    }
  }
  
}

double logit(double theta, double a, double b){
  return log((theta-a)/(b-theta));
}

double logitInv(double z, double a, double b){
  return b-(b-a)/(1+exp(z));
}

double dist2(double &a1, double &a2, double &b1, double &b2){
  return(sqrt(pow(a1-b1,2)+pow(a2-b2,2)));
}
