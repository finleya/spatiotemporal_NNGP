#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <limits>
#include <omp.h>
using namespace std;

#include "../../libs/kvpar.h"

#define MATHLIB_STANDALONE
#include <Rmath.h>
#include <Rinternals.h>

#include <time.h>
#include <sys/time.h>
#define CPUTIME (SuiteSparse_time ( ))

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
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

void show(int *a, int n);
void show(double *a, int n);
void show(int *a, int r, int c);
void show(double *a, int r, int c);
void zeros(double *a, int n);
void writeRMatrix(string outfile, double * a, int nrow, int ncol);
void writeRMatrix(string outfile, int * a, int nrow, int ncol);
double dist2(double &a1, double &a2, double &b1, double &b2);

void dimCheck(string test, int i, int j){
  if(i != j)
    cout << test << " " << i << "!=" << j << endl;
}


double nsCor(double h, double u, double alpha, double gamma, double kappa){
  double r = 1.0/std::pow(alpha*std::pow(u, 2.0) + 1.0, kappa)*exp(-gamma*h/std::pow(alpha*std::pow(u, 2.0) + 1.0, kappa/2.0));
       
  return(r);
}

int main(int argc, char **argv){
  int i, j, k, l, s;
  int info = 0;
  int inc = 1;
  double one = 1.0;
  double negOne = -1.0;
  double zero = 0.0;
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
  
  int nThreads; par.getVal("n.threads", nThreads);
  int seed; par.getVal("seed", seed);
  int nSamples; par.getVal("n.samples", nSamples);
  int nReport; par.getVal("n.report", nReport);
  string outFile; par.getVal("out.file", outFile);
    
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
  int mm = m*m;
  
  int n0; par.getVal("n0", n0);

  //Data and starting values.
  double *X0 = NULL; par.getFile("X0.file", X0, i, j);
  double *coords0 = NULL; par.getFile("coords0.file", coords0, i, j);
  double *coords = NULL; par.getFile("coords.file", coords, i, j);

  int *nnIndx0 = NULL; par.getFile("nnIndx.0.file", nnIndx0, i, j);
  
  double *beta = NULL; par.getFile("beta.samples.file", beta, i, j);
  double *sigmaSq = NULL; par.getFile("sigmaSq.samples.file", sigmaSq, i, j);
  double *alpha = NULL; par.getFile("alpha.samples.file", alpha, i, j);
  double *gamma = NULL; par.getFile("gamma.samples.file", gamma, i, j);
  double *kappa = NULL; par.getFile("kappa.samples.file", kappa, i, j);
  double *w = NULL; par.getFile("w.samples.file", w, i, j);

  double *C = new double[nThreads*mm];
  double *c = new double[nThreads*m];
  double *tmp_m = new double[nThreads*m];
  
  double *p0 = new double[n0*nSamples]; zeros(p0, n0*nSamples);
  double *w0 = new double[n0*nSamples]; zeros(w0, n0*nSamples);
  
  int status = 0;
  double wall_start_0 = get_wall_time();

  int zWIndx = -1;
  double *wZ = new double[n0*nSamples];
  for(i = 0; i < n0*nSamples; i++){
    wZ[i] = rnorm(0.0, 1.0);
  }

  double h, u, d;

  int threadID;
  
  cout << "start sampling" << endl;

  for(i = 0; i < n0; i++){

#pragma omp parallel for private(threadID, j, k, l, u, h, info)
    for(s = 0; s < nSamples; s++){

      threadID = omp_get_thread_num();
      
      for(k = 0; k < m; k++){
	h = dist2(coords[nnIndx0[i+n0*k]], coords[n+nnIndx0[i+n0*k]], coords0[i], coords0[n0+i]);
	u = dist2(coords[2*n+nnIndx0[i+n0*k]], zero, coords0[2*n0+i], zero);
	c[threadID*m+k] = sigmaSq[s] * nsCor(h, u, alpha[s], gamma[s], kappa[s]);
	for(l = 0; l < m; l++){
	  h = dist2(coords[nnIndx0[i+n0*k]], coords[n+nnIndx0[i+n0*k]], coords[nnIndx0[i+n0*l]], coords[n+nnIndx0[i+n0*l]]);
	  u = dist2(coords[2*n+nnIndx0[i+n0*k]], zero, coords[2*n+nnIndx0[i+n0*l]], zero);
	  C[threadID*mm+l*m+k] = sigmaSq[s] * nsCor(h, u, alpha[s], gamma[s], kappa[s]);
	}
	
      }
      
      dpotrf_(&lower, &m, &C[threadID*mm], &m, &info); if(info != 0){cout << "c++ error: dpotrf failed" << endl;}
      dpotri_(&lower, &m, &C[threadID*mm], &m, &info); if(info != 0){cout << "c++ error: dpotri failed" << endl;}
      
      dsymv_(&lower, &m, &one, &C[threadID*mm], &m, &c[threadID*m], &inc, &zero, &tmp_m[threadID*m], &inc);
      
      d = 0;
      for(k = 0; k < m; k++){
      	d += tmp_m[threadID*m+k]*w[s*n+nnIndx0[i+n0*k]]; 
      }
      
#pragma omp atomic
      zWIndx++;
	
      //cout << sigmaSq[s] << "\t" << phi[s] << "\t" << d << endl;
      w0[s*n0+i] = sqrt(sigmaSq[s] - ddot_(&m, &tmp_m[threadID*m], &inc, &c[threadID*m], &inc))*wZ[zWIndx]  + d;

      p0[s*n0+i] = 1.0/(1.0 + exp(-(ddot_(&p, &X0[i], &n0, &beta[s*p], &inc) + w0[s*n0+i])));

    }//end s

    if(status == nReport){

      cout << 100.0*i/n0 << endl;
      cout << "---------------" << endl;
      status = 0;
      
    }
    status++;
    
  }//end i
  
      
  cout << "Runtime: " << get_wall_time() - wall_start_0 << endl;

  writeRMatrix(outFile+"-w0", w0, n0, nSamples);
  writeRMatrix(outFile+"-p0", p0, n0, nSamples);


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


double dist2(double &a1, double &a2, double &b1, double &b2){
  return(sqrt(pow(a1-b1,2)+pow(a2-b2,2)));
}
