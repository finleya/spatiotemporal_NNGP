#ifndef R_NO_REMAP
#  define R_NO_REMAP
#endif

#include <iomanip>
#include <string>
#include <limits>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "util.h"
#include "nn.h"
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include <R_ext/BLAS.h>
#include <R_ext/Utils.h>

///////////////////////////////////////////////////////////////////
//u index 
///////////////////////////////////////////////////////////////////
SEXP mkUIndx(SEXP n_r, SEXP m_r, SEXP nnIndx_r, SEXP uIndx_r, SEXP uIndxLU_r, SEXP uiIndx_r, SEXP nnIndxLU_r, SEXP searchType_r){

  int n = INTEGER(n_r)[0];
  int m = INTEGER(m_r)[0];
  int *nnIndx = INTEGER(nnIndx_r);
  int *uIndx = INTEGER(uIndx_r);
  int *uIndxLU = INTEGER(uIndxLU_r);
  int *uiIndx = INTEGER(uiIndx_r);
  int *nnIndxLU = INTEGER(nnIndxLU_r);
  int searchType = INTEGER(searchType_r)[0];
  int i, j, k;
  int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);

  if(searchType == 0){
    mkUIndx0(n, m, nnIndx, uIndx, uIndxLU);
  }else if(searchType == 1){
    mkUIndx1(n, m, nnIndx, uIndx, uIndxLU);
  }else{
    mkUIndx2(n, m, nnIndx, nnIndxLU, uIndx, uIndxLU);
  }
  
  //u lists those locations that have the i-th location as a neighbor
  //then for each of those locations that have i as a neighbor, we need to know the index of i in each of their B vectors (i.e. where does i fall in their neighbor set)
  for(i = 0; i < n; i++){//for each i
    for(j = 0; j < uIndxLU[n+i]; j++){//for each location that has i as a neighbor
  	k = uIndx[uIndxLU[i]+j];//index of a location that has i as a neighbor
  	uiIndx[uIndxLU[i]+j] = which(i, &nnIndx[nnIndxLU[k]], nnIndxLU[n+k]);
    }
  }
  
  return R_NilValue;
}


///////////////////////////////////////////////////////////////////
//Brute force 
///////////////////////////////////////////////////////////////////

SEXP mkNNIndx(SEXP n_r, SEXP m_r, SEXP r_r, SEXP coords_r, SEXP nnIndx_r, SEXP nnDist_r, SEXP nnIndxLU_r, SEXP nThreads_r){
  
  int i, j, iNNIndx, iNN;
  double d;
  
  int n = INTEGER(n_r)[0];
  int m = INTEGER(m_r)[0];
  int r = INTEGER(r_r)[0];
  double *coords = REAL(coords_r);
  int *nnIndx = INTEGER(nnIndx_r);
  double *nnDist = REAL(nnDist_r);
  int *nnIndxLU = INTEGER(nnIndxLU_r);
  int nThreads = INTEGER(nThreads_r)[0];
    
#ifdef _OPENMP
  omp_set_num_threads(nThreads);
#else
  if(nThreads > 1){
    Rf_warning("n.omp.threads > %i, but source not compiled with OpenMP support.", nThreads);
    nThreads = 1;
  }
#endif

  int nIndx = static_cast<int>(static_cast<double>(1+m)/2*m+(n-m-1)*m);
  
  for(i = 0; i < nIndx; i++){
    nnDist[i] = std::numeric_limits<double>::infinity();
  }
  
#ifdef _OPENMP
#pragma omp parallel for private(j, iNNIndx, iNN, d)
#endif
  for(i = 0; i < n; i++){
    getNNIndx(i, m, iNNIndx, iNN);
    nnIndxLU[i] = iNNIndx;
    nnIndxLU[n+i] = iNN;   
    if(i != 0){  
      for(j = 0; j < i; j++){	
	if(r == 3){
	  d = dist3(coords[i], coords[n+i], coords[2*n+i], coords[j], coords[n+j], coords[2*n+j]);
	}else if(r == 2){
	  dist2(coords[i], coords[n+i], coords[j], coords[n+j]);
	}else{
	  Rf_error("r is not specified correctly in nn.cpp\n");
	}
	if(d < nnDist[iNNIndx+iNN-1]){	  
	  nnDist[iNNIndx+iNN-1] = d;
	  nnIndx[iNNIndx+iNN-1] = j;
	  rsort_with_index(&nnDist[iNNIndx], &nnIndx[iNNIndx], iNN);
	}	
      }
    }
  }
  
  return R_NilValue;
}

