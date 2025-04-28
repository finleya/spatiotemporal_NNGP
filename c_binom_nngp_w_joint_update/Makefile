CXX = g++
OPTFLAGS = -O3 -fopenmp
CXXFLAGS = -I/opt/OpenBLAS/include 
LIBS = -I/opt/OpenBLAS/include -I/usr/local/lib/R/include -L /opt/OpenBLAS/lib -lopenblas -lpthread -lgfortran -L/usr/local/lib -lRmath  -lcholmod -lsuitesparseconfig -lcolamd -lcamd -lcholmod -lamd -lccolamd -lmetis

binom_nngp: binom_nngp.o kvpar.o
	$(CXX) $(OPTFLAGS) -o binom_nngp binom_nngp.o kvpar.o $(LIBS) 

binom_nngp.o: binom_nngp.cpp
	$(CXX) $(OPTFLAGS) binom_nngp.cpp -c $(LIBS)

kvpar.o: ../libs/kvpar.cpp
	$(CXX) $(OPTFLAGS) ../libs/kvpar.cpp -c

clean:
	rm -rf *o binom_nngp
