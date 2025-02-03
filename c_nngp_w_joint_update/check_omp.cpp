#include <iostream>

int main() {
#ifdef _OPENMP
    std::cout << "OpenMP version: " << _OPENMP << std::endl;
#else
    std::cout << "OpenMP is not enabled!" << std::endl;
#endif
    return 0;
}
