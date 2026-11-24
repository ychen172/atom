from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.memory cimport unique_ptr, make_unique

cdef extern from "thermTrans.h":
    cdef cppclass CxxCanteraGas "CanteraGas":
        CxxCanteraGas(const string& mechanismFile, const string& phaseID) except +
        void setTPY(double T, double P, const vector[double]& Y) except +
        void equiHP() except +
        vector[double] getTP() const

cdef class CanteraGas:
    cdef unique_ptr[CxxCanteraGas] ptrCxxCanteraGas