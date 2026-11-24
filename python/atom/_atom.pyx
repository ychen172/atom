# cython: language_level=3

cdef class CanteraGas:
    def __cinit__(self, str mechanismFile, str phase_id=""):
        self.ptrCxxCanteraGas = make_unique[CxxCanteraGas](mechanismFile.encode(),phase_id.encode())

    def setTPY(self, double T, double P, Y):
        self.ptrCxxCanteraGas.get().setTPY(T, P, Y)

    def equiHP(self,):
        self.ptrCxxCanteraGas.get().equiHP()
    
    def getTP(self):
        return self.ptrCxxCanteraGas.get().getTP()