#include "thermTrans.h"

CanteraGas::CanteraGas(const std::string& mechanismFile, const std::string& phaseID)
: soln(Cantera::newSolution(mechanismFile, phaseID, "mixture-averaged")),
thermo(soln->thermo().get()),
trans(soln->transport().get())
{
}

void CanteraGas::setTPY(double T, double P, const std::vector<double>& Y)
{
    thermo->setState_TPY(T, P, Y.data());
}

void CanteraGas::equiHP()
{
    thermo->equilibrate("HP");
}

std::vector<double> CanteraGas::getTP() const
{
    return std::vector{thermo->temperature(), thermo->pressure()}; 
}