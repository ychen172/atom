#pragma once

#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include "cantera/base/Solution.h"
#include "cantera/thermo/ThermoPhase.h"
#include "cantera/transport/Transport.h"

class CanteraGas
{
public:
    CanteraGas(const std::string& mechanismFile, const std::string& phaseID);
    void setTPY(double T, double P, const std::vector<double>& Y);
    void equiHP();
    std::vector<double> getTP() const;
protected:
    std::shared_ptr<Cantera::Solution> soln;
    Cantera::ThermoPhase* thermo;
    Cantera::Transport* trans;
};
