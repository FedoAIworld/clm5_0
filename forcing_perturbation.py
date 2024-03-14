#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import netCDF4 as nc
import os

from perturb_helper_funcs import *


# Simulation period and number of ensembles
years = list(range(2009, 2019))
num_ensemble = 110


def perturb_nc_file(year, month, iensemble):
    # standard deviation and mean of perturbations
    sd = [ln_to_n(0.6, 1.0)[0], ln_to_n(0.5, 1.0)[0], 20.0, 2.0, ln_to_n(0.4, 1.0)[0], 5.0]
    mean = [ln_to_n(0.6, 1.0)[1], ln_to_n(0.5, 1.0)[1], 0.0, 0.0, ln_to_n(0.4, 1.0)[1], 0.0]

    # Correlation matrix based on:
    # Correlation study using COSMOSREA6 daily time series data for 10 years
    # Across 71 European measurement sites.
    # Data assumes stationarity (mean, variance, and autocorrelation is constant over time) 
    # Precipitation | ShortWave Radiation | LongWave Radiation | Air Temperature at 2m | Wind Speed | RH
    correl_spatial = np.array([[ 1.00,  -0.27,  0.26,  0.00,  0.20,  0.31],
                               [-0.27,   1.00,  0.20,  0.65, -0.20, -0.68],
                               [ 0.26,   0.20,  1.00,  0.76,  0.00,  0.00],
                               [ 0.00,   0.65,  0.76,  1.00,  0.00, -0.44],
                               [ 0.20,  -0.20,  0.00,  0.00,  1.00,  0.00],
                               [ 0.31,  -0.68,  0.00, -0.44,  0.00,  1.00]])
   

    fname = ("/home/fernand/JURECA/CLM5_DATA/inputdata/atm/datm7/SE-Svb/" +
            str(year) + "-" + str(month).zfill(2) + ".nc")
    outname = ("/home/fernand/JURECA/CLM5_DATA/inputdata/atm/datm7/SE-Svb/" +
               "Ensemble/real_" + 
               str(iensemble + 1).zfill(5) + "/" +
               str(year) + "-" + str(month).zfill(2) + ".nc")
               
    os.makedirs(os.path.dirname(outname), exist_ok=True)

    with nc.Dataset(fname) as src, nc.Dataset(outname, "w") as dst:

        copy_attr_dim(src, dst)

        dim_time = src.dimensions["time"].size
        dim_lat  = src.dimensions["lat"].size
        dim_lon  = src.dimensions["lon"].size

        # Perturbations:
        # Use built-in multivariate, correlated pseudo-number generator directly
        # instead of Cholesky decomposition and matrix-matrix multiplication manually
        rnd_spatial  = np.random.multivariate_normal(mean, correl_spatial, dim_time*dim_lat*dim_lon)
        

        # Precipitation, ShortWave, and Wind Speed are log-normal distributed perturbations
        # therefore exponential from normal distributed pseudo-random number.
        perturbations = np.zeros_like(rnd_spatial)
        perturbations[:, 0] = np.exp(mean[0] + sd[0] * rnd_spatial[:, 0])
        perturbations[:, 1] = np.exp(mean[1] + sd[1] * rnd_spatial[:, 1])
        perturbations[:, 2] = rnd_spatial[:, 2] * sd[2]
        perturbations[:, 3] = rnd_spatial[:, 3] * sd[3]
        perturbations[:, 4] = np.exp(mean[4] + sd[4] * rnd_spatial[:, 4])
        perturbations[:, 5] = rnd_spatial[:, 5] * sd[5]

        # Scale the perturbations for precipitation, shortwave radiation, and wind speed
        # Divide by their respective means
        mean_precip    = np.mean(perturbations[:, 0])
        mean_shortwave = np.mean(perturbations[:, 1])
        mean_wind      = np.mean(perturbations[:, 4])

        perturbations[:, 0] = np.divide(perturbations[:, 0], mean_precip)
        perturbations[:, 1] = np.divide(perturbations[:, 1], mean_shortwave)
        perturbations[:, 4] = np.divide(perturbations[:, 4], mean_wind)

        
        # After generating all random variables
        # save state of random number generator to file
        if not force_seed:
            rnd_state_serialize()

        # netCDF variables
        # Copy non-perturbed variables:
        for name, var in src.variables.items():
            if name != "TBOT" and name != "PRECTmms" and name != "FSDS" and name != "FLDS" and name != "WIND" and name != "RH":
                nvar = dst.createVariable(name, var.datatype, var.dimensions)
                dst[name].setncatts(src[name].__dict__)
                dst[name][:] = src[name][:]

        # Add / multiply perturbations
        tbot = dst.createVariable("TBOT", datatype=np.float64,
                                    dimensions=("time", "lat", "lon",), 
                                    fill_value=9.96921e+36)
        tbot.setncatts({"height": u"2", "units": u"K", 
                        "missing_value": -9.e+33})
        dst.variables["TBOT"][:, :, :] = (src.variables["TBOT"][:, :, :] + 
                   perturbations[:, 3].reshape(src.variables["TBOT"][:, :, :].shape))

        prectmms = dst.createVariable("PRECTmms", datatype=np.float64,
                                    dimensions=("time", "lat", "lon",), 
                                    fill_value=-9.e+33)
        prectmms.setncatts({"units": u"mm/s", 
                            "missing_value": -9.e+33})
        dst.variables["PRECTmms"][:] = (src.variables["PRECTmms"][:] *
                                        perturbations[:, 0].reshape(
                                        src.variables["PRECTmms"][:, :, :].shape))

        fsds = dst.createVariable("FSDS", datatype=np.float64,
                                    dimensions=("time", "lat", "lon",), 
                                    fill_value=-9.e+33)
        fsds.setncatts({"units": u"W/m^2", "missing_value": -9.e+33})
        dst.variables["FSDS"][:] = (src.variables["FSDS"][:] *
                                    perturbations[:, 1].reshape(
                                    src.variables["FSDS"][:, :, :].shape))
        
        wind = dst.createVariable("WIND", datatype=np.float64,
                                    dimensions=("time", "lat", "lon",), 
                                    fill_value=-9.e+33)
        wind.setncatts({"missing_value": -9.e+33})
        dst.variables["WIND"][:] = (src.variables["WIND"][:] *
                                    perturbations[:, 4].reshape(
                                    src.variables["WIND"][:, :, :].shape))
        
        rh = dst.createVariable("RH", datatype=np.float64,
                                    dimensions=("time", "lat", "lon",), 
                                    fill_value=-9.e+33)
        rh.setncatts({"missing_value": -9.e+33})
        dst.variables["RH"][:] = np.clip(src.variables["RH"][:] + 
                                         perturbations[:, 5].reshape(
                                         src.variables["RH"][:, :, :].shape), 0, 100)  # Clip values between 0 and 100
        
        flds = dst.createVariable("FLDS", datatype=np.float64,
                                    dimensions=("time", "lat", "lon",), 
                                    fill_value=-9.e+33)
        flds.setncatts({"units": u"W/m^2", "missing_value": -9.e+33})
        
        dst.variables["FLDS"][:] = (clm3_5_flds_calc(src.variables["PSRF"][:],
                                                     src.variables["RH"][:],
                                                     src.variables["TBOT"][:]) + 
                                                     perturbations[:, 2].reshape(
                                                     src.variables["FSDS"][:, :, :].shape))

# Settings / parameters
rnd_state_file = "rnd_state.json"
force_seed = False 
# Either seed random number generator or continue with existing state
if not os.path.isfile(rnd_state_file) or force_seed:
    np.random.seed(42)
else:
    rnd_state_deserialize()

for ens in range(num_ensemble):
    for y in years:
        print("Done with year " + str(y) + " ensemble " + str(ens))
        for m in range(1, 13):
            perturb_nc_file(y, m, ens)