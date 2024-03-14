import numpy as np
import netCDF4 as nc
import os
import json
import datetime
from scipy import interpolate

# Helper functions

def ln_to_n(sd_ln, mean_ln):
    """
     This is a function used to convert between log-normal and normal distributions.
    """
    term = 1.0 + sd_ln * sd_ln / mean_ln / mean_ln
    return (np.sqrt(np.log(term)), np.log(mean_ln / np.sqrt(term)))

def n_to_ln(sd_n, mean_n):
    """
     This is a function used to convert between normal and log-normal distributions.
    """
    return ((np.exp(sd_n * sd_n) - 1.0) * np.exp(2.0 * mean_n + sd_n * sd_n),
             np.exp(mean_n + sd_n * sd_n / 2.0))

# Helper function to serialize / deserialize random state with json

def rnd_state_serialize():
    """
     The function serializes the current state of the NumPy random number generator to a 
     JSON file called "rnd_state.json".
    """
    # retrieve the current state of the NumPy random number generator
    tmp_state = np.random.get_state()

    # initialize a tuple to hold the serialized state
    save_state = ()

    # loop over each element of the state
    for i in tmp_state:
            # check if the element is a NumPy array
            if type(i) is np.ndarray:
                    # convert the NumPy array to a list and add it to the save_state tuple
                    save_state = save_state + (i.tolist(),)
            else:
                    # if not, it is added to the save_state tuple as is
                    save_state = save_state + (i,)
    # write the serialized state to a JSON file called "rnd_state.json".
    json.dump(save_state, open("rnd_state.json", "w"))

def rnd_state_deserialize():
    """
     The function that deserializes the state of the NumPy random number generator from the 
     "rnd_state.json" file.
    """
    # read the serialized state from the "rnd_state.json" file 
    tmp_state = json.load(open("rnd_state.json", "r"))

    # initialize a tuple to hold the deserialized state
    load_state = ()

    # loop over each element of the serialized state 
    for i in tmp_state:
        # check if the element is a list
        if type(i) is list:
                # convert the list to a NumPy array and add it to the load_state tuple
                load_state = load_state + (np.array(i),)
        else:
                # if not, it is added to the load_state tuple as is
                load_state = load_state + (i,)
    # set the state of the NumPy random number generator to the deserialized state.
    np.random.set_state(load_state)

# Helper function to calculate downward longwave radiation flux 
# from pressure, relative humidity and temperature. 
# Same function as done in clm 3.5 if no FLDS is present in atm forcing files.
# see clm3.5/src/main/atmdrvMod.f90 line 939ff
##No need if FLDS is there

def tdc(t):
    return(np.minimum(50.0, np.maximum(-50.0, t - 273.15)))#

def esatw(t):
    return(100.0 * (6.107799961 + t * (4.436518521e-1 + t * 
          (1.428945805e-2 + t * (2.650648471e-4 + t * (3.031240396e-6 + t * 
          (2.034080948e-8 + t * 6.136820929e-11)))))))

def esati(t):
    return(100.0 * (6.109177956 + t * (5.034698970e-1 + t * 
          (1.886013408e-2 + t * (4.176223716e-4 + t * (5.824720280e-6 + t * 
          (4.838803174e-8 + t * 1.838826904e-10)))))))

def clm3_5_flds_calc(psrf, rh, t):
    vp = np.where(t > 273.15, rh / 100.0 * esatw(tdc(t)), rh / 100.0 * esati(tdc(t)))
    # Could just use this "vp" - but in clm 3.5 if RH is given it is converted like this.
    q = 0.622 * vp / (psrf - 0.378 * vp)
    e = psrf * q / (0.622 + 0.378 * q)
    ea = 0.7 + 5.95e-5 * 0.01 * e * np.exp(1500.0 / t)
    return (ea * 5.67e-8 * t**4)

# Helper function - copy attributes and dimensions
def copy_attr_dim(src, dst):
    """
     The function that copies the attributes and dimensions from a source to a destination.

     Args:
     src: source of NetCDF file
     dst: destination of NetCDF file
    """
    # copy attributes
    for name in src.ncattrs():
        dst.setncattr("original_attribute_" + name, src.getncattr(name))
    # copy dimensions
    for name, dimension in src.dimensions.items():
            dst.createDimension( name, len(dimension))
    # Additional attribute
    dst.setncattr("perturbed_by", "F.B. Eloundou")
    dst.setncattr("perturbed_on_date", datetime.datetime.today().strftime("%d.%m.%y"))


def interpolate_soil_properties(src):
    # Original PCT_SAND and PCT_CLAY values
    sand_original = src.variables["PCT_SAND"][:]
    clay_original = src.variables["PCT_CLAY"][:]

    # Create an array of indices for the original data
    old_indices = np.arange(sand_original.shape[0])

    # Create an array of indices for the interpolated data
    new_indices = np.linspace(sand_original.shape[0], 24, 25 - sand_original.shape[0])

    # Interpolate the data
    sand_interpolated = np.zeros((25,1,1))
    clay_interpolated = np.zeros((25,1,1))

    for i in range(sand_original.shape[1]):
        for j in range(sand_original.shape[2]):
            f_sand = interpolate.interp1d(old_indices, sand_original[:,i,j], kind='previous', fill_value="extrapolate")
            f_clay = interpolate.interp1d(old_indices, clay_original[:,i,j], kind='previous', fill_value="extrapolate")
            sand_interpolated[:sand_original.shape[0],i,j] = sand_original[:,i,j]
            clay_interpolated[:clay_original.shape[0],i,j] = clay_original[:,i,j]
            sand_interpolated[sand_original.shape[0]:,i,j] = np.round(f_sand(new_indices))
            clay_interpolated[clay_original.shape[0]:,i,j] = np.round(f_clay(new_indices))

    return sand_interpolated, clay_interpolated

