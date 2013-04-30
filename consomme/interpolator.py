import numpy as np
import ctypes as ct

def get_interp(gsl_lib_loc,func_path):
    """
    Return cublic spline interpolation function from library
    """
    ct.CDLL(gsl_lib_loc+'libgslcblas.so',mode=ct.RTLD_GLOBAL)
    ct.CDLL(gsl_lib_loc+'libgsl.so',mode=ct.RTLD_GLOBAL)
    dll = ct.CDLL(func_path)
    intp = dll.cubic_spline_interp_1d
    intp.argtypes = [ct.c_long,ct.c_long,ct.POINTER(ct.c_double),
                     ct.POINTER(ct.c_double),ct.POINTER(ct.c_double),
                     ct.POINTER(ct.c_double),ct.POINTER(ct.c_int)]
    return intp

def interpolate_spectrum(interp,wave_ini,flux_ini,wave_fnl,flux_fnl):
    """
    Interpolate a spectrum using cublc splines.  Interpolation 
    is done in memory.
    """
    wave_ini_p = wave_ini.ctypes.data_as(ct.POINTER(ct.c_double))
    flux_ini_p = flux_ini.ctypes.data_as(ct.POINTER(ct.c_double))
    wave_fnl_p = wave_fnl.ctypes.data_as(ct.POINTER(ct.c_double))
    flux_fnl_p = flux_fnl.ctypes.data_as(ct.POINTER(ct.c_double))

    mask = np.zeros_like(wave_fnl).astype('int32')
    mask_p = mask.ctypes.data_as(ct.POINTER(ct.c_int))

    interp(wave_ini.shape[0],wave_fnl.shape[0],
           wave_ini_p,flux_ini_p,
           wave_fnl_p,flux_fnl_p,mask_p)

    return mask
