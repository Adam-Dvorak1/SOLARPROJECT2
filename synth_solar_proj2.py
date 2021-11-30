#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Synthetic Solar Radiation Module
Copyright (C) 2014 Julius Susanto

Last edited: August 2014

Main Functions
--------------
- Aguiar_hourly_G0: generate annual sequence of hourly irradiances on a horizontal plane (W/m2)
- Aguiar_hourly_kt: generate sequence of hourly clearness indices for a single solar day
- Aguiar_daily_Kt: generate sequence of daily clearness indices given mean monthly Kt
- trend_sequence: generate annual sequence of hourly trend irradiances (no randomness)

Utility Functions
-----------------
- declination: calculate solar declination
- sunrise: calculate sunrise / sunset angle (in radians)
- eccentricity: calculate earth eccentricity correction factor 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""


import numpy as np


def declination(n):
    """
    Returns the solar declination (in radians) on the n-th day of the year using the accurate 
    approximation: 
        delta = -arcsin (0.39779 cos [0.98565(n+10) + 1.914 sin (0.98565 (n-2))])
    
    (note that the quantities inside the cosine and sine are in degrees)
    
    Inputs: n is the day of the year (where n=1 is midnight on January 1)
    """
    
    delta = -np.arcsin(0.39779 * np.cos(np.radians(0.98565 * (n+10) + 1.914 * np.sin (np.radians(0.98565 * (n-2))))))
    
    return delta
    
def sunrise(delta, lat):
    """
    Returns the sunrise / sunset angle (in radians) 
    
    Inputs: delta is the solar declination angle (in radians)
            lat is the latitude of the location (in radians)
    """    
    omega = np.arccos(-np.tan(delta) * np.tan(lat))
    
    return omega

def eccentricity(n):
    """
    Returns the earth's eccentricity correction factor according to Spencer's formula in the paper:
    J. W. Spencer, “Fourier series representation of the position of the Sun”, Search, Vol. 2, 1972
    
    Inputs: n is the day of the year (where n=1 is midnight on January 1)   
    """
    day_ang = 2 * np.pi * (n - 1) / 365
    epsilon = 1.00011 + 0.034221 * np.cos(day_ang) + 0.00128 * np.sin(day_ang) + 0.000719 * np.cos(2 * day_ang) + 0.000077 * np.sin(2 * day_ang)
    
    return epsilon
    
def Aguiar_daily_Kt(Ktm, Kt0, nd):
    """
    Generates a sequence of synthetic daily clearness indices (Kt) using the mean monthly clearness
    index as the input. The algorithm is based on the method by Aguiar et al in the paper:
    
    R. Aguiar and M. Collares-Pereira, "A simple procedure for the generation of sequences of 
      daily radiation values using Markov transition matrices", Solar Energy, vol. 40, 269-279, 1988
    
    Inputs: Ktm is the mean clearness index for the month
            Kt0 is initial clearness index (on the first day of the month)
            nd is the number of daily clearness indices to generate
    """

    # Markov Transition Matrices
    MTM_lib = {}
    MTM_states = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    MTM_min = [0.031, 0.058, 0.051, 0.052, 0.028, 0.053, 0.044, 0.085, 0.010, 0.319]
    MTM_max = [0.705, 0.694, 0.753, 0.753, 0.807, 0.856, 0.818, 0.846, 0.842, 0.865]

    # Kt <= 0.30
    MTM_lib[0] = np.matrix([[0.229,0.333,0.208,0.042,0.083,0.042,0.042,0.021,0.000,0.000],
                    [0.167,0.319,0.194,0.139,0.097,0.028,0.042,0.000,0.014,0.000],
                    [0.250,0.250,0.091,0.136,0.091,0.046,0.046,0.023,0.068,0.000],
                    [0.158,0.237,0.158,0.263,0.026,0.053,0.079,0.026,0.000,0.000],
                    [0.211,0.053,0.211,0.158,0.053,0.053,0.158,0.105,0.000,0.000],
                    [0.125,0.125,0.250,0.188,0.063,0.125,0.000,0.125,0.000,0.000],
                    [0.040,0.240,0.080,0.120,0.080,0.080,0.120,0.120,0.080,0.040],
                    [0.000,0.250,0.000,0.125,0.000,0.125,0.125,0.250,0.063,0.063],
                    [0.000,0.250,0.000,0.125,0.250,0.000,0.250,0.000,0.000,0.125],
                    [0.000,0.000,0.000,0.000,0.000,0.000,0.500,0.250,0.000,0.250]])

    # 0.30 < Kt <= 0.35
    MTM_lib[1] = np.matrix([[0.000,0.000,0.091,0.000,0.364,0.091,0.182,0.000,0.273,0.000],
                    [0.118,0.118,0.176,0.118,0.059,0.118,0.176,0.059,0.059,0.000],
                    [0.067,0.267,0.067,0.200,0.067,0.000,0.133,0.133,0.000,0.067],
                    [0.118,0.235,0.000,0.235,0.059,0.176,0.118,0.000,0.059,0.000],
                    [0.077,0.154,0.308,0.077,0.154,0.077,0.000,0.077,0.077,0.000],
                    [0.083,0.000,0.167,0.250,0.083,0.167,0.000,0.083,0.167,0.000],
                    [0.222,0.222,0.000,0.111,0.111,0.000,0.111,0.222,0.000,0.000],
                    [0.091,0.182,0.273,0.000,0.091,0.273,0.000,0.091,0.000,0.000],
                    [0.111,0.111,0.111,0.222,0.000,0.000,0.000,0.222,0.111,0.111],
                    [0.000,0.000,0.000,0.000,0.000,0.000,0.500,0.000,0.000,0.500]])

    # 0.35 < Kt <= 0.40
    MTM_lib[2] = np.matrix([[0.206,0.088,0.176,0.176,0.088,0.029,0.176,0.029,0.029,0.000],
                    [0.120,0.100,0.140,0.160,0.120,0.220,0.100,0.000,0.020,0.020],
                    [0.077,0.123,0.185,0.123,0.077,0.139,0.092,0.123,0.061,0.000],
                    [0.048,0.111,0.095,0.206,0.206,0.190,0.095,0.048,0.000,0.000],
                    [0.059,0.137,0.118,0.137,0.098,0.118,0.118,0.157,0.059,0.000],
                    [0.014,0.097,0.139,0.153,0.125,0.139,0.208,0.056,0.042,0.028],
                    [0.073,0.101,0.116,0.145,0.087,0.159,0.203,0.087,0.029,0.000],
                    [0.019,0.037,0.111,0.056,0.074,0.111,0.185,0.296,0.074,0.037],
                    [0.035,0.069,0.035,0.000,0.035,0.103,0.172,0.138,0.379,0.035],
                    [0.000,0.167,0.167,0.000,0.167,0.000,0.000,0.333,0.000,0.167]])

    # 0.40 < Kt <= 0.45                 
    MTM_lib[3] = np.matrix([[0.167,0.167,0.167,0.000,0.083,0.125,0.000,0.167,0.125,0.000],
                    [0.117,0.117,0.150,0.117,0.083,0.117,0.200,0.067,0.017,0.017],
                    [0.049,0.085,0.134,0.158,0.098,0.110,0.134,0.134,0.061,0.037],
                    [0.039,0.090,0.141,0.141,0.167,0.141,0.090,0.141,0.039,0.013],
                    [0.009,0.139,0.074,0.093,0.194,0.139,0.167,0.093,0.074,0.019],
                    [0.036,0.018,0.117,0.099,0.144,0.180,0.180,0.117,0.072,0.036],
                    [0.000,0.046,0.061,0.061,0.136,0.159,0.273,0.167,0.098,0.000],
                    [0.016,0.056,0.080,0.128,0.104,0.080,0.160,0.208,0.136,0.032],
                    [0.011,0.053,0.021,0.043,0.128,0.096,0.074,0.223,0.277,0.074],
                    [0.000,0.074,0.037,0.000,0.074,0.074,0.074,0.074,0.333,0.259]])              

    # 0.45 < Kt <= 0.50
    MTM_lib[4] = np.matrix([[0.120,0.200,0.160,0.120,0.120,0.120,0.080,0.000,0.040,0.040],
                    [0.100,0.080,0.120,0.140,0.140,0.200,0.180,0.040,0.000,0.000],
                    [0.046,0.114,0.068,0.171,0.125,0.171,0.080,0.159,0.057,0.011],
                    [0.015,0.061,0.084,0.099,0.191,0.153,0.153,0.115,0.115,0.015],
                    [0.024,0.030,0.098,0.098,0.165,0.195,0.195,0.140,0.043,0.012],
                    [0.015,0.026,0.062,0.124,0.144,0.170,0.170,0.222,0.062,0.005],
                    [0.000,0.013,0.045,0.108,0.112,0.175,0.188,0.224,0.117,0.018],
                    [0.008,0.023,0.054,0.066,0.093,0.125,0.191,0.253,0.183,0.004],
                    [0.006,0.022,0.061,0.033,0.067,0.083,0.139,0.222,0.322,0.044],
                    [0.000,0.046,0.091,0.091,0.046,0.046,0.136,0.091,0.273,0.182]])

    # 0.50 < Kt <= 0.55
    MTM_lib[5] = np.matrix([[0.250,0.179,0.107,0.107,0.143,0.071,0.107,0.036,0.000,0.000],
                    [0.133,0.022,0.089,0.111,0.156,0.178,0.111,0.133,0.067,0.000],
                    [0.064,0.048,0.143,0.048,0.175,0.143,0.206,0.095,0.079,0.000],
                    [0.000,0.022,0.078,0.111,0.156,0.156,0.244,0.167,0.044,0.022],
                    [0.016,0.027,0.037,0.069,0.160,0.219,0.230,0.160,0.075,0.005],
                    [0.013,0.025,0.030,0.093,0.144,0.202,0.215,0.219,0.055,0.004],
                    [0.006,0.041,0.035,0.064,0.090,0.180,0.337,0.192,0.049,0.006],
                    [0.012,0.021,0.029,0.035,0.132,0.123,0.184,0.371,0.082,0.012],
                    [0.008,0.016,0.016,0.024,0.071,0.103,0.159,0.270,0.309,0.024],
                    [0.000,0.000,0.000,0.000,0.059,0.000,0.059,0.294,0.412,0.176]])

    # 0.55 < Kt <= 0.60
    MTM_lib[6] = np.matrix([[0.217,0.087,0.000,0.174,0.130,0.087,0.087,0.130,0.087,0.000],
                    [0.026,0.079,0.132,0.079,0.026,0.158,0.158,0.132,0.158,0.053],
                    [0.020,0.020,0.020,0.040,0.160,0.180,0.160,0.200,0.100,0.100],
                    [0.025,0.013,0.038,0.076,0.076,0.139,0.139,0.266,0.215,0.013],
                    [0.030,0.030,0.050,0.020,0.091,0.131,0.162,0.283,0.131,0.071],
                    [0.006,0.006,0.013,0.057,0.057,0.121,0.204,0.287,0.185,0.064],
                    [0.004,0.026,0.037,0.030,0.093,0.107,0.193,0.307,0.167,0.037],
                    [0.011,0.009,0.014,0.042,0.041,0.071,0.152,0.418,0.203,0.041],
                    [0.012,0.022,0.022,0.038,0.019,0.050,0.113,0.281,0.360,0.084],
                    [0.008,0.024,0.039,0.039,0.063,0.039,0.118,0.118,0.284,0.268]])

    # 0.60 < Kt <= 0.65
    MTM_lib[7] = np.matrix([[0.067,0.133,0.133,0.067,0.067,0.200,0.133,0.133,0.067,0.000],
                    [0.118,0.059,0.059,0.059,0.059,0.118,0.118,0.235,0.118,0.059],
                    [0.000,0.024,0.024,0.049,0.146,0.073,0.195,0.244,0.195,0.049],
                    [0.026,0.000,0.026,0.026,0.053,0.184,0.263,0.184,0.237,0.000],
                    [0.014,0.000,0.042,0.056,0.069,0.097,0.139,0.306,0.278,0.000],
                    [0.009,0.009,0.052,0.069,0.052,0.112,0.215,0.285,0.138,0.060],
                    [0.009,0.009,0.026,0.017,0.094,0.099,0.232,0.283,0.210,0.021],
                    [0.010,0.014,0.016,0.019,0.027,0.062,0.163,0.467,0.202,0.019],
                    [0.004,0.007,0.031,0.017,0.033,0.050,0.086,0.252,0.469,0.050],
                    [0.000,0.000,0.015,0.046,0.031,0.046,0.077,0.123,0.446,0.215]])

    # 0.65 < Kt <= 0.70
    MTM_lib[8] = np.matrix([[0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.000,0.000],
                    [0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.000,0.000],
                    [0.000,0.000,0.000,0.000,0.000,0.000,0.250,0.250,0.500,0.000],
                    [0.000,0.000,0.000,0.000,0.250,0.000,0.000,0.375,0.250,0.125],
                    [0.000,0.000,0.000,0.083,0.000,0.167,0.167,0.250,0.333,0.000],
                    [0.000,0.000,0.042,0.042,0.042,0.083,0.083,0.292,0.292,0.125],
                    [0.000,0.000,0.032,0.000,0.000,0.032,0.129,0.387,0.355,0.065],
                    [0.000,0.000,0.000,0.038,0.038,0.075,0.047,0.340,0.415,0.047],
                    [0.004,0.004,0.007,0.007,0.011,0.030,0.052,0.141,0.654,0.089],
                    [0.000,0.000,0.000,0.000,0.061,0.061,0.030,0.030,0.349,0.470]])

    # Kt > 0.70
    MTM_lib[9] = np.matrix([[0.000,0.000,0.000,0.000,0.000,0.000,0.000,0.000,1.000,0.000],
                    [0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100,0.100],
                    [0.000,0.000,0.000,0.250,0.000,0.000,0.000,0.500,0.250,0.000],
                    [0.000,0.000,0.143,0.143,0.000,0.143,0.143,0.429,0.000,0.000],
                    [0.000,0.000,0.000,0.200,0.000,0.000,0.200,0.400,0.200,0.000],
                    [0.000,0.000,0.000,0.000,0.000,0.000,0.222,0.444,0.333,0.000],
                    [0.000,0.000,0.000,0.000,0.080,0.080,0.080,0.480,0.240,0.040],
                    [0.000,0.000,0.027,0.009,0.027,0.018,0.135,0.523,0.252,0.009],
                    [0.000,0.000,0.000,0.022,0.000,0.043,0.043,0.326,0.511,0.054],
                    [0.000,0.000,0.000,0.143,0.000,0.000,0.000,0.143,0.714,0.000]])
    
    # Determine the appropriate MTM based on the mean monthly Kt    
    MTM_index = np.digitize([Ktm], MTM_states)[0]
    MTM = MTM_lib[MTM_index]
    
    # Calculate states and step sizes
    min_state = MTM_min[MTM_index]                 
    max_state = MTM_max[MTM_index] 
    step_size = (max_state - min_state)/10
    states = np.arange(min_state, max_state, step_size)
    
    # Generate daily clearness indices for nd days
    Kti = Kt0
    Kt = [Kti]  
    for i in range(nd-1):
        MTM_row = np.digitize([Kti], states)[0]
        MTM_cum = np.ravel(np.cumsum(MTM[MTM_row - 1,:]))
        R = np.random.rand()
        new_state = np.digitize([R], MTM_cum)[0] + 1
        
        if new_state > 10:
            new_state = 10
        
        # Calculate interpolation factor
        if new_state == 1:
            k_interp = R / MTM_cum[new_state-1]
        else:
            k_interp = (R - MTM_cum[new_state-2]) / (MTM_cum[new_state-1]-MTM_cum[new_state-2])
        
        Kti = states[new_state-1] + k_interp * step_size
        Kt.append(Kti)
    
    return Kt

def Aguiar_hourly_kt(Kt, n, lat, max_iter):    
    """
    Generates a sequence of synthetic hourly clearness indices (kt) using the mean daily clearness 
    index (Kt) as the input. The algorithm is based on the method by Aguiar et al in the paper:
    
    R. Aguiar and M. Collares-Pereira, "TAG: A time-dependent, autoregressive Gaussian model 
      for generating synthetic hourly radiation", Solar Energy, vol. 49, 167-174, 1992
    
    Inputs: Kt is the mean clearness index for the day
            n is the day of the year (n=1 is midnight on January 1)
            lat is the latitude of the location (degrees)
            max_iter is the maximum number of iterations for each new kt
    """
    # Solar declination in radians
    delta = declination(n)  
    
    # Sunrise angle in radians
    omega = sunrise(delta, np.radians(lat))
    
    # Autocorrelation coefficient
    phi = 0.38 + 0.06 * np.cos(7.4*Kt - 2.5)
    
    # Calculate algorithm constants
    lmbda = -0.19 + 1.12 * Kt + 0.24 * np.exp(-8 * Kt)
    eta = 0.32 - 1.6 * (Kt - 0.5) ** 2
    kappa = 0.19 + 2.27 * Kt ** 2 - 2.51 * Kt ** 3
    A = 0.14 * np.exp (-20 * (Kt - 0.35) ** 2)
    B = 3 * (Kt - 0.45) ** 2 + 16 * Kt ** 5
    
    # Generate kt for each solar hour
    kt = []
    y = []
    for h in range(1,25):
        angle_start = (h - 13) * np.pi / 12     # Start of hour
        angle_end = (h - 12) * np.pi / 12       # End of hour   
        
        if (angle_start > -omega) and (angle_end < omega):
            # Clear sky clearness index
            kcs = 0.88 * np.cos(np.pi * (h - 12.5) / 30)
            
            # Angle at centre of hour  
            h_ang = (h - 12.5) * np.pi / 12  
            
            # Solar elevation/ altitude angle
            hs = np.arcsin (np.cos(h_ang) * np.cos(delta) * np.cos(np.radians(lat)) + np.sin(delta) * np.sin(np.radians(lat))) 
            
            # Average clearness index
            ktm = lmbda + eta * np.exp(-kappa / np.sin(hs))
            
            # Standard deviation
            sigma = A * np.exp (B * (1 - np.sin(hs)))
            
            # Generate new kt only if greater than 0 and less than clear sky kt
            kti = -1
            iter = 0
            while (kti < 0) or (kti > kcs):
                z = np.random.rand()
                r = sigma * (z ** 0.135 - (1 - z) ** 0.135) / 0.1975               
                yi = phi * y[h-2] + r
                kti = ktm + sigma * yi
                
                # Iteration control
                iter = iter + 1
                if iter > max_iter:
                    if kti < 0:
                        kti = 0
                    if kti > kcs:
                        kti = kcs
            
            kt.append(kti)
            y.append(yi)
        else:
            # For non-sunlight hours, set kt to zero
            kt.append(0)
            y.append(0)
    
    return kt
    
def trend_sequence(lat):
    """
    Generates an annual sequence of clear sky (extraterrestrial) hourly irradiance values 
    (on a horizontal plane) based on the annual trend for a given latitude. Estimation of the
    daily irradiation and hourly irradiance on a horizontal plane is based on the method outlined 
    in Chapter 20 of:
    
    A. Luque, S. Hegedus, “Handbook of Photovoltaic Science and Engineering”, Wiley, 2003
    
    Inputs: lat is the latitude of the location (in decimals)
    """
    
    n = np.arange(1,366)
    lat_rad = np.radians(lat)
    epsilon = eccentricity(n)
    delta = declination(n)
    omega = sunrise(delta, lat_rad)
    
    # Daily extraterrestrial irradiation (on a horizontal plane) Wh/m2/day
    # (Refer to Section 20.4 of Luque and Hegedus)
    B0d = 24 / np.pi * 1367 * epsilon * (omega * np.sin(delta) * np.sin(lat_rad) - np.cos(delta) * np.cos(lat_rad) * np.sin(-omega))
    
    # Set up hour angles
    h = np.arange(1,25)
    h_start = (h - 13) * np.pi / 12     # Start of hour
    h_end = (h - 12) * np.pi / 12       # End of hour
    h_ang = (h - 12.5) * np.pi / 12     # Centre of hour
    
    # Hourly clear sky irradiance (on a horizontal plane) W/m2
    G0c = []
    for d in range(1,366):
        omega_s = -omega[d-1]   # Sunrise angle for the day
        a = 0.409 - 0.5016 * np.sin(omega_s + 60 * np.pi/180)
        b = 0.6609 + 0.4767 * np.sin(omega_s + 60 * np.pi/180)
        
        # Hourly irradiance on a horizontal plane
        # (Refer to Section 20.5.2 of Luque and Hegedus)
        G0ci = np.pi / 24 * (np.cos(h_ang) - np.cos(omega_s)) / (omega_s * np.cos(omega_s) - np.sin(omega_s)) * (a + b * np.cos(h_ang)) * B0d[d-1]
        h_sunrise = np.digitize([omega_s], h_start)[0]
        h_sunset = np.digitize([-omega_s], h_end)[0]
        G0ci[0:h_sunrise] = 0
        G0ci[h_sunset:24] = 0
        G0c.extend(G0ci)
    
    return G0c

def Aguiar_hourly_G0(lat):
    """
    Generates an annual sequence of synthetic hourly irradiance values G0 (on a horizontal plane)
    based on monthly mean clearness indices. The methods proposed by Aguiar et al for the generation
    of synthetic daily and hourly irradiance values is used to create the sequence.
    
    Inputs: Ktm is an array of monthly mean clearness indices
            lat is the latitude of the location (in decimals)
    """
  
   
    
    # Generate hourly clearness indices for each hour in the year
    
    
    # Generate trend irradiances for each hour in the year
    G0c = trend_sequence(lat)
    
    # Calculate synthetic irradiance for each hour of the year
    G0 = np.dot(G0c,0.7)
    
    return G0

'''Adam Dvorak and Alex Sfetcu: Solar Energy Project 2 

This is where our code begins'''

import matplotlib.pyplot as plt
import pandas as pd
import datetime
from math import pi, sin, cos, sqrt


####MAKING THE PLOT FOR PART 1########

lat = 56.1629

G0 = Aguiar_hourly_G0(lat)
hours_in_2018 = pd.date_range('2018-01-01', '2018-12-31 23:00:00', freq = "H")
g0df = pd.DataFrame(data = G0, columns=["G0"])
g0df = g0df.set_index(hours_in_2018)
# plt.plot(G0)
# plt.show()



#At this point, we are able to model the global radiation on Aarhus.
#We now need to be able to to separate it out into direct and diffuse
#using a diffuse fraction F_d. This is calculated as cloud_cover/100.



######PART 2##########


df = pd.DataFrame(index = hours_in_2018)
#print(df)

def roundTime(dt=None, roundTo=60):
   """Round a datetime object to any time lapse in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 1 minute.
   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   """
   seconds = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + datetime.timedelta(0,rounding-seconds,-dt.microsecond)


def hour_rounder(t):
    
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
            +datetime.timedelta(hours=t.minute//30))




#read csv
weather_data = pd.read_csv("weather_station/weather_data.csv", sep= ";")
#set index

#drop nan values
weather_data = weather_data.apply(lambda row: row.dropna())
#"TimeStamp" column to datetime along the column
weather_data["TimeStamp"] = weather_data.apply(lambda row: pd.to_datetime(row['TimeStamp']), axis = 1)
#Round values in column to nearest hour
weather_data["TimeStamp"] = weather_data.apply(lambda row: hour_rounder(row['TimeStamp']), axis = 1)

weather_data = weather_data.drop_duplicates(subset = ["TimeStamp"], keep = 'last')
weather_data = weather_data.set_index("TimeStamp")



idx = pd.date_range('2018-01-01', '2018-12-31 23:00:00', freq = "H")

weather_data = weather_data.reindex(idx)

weather_data.iloc[[0]] = 0
weather_data.iloc[[8759]] = 0


weather_data = weather_data.interpolate()




#####

df['diffuse_frac'] = weather_data.apply(lambda row: row["Cloud"]/100, axis = 1)
########START HERE NEXT TIME
df['G0'] = g0df.apply(lambda row: row["G0"], axis = 1)
df['B0_direct'] = df.apply(lambda row: (1-row ['diffuse_frac']) * row['G0'], axis = 1)
df['D0_diffuse'] = df.apply(lambda row: row['diffuse_frac'] * row['G0'], axis = 1)
#print (df.head())
#I finally have a datatable with cloud cover. 


# diffuse = g0df['G0'] * df["diffuse_frac"]
# plt.plot(diffuse)
# plt.show()


# direct = g0df['G0'] * (1-df['diffuse_frac'])
# plt.plot(direct)
# plt.show()

#####NUMBER 5#######

#DIRECT RADIATION:
#B(0)/sin(gamma)* cos (theta), or 0, whichever is higher
#sin (gamma) = sin (delta)sin(phi) + cos(delta)cos(phi)cos(omega)
#gamma = solar altitude
#delta = declination
#phi = latitude = 56.1629 degrees north
#omega = true solar time
#alpha = angle from due south. They are pointing at 167 degrees, which means alpha = 13
#theta = incident angle--needs delta, phi, beta, omega, alpha


#We need functions for gamma, delta, and omega, and then we can find theta


#gamma function (solar altitude) will need delta, phi, omega


phi = 56.1629 #latitude

xi = 10.2039 #longitude

beta = 12 #angle of panel

alpha = 13 #angle of panel from south

df['hour'] = range(0, 8760)

df['hour_utc'] = df.apply(lambda row: row['hour'] % 24 , axis = 1)
df['day'] = df.apply(lambda row: row['hour'] //24 + 1, axis = 1)
df['B'] = df.apply(lambda row: (row['day'] - 81) * 360/364, axis = 1)
df['ET'] = df.apply (lambda row: 9.87 * sin(pi /180 * 2 * row['B']) - 7.53 * cos(pi/180 * row['B']) - 1.5 * sin(pi/180*row['B']), axis = 1)
df['solar_time'] = df.apply (lambda row: row['hour_utc'] + row['ET']/60 + xi/15, axis = 1)
df['omega'] = df.apply(lambda row: 15* (row['solar_time']-12), axis = 1) #true solar time
df ['delta'] = df.apply(lambda row: 23.45 * sin(pi/180 * 360 * (row['day']+284) / 365), axis = 1)
df['sin_gamma'] = df.apply (lambda row: sin(pi/180 * row['delta']) * sin(pi/180 * phi) + cos(pi/180 * row['delta']) * cos(pi/180 * phi) * cos(pi/180 * row['omega']), axis = 1)
df['cos_theta_prelim'] = df.apply (lambda row: sin(pi/180 * row['delta']) * sin(pi/180 * phi) * cos(pi/180 * beta)
-sin(pi/180*row['delta'])*cos(pi/180*phi) * sin(pi/180*beta) * cos(pi/180*alpha)
+cos(pi/180*row['delta'])*cos(pi/180*phi)*cos(pi/180*beta)*cos(pi/180*row['omega'])
+cos(pi/180*row['delta'])*sin(pi/180*phi)*sin(pi/180*beta)*cos(pi/180*alpha)*cos(pi/180*row['omega'])
+cos(pi/180*row['delta'])*sin(pi/180*alpha)*sin(pi/180*row['omega'])*sin(pi/180*beta), axis = 1)
df['cos_theta_final'] = df.apply (lambda row: row['cos_theta_prelim'] if row['cos_theta_prelim'] > 0 else 0, axis = 1)
df['B_final'] = df.apply(lambda row: row['cos_theta_final'] * row['B0_direct'] / row['sin_gamma'], axis = 1)
df['D_final'] = df.apply(lambda row: (1 + cos(pi/180 * beta))/2 * row['D0_diffuse'], axis = 1) #isotropic
df['R_final'] = df.apply(lambda row: 0.05 * row['G0'] * (1-cos(pi/180 * beta))/2, axis = 1)
df['G_adjust'] = df.apply(lambda row: row['B_final'] + row['D_final'] + row['R_final'], axis = 1)
df['G_final'] = df.apply(lambda row: 0 if row['G_adjust'] > 2000 or row['G_adjust'] < 0 else row['G_adjust'], axis = 1)

fig, ax = plt.subplots()
ax.plot(df['G_final'][744:912])
ax.set_title('Global irradiance on a tilted surface--February')
ax.set_xlabel('Hours')
ax.set_ylabel('Irradiance (W/m^2)')


plt.subplots_adjust(bottom=0.25)
plt.xticks(rotation = 45)
#fig.savefig('FebruaryGlobal.png')
#plt.show()


'''

fig1,ax1 = plt.subplots()
ax1.plot(df['G_final'][3624:3792])
ax1.set_title('Global irradiance on a tilted surface--June')
ax1.set_xlabel('Hours')
ax1.set_ylabel('Irradiance (W/m^2)')

plt.subplots_adjust(bottom=0.25)
plt.xticks(rotation = 45)
fig1.savefig('JuneGlobal.png')
plt.show()

'''

#This section is real data
real_data = pd.read_excel("pv_data/CTS Data Aflæsning Strom.xlsx", skiprows=2)

real_data.iloc[:] = real_data.iloc[::-1].values
real_data = real_data.drop(real_data.columns[-4:], axis=1)
real_data = real_data.drop(real_data.columns[[0, 2, 3, 4, 5]], axis=1)
real_data = real_data.set_index(['Dato'])

df2 = pd.DataFrame(index = pd.date_range('2018-02-01', '2018-12-31', freq = "D"))
df2 = df2.join(real_data)


df_hour = pd.DataFrame()
df_test = pd.DataFrame()
idx= pd.date_range('2018-02-01 01:00:00', '2018-12-31 23:00:00', freq = "h")



#we want the row function to be the bigger loop, and then iterate over columns
for i in range(len(df2)):
    df_test['Data'] = df2.apply(lambda col: col[i])
    df_hour = df_hour.append(df_test)

df_hour = df_hour[:-1]
df_hour = df_hour.set_index(idx)


#This section is modeled data

ns = 1000              # Numbers of solar cells
P = 255                # Power of 1 solarcell - 250 Watt

P_STC = (ns*P)/1000       # Total power in kW standard test conditions
gamma = -0.44
G_STC = 1000          # Stanard G.

power = pd.DataFrame(index = pd.date_range('2018-01-01', '2018-12-31 23:00:00', freq = "H"))
power['Modeled power'] = df['G_final']/G_STC * (1+(-0.0044*(weather_data['Temp']-25)))*P_STC

df_hour = df_hour.join(power)





d_avg = df_hour.groupby(pd.Grouper(freq = 'D')).mean()

w_avg = df_hour.groupby(pd.Grouper(freq='W')).mean()

m_avg = df_hour.groupby(pd.Grouper(freq='M')).mean()



df_hour["Difference squared"] = df_hour.apply(lambda row: abs(row['Data']-row['Modeled power'])**2, axis = 1)
d_avg["Difference squared"] = d_avg.apply(lambda row: abs(row['Data']-row['Modeled power'])**2, axis = 1)
w_avg["Difference squared"] = w_avg.apply(lambda row: abs(row['Data']-row['Modeled power'])**2, axis = 1)
m_avg["Difference squared"] = m_avg.apply(lambda row: abs(row['Data']-row['Modeled power'])**2, axis = 1)



relhourRMSE = sqrt(df_hour["Difference squared"].sum() / len(df_hour))/df_hour["Data"].mean()
reldayRMSE = sqrt(d_avg["Difference squared"].sum() / len(d_avg))/d_avg["Data"].mean()
relweekRMSE = sqrt(w_avg["Difference squared"].sum() / len(w_avg))/w_avg["Data"].mean()
relmonthRMSE = sqrt(m_avg["Difference squared"].sum() / len(m_avg))/m_avg["Data"].mean()



# df_day = df_hour["Data"].rolling(24).mean()
# df_day['Modeled Power'] = df_hour[]

print(relhourRMSE, reldayRMSE, relweekRMSE, relmonthRMSE)








#DIFFUSE RADIATION
#D(0) * (1+cos)



