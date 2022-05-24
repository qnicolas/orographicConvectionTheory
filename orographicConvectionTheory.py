#This file contains code to solve for the equations of mechanically-forced, convective orographic precipitation described in Nicolas & Boos, "A theory for the response of tropical moist convection to mechanical orographic forcing". It is meant to be run with Python3, using the packages imported below.

import numpy as np
from scipy.integrate import odeint
from scipy.special import expi

cp = 1004. # Heat capacity of air in J/kg/K
Lc = 2.5e6 # latent heat of condensation in J/kg
g = 9.81   # m/s2
Rv = 461   # Gas constant for water vapor, J/kg/K 
Rd = 287   # Gas constant for dry air, J/kg/K 

def k_vector(Nx,dx):
    """Given an x grid, return the grid of wavenumbers on which an FFT will be computed
    args:
     - Nx, number of points in grid
     - dx, grid spacing
    returns:
     - k, wavenumber array
    """
    return 2*np.pi*np.fft.fftfreq(Nx,dx)

def m_exponent(k,N,U):
    """Vectorized function to compute the vertical wavenumbers in linear mountain wave theory
    args:
     - k, wavenumber array
     - N, Brunt-Vaisala frequency
     - U, basic-state wind
    returns:
     - m, vertical wavenumber array
    """
    return - np.sign(k**2-N**2/U**2)**((np.sign(k+1e-4*k[1])+1)/2) * np.sqrt(k**2-N**2/U**2+0.j) #1e-4*k[1] it there to avoid fractional powers when k[i]=0

def compute_Lq(Ms_ov_M,U,tauq):
    tauqtilde = 0.6*tauq # conversion from full-tropospheric average to lower tropospheric average, see Ahmed et al 2020
    return Ms_ov_M*U*tauqtilde

def lapse_rates():
    """Compute the lapse rates and B-V frequency for input into the linear theory
    Assumptions : 
     * ds0/dz=3K/km
     * dq0dz computed from an exponentially decreasing profile starting from 80%RH at 300K and moisture scale height = 2.5km (=-8.1K/km), averages from 1000m to 3000 m"""
    zbot=1000
    ztop=3000

    ds0dz = cp * 3e-3 # 3K/km
    dq0dz = -Lc * 0.8 *0.022/2500 * 2500/(ztop-zbot) * (np.exp(-zbot/2500)-np.exp(-ztop/2500)) # 0.022 = qsat at 300K, used 80%RH and moisture scale height = 2.5km
    N=np.sqrt(g/300 * ds0dz/cp)

    return ds0dz,dq0dz,N

def topographic_profile(kind,a=100e3,A=1000):
    """Computes one of three default topographic profiles (Witch of Agnesi, Gaussian ridge & truncated cosine ridge)
    args:
     - kind, type of mountain shape (either agnesi, gaussian or cos)
     - a, mountain half-width in m
     - A, mountain height in m
    returns:
     - xx, x-grid in m
     - hx, topographic profile in m
     """
    xx=np.arange(-10000e3,20000e3,5e3)
    if kind=='gaussian':
        hx = A*np.exp(-xx**2/2/(a/2)**2)
    elif kind=='cos':
        hx = A/2*(1+np.cos(np.pi*np.maximum(-1,np.minimum(1,xx/a))))
    elif kind=='agnesi':
        hx = A*a**2/(xx**2+a**2)
    return xx,hx

#######################################
#  THEORIES DEVELOPPED IN THIS PAPER  #
#######################################

def linear_TdL_qdL(xx,hx,U):
    """Computes the lower-tropospheric averaged temperature and moisture perturbations predicted by the linear theory.
    Note that this function is not needed for the linear theory, (it computes an intermediate result) we only used it for Fig. 7.
    Assumptions : 
     * assumptions about lapse rates detailed in the function lapse_rates
     * Averages taken between z=1000m and z=3000m
    args:
     - xx, x-grid in m. Must be sorted in the order the wind is blowing (i.e. east to west if the wind is westward)
     - hx, topographic profile in m
     - U, basic-state wind in m/s
    returns:
     - TdL, lower-tropospheric averaged temperature perturbation in K
     - qdL, lower-tropospheric averaged moisture perturbation in K
     """
    cp = 1004. #Heat capacity of air in J/kg/K

    z=np.arange(0,10000,100)
    k=k_vector(len(xx),xx[1]-xx[0])

    zbot=1000
    ztop=3000
    z_slice = z[np.where((z>=zbot) & (z<=ztop))]

    ds0dz,dq0dz,N = lapse_rates()

    Tdhat = -ds0dz*np.fft.fft(hx)*np.exp( m_exponent(k[:,None],N,U)  *  z_slice[None,:]).mean(axis=1)
    qdhat = -dq0dz*np.fft.fft(hx)*np.exp( m_exponent(k[:,None],N,U)  *  z_slice[None,:]).mean(axis=1)

    return np.real(np.fft.ifft(Tdhat))/cp,np.real(np.fft.ifft(qdhat))/cp

def linear_precip_theory(xx,hx,U,tauT=3,tauq=11,P0=4.,switch=1):
    """Computes the precipitation profile predicted by the linear theory (equation (12) in Nicolas&Boos 2021).
    Assumptions : 
     * assumptions about lapse rates detailed in the function lapse_rates
     * Averages taken between z=1000m and z=3000m
     * Ms/M=5
     * pT/g=8000 kg/m2
    args:
     - xx, x-grid in m. Must be sorted in the order the wind is blowing (i.e. east to west if the wind is westward)
     - hx, topographic profile in m
     - U, basic-state wind in m/s
     - tauT, temperature adjustment time scale in hours
     - tauq, moisture adjustment time scale in hours
     - P0, basic-state precipitation in mm/day
     - switch, set to 0 to turn off the effect of Lq (equivalent to setting Lq=0, no precip relaxation).
    returns:
     - P, precipitation in mm/day
     """
    pT_ov_g = 8e3 #mass of troposphere in kg/m2
    
    z=np.arange(0,10000,100)
    k=k_vector(len(xx),xx[1]-xx[0])
    
    tauT*=3600
    tauq*=3600
    
    Lq=compute_Lq(5,U,tauq)
    
    ds0dz,dq0dz,N = lapse_rates()
    chi = pT_ov_g * (ds0dz/tauT - dq0dz/tauq)/ Lc * 86400
    
    zbot=1000
    ztop=3000    
    z_slice = z[np.where((z>=zbot) & (z<=ztop))]
    Pprimehat = 1j*k/(1j*k + switch*1/Lq) * chi * np.fft.fft(hx) * np.exp( m_exponent(k[:,None],N,U)  *  z_slice[None,:]).mean(axis=1) 
    
    P = P0 + np.real(np.fft.ifft(Pprimehat))
    P = np.maximum(0.,P)
    return P

def analytical_precip_agnesi(U,a=100e3,A=1000,tauT=3,tauq=11,P0 = 4.,switch=1):
    """Computes the precipitation profile predicted by the analytical expression for a Witch-of-Agnesi ridge (equation (15) in Nicolas&Boos 2021).
    Assumptions : 
     * assumptions about lapse rates detailed in the function lapse_rates
     * Averages taken between z=1000m and z=3000m
     * Ms/M=5
     * pT/g=8000 kg/m2
    args:
     - U, basic-state wind in m/s
     - a, mountain half-width in km
     - A, mountain height in m
     - tauT, temperature adjustment time scale in hours
     - tauq, moisture adjustment time scale in hours
     - P0, basic-state precipitation
     - switch, set to 0 to turn off the effect of Lq (equivalent to setting Lq=0, no precip relaxation).
    returns:
     - xx, x-grid in m. Must be sorted in the order the wind is blowing (i.e. east to west if the wind is westward)
     - P, precipitation in mm/day
     """
    pT_ov_g = 8e3 #mass of troposphere in kg/m2
    
    xx=np.arange(-10000e3,20000e3,5e3)
    
    tauT*=3600
    tauq*=3600
    
    Lq=compute_Lq(5,U,tauq)
    
    ds0dz,dq0dz,N = lapse_rates()
    chi = pT_ov_g * (ds0dz/tauT - dq0dz/tauq)/ Lc * 86400
    
    zbot=1000
    ztop=3000
    l=N/U
    s=(np.cos(l*zbot)-np.cos(l*ztop))/(l*ztop-l*zbot)
    c=(np.sin(l*ztop)-np.sin(l*zbot))/(l*ztop-l*zbot)

    P = P0 + chi * s * A*a *(-xx/(xx**2+a**2)+switch*1/Lq*np.exp(-xx/Lq)*expi(xx/Lq)) + chi * c * A*a**2/(xx**2+a**2)
    P = np.maximum(0.,P)
    return xx,P

def nonlinear_precip_theory(xx,TdL,qdL,U,tauT=3,tauq=11,P0=4.5,alpha=None):
    """Computes the precipitation profile predicted by the nonlinear theory (equation (6) in Nicolas&Boos 2021).
    In order to avoid having to finite-differentiate That and qhat, we solve an equivalent equation for <q_m>/tauq_tilde, then diagnose P = max(<q_m>/tauq_tilde + qdL/tauq - TdL/tauT,0). This equation can be derived from eqn (2) and reads d/dx(<q_m>/tauq_tilde) + 1/Lq * (P-P_0) = 0.
    Assumptions : 
     * Ms/M=5
     * pT/g=8000 kg/m2
    args:
     - xx, x-grid in m. Must be sorted in the order the wind is blowing (i.e. east to west if the wind is westward)
     - TdL, Lower-tropospheric averaged temperature perturbation in K
     - qdL, Lower-tropospheric averaged moisture perturbation in K
     - U, basic-state wind in m/s
     - tauT, temperature adjustment time scale in hours
     - tauq, moisture adjustment time scale in hours
     - P0, basic-state precipitation in mm/day
     - alpha, array giving relative variations of P0 (i.e. spatially-varying P0 is (P0 in argument)*(1+alpha)). If None, assumes P0 is constant, i.e. alpha = array of zeros
    returns:
     - P, precipitation in mm/day (defined on grid in argument, xgrid)
     """
    pT_ov_g = 8e3 #mass of troposphere in kg/m2
    
    if alpha is None: # Assume no variations in P0
        alpha = 0*np.array(xx)
    
    tauT*=3600
    tauq*=3600
    
    Lq=compute_Lq(5,U,tauq)

    xx=np.array(xx)
    qdL=cp*np.array(qdL)
    TdL=cp*np.array(TdL)
    conv=1/ Lc * 86400 * pT_ov_g # conversion factor from J/kg/s to kg/m2/day (=mm/day if density of water is 1000km/m2)

    #Solve for <q_m>/tauq_tilde
    def fun(qmb_ov_tauq_tilde,x):
        ix=np.argmin((xx-x)**2) # Get index of current location
        return -1/Lq* (np.maximum(0.,qmb_ov_tauq_tilde+(qdL*conv/tauq-TdL*conv/tauT)[ix])-P0*(1+alpha[ix]))
    qmb_ov_tauq_tilde = odeint(fun,P0,xx)[:,0]
    
    return np.maximum(0.,qmb_ov_tauq_tilde+qdL*conv/tauq-TdL*conv/tauT)

    
###################################
#  SMITH & BARSTAD (2004) THEORY  #
###################################

def humidsat(t,p):
    """computes saturation vapor pressure (esat), saturation specific humidity (qsat),
    and saturation mixing ratio (rsat) given inputs temperature (t) in K and
    pressure (p) in hPa.
    
    these are all computed using the modified Tetens-like formulae given by
    Buck (1981, J. Appl. Meteorol.)
    for vapor pressure over liquid water at temperatures over 0 C, and for
    vapor pressure over ice at temperatures below -23 C, and a quadratic
    polynomial interpolation for intermediate temperatures."""
    
    tc=t-273.16
    tice=-23
    t0=0
    Rd=287.04
    Rv=461.5
    epsilon=Rd/Rv


    # first compute saturation vapor pressure over water
    ewat=(1.0007+(3.46e-6*p))*6.1121*np.exp(17.502*tc/(240.97+tc))
    eice=(1.0003+(4.18e-6*p))*6.1115*np.exp(22.452*tc/(272.55+tc))
    #alternatively don't use enhancement factor for non-ideal gas correction
    #ewat=6.1121*exp(17.502*tc/(240.97+tc));
    #eice=6.1115*exp(22.452*tc/(272.55+tc));
    eint=eice+(ewat-eice)*((tc-tice)/(t0-tice))*((tc-tice)/(t0-tice))

    esat=(tc<tice)*eice + (tc>t0)*ewat + (tc>tice)*(tc<t0)*eint

    #now convert vapor pressure to specific humidity and mixing ratio
    rsat=epsilon*esat/(p-esat);
    qsat=epsilon*esat/(p-esat*(1-epsilon));
    
    return esat,qsat,rsat

def hw_cw(ts,ps,gamma,gamma_m):
    """Compute water vapor scale height and coefficient Cw for the Smith&Barstad (2004) model.
    args:
     - ts, surface temperature in K
     - ps, surface pressure in Pa
     - gamma, environmental lapse rate in K/m
     - gamma_m, moist-adiabatic lapse rate in K/m
    returns:
     - Hw, Water vapor scale height in m
     - Cw, Uplift sensitivity factor in kg/m^3
    """

    Hw = Rv*ts**2/(Lc*gamma)
    Cw = humidsat(ts,ps/100)[0]*100/Rd/ts*gamma_m/gamma
    return Hw,Cw

def smith_theory(xx,hx,U,gamma,gamma_m,ts=300.,ps=100000.,tau=2000, P0=4.):
    """Compute precipitation from the Smith&Barstad (2004) model. 
    If the environmental lapse rate is steeper than the moist adiabat (conditionally unstable environment), a dry static stability is used tu compute airflow dynamics.
    args:
     - xx, x-grid in m. Must be sorted in the order the wind is blowing (i.e. east to west if the wind is westward)
     - hx, topographic profile in m
     - U, basic-state wind in m/s
     - gamma, environmental lapse rate in K/m
     - gamma_m, moist-adiabatic lapse rate in K/m
     - ts, surface temperature in K
     - ps, surface pressure in Pa
     - tau, conversion and fallout time scale in s
     - P0, basic-state precipitation in mm/day
    returns:
     - Hw, Water vapor scale height
     - Cw, Uplift sensitivity factor in kg/m^3    
    """
    
    Hw,Cw = hw_cw(ts,ps,gamma,gamma_m)
    if gamma < gamma_m:
        N=np.sqrt(9.81/ts*(gamma_m-gamma))
        print("using moist stability, N=%.3f s^-1"%N)
    else:
        N=np.sqrt(9.81/ts*(9.81/1000-gamma))
        print("using dry stability, N=%.3f s^-1"%N)

    tau_c=tau
    tau_f=tau
    
    k=k_vector(len(xx),xx[1]-xx[0])
    
    P=np.maximum(P0+86400*np.real(np.fft.ifft(Cw*np.fft.fft(hx)*1j*U*k/(1-Hw*m_exponent(k,N,U))/(1+1j*U*k*tau_c)/(1+1j*U*k*tau_f))),0.)
    return xx,hx,P