import numpy as np
import sxs
import scipy
import qnm
import timeit

from scipy import signal
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from QNM_package import *
from overtone_analysis_package import *

def total_flux(w, max_ell=8, w_type='rMPsi4', bDebug=False):
    """
    Calculate the fluxes at a given timestep (currently only energy and Penrose scalar rMPsi4, nana - todo angular momentum components and strain h)
    Based on formulas 3.7-3.8 in arxiv:0707.4654 or 17, B2-B3 in arxiv:0610122
    """

    assert w_type in ['rMPsi4', 'rhOverM']
    assert max_ell < 9

    if bDebug:
        # for debugging we compare the for loop to the built-in .norm function which sums over all available (l,m) modes.
        max_ell = w.ell_max
    timediff = w.t[1:] - w.t[:-1]
    flux = 0

    if w_type == 'rMPsi4':
        for ell in range(2, max_ell + 1):
            for m in range(-ell, ell + 1):
                Dlm = (w[:, w.index(ell, m)][1:] * timediff).cumsum()
                flux += Dlm.abs ** 2
    elif w_type == 'rhOverM':
        for ell in range(2, max_ell + 1):
            for m in range(-ell, ell + 1):
                Hlm_der = w[:, w.index(ell, m)].derivative()
                flux += Hlm_der.abs ** 2
        if bDebug:
            flux_dbg = w.derivative().norm ** 2
            plt.figure(1)
            plt.plot(flux.t, flux)
            plt.plot(flux_dbg.t, flux_dbg)
            plt.show(block=True)
            plt.close()
    else:
        print('Invalid wavefunction type, should be either a strain (rhOverM) or a Penrose scalar (rMPsi4), '
              f'currently {w_type:s}.\n')
        assert False
    return flux / (16 * np.pi)

def swsh(s, lm_list, theta, phi, psi=0):
    import spherical_functions as sf
    import quaternion as qt
    """
    Given a list of modes as (l,m) pairs, returns a list of values of those
    spin-weighted spherical harmonic modes at a point.
    s = spin-weight
    lm_list = [(l,m)] or [(l1,m1),(l2,m2),...]
    theta = polar angle
    phi = azimuthal angle 
    psi = frame orientation
    """
    return sf.SWSH(qt.from_spherical_coords(theta, phi), s, lm_list) * np.exp(1j * s * psi)


def get_at_point(waveform_modes, lm_list, theta, phi, psi=0, s=-2):
    """
    Computes the value of a gravitational wave quantity at a point on the sphere.
    waveform_modes = array where each element is a 1D complex waveform of a single (l,m) mode
    lm_list = a list of all the (l,m) pairs in waveform_modes
    theta = polar angle
    phi = azimuthal angle
    psi = frame orientation
    s = spin-weight
    """
    waveform_at_point = np.empty(waveform_modes.shape[1], dtype=complex)
    # For each timestep, compute Eq. (1)
    for t in range(waveform_modes.shape[1]):
        waveform_at_point[t] = np.sum(
            waveform_modes[:, t] * swsh(s, lm_list, theta, phi, psi))
    return waveform_at_point


def calc_range_offset(signal):
    # Extract range as follows:
    # 1. Calculate t_peak as the point as maximal strain.
    # 2. Calculate the point the noise starts as where the 2nd derivative of the phase first goes positive.
    # 3. Choose the starting and ending indices to be 800 before t0 and 800 after the noise index.

    # Find the time of the merger.
    ind_peak = signal.argmax()
    phase_der = -signal[ind_peak:].arg_unwrapped.derivative(2)
    t_peak = signal.t[ind_peak]

    # Find the earliest point in the tail where numerical noise appears for the frequency (assume it is where the
    # frequency begins to decrease).
    ind_noise = ind_peak + np.argmax(phase_der < 0)
    return ind_peak - 800, ind_noise + 800, t_peak


def fit_func(t, amps, phases, mass, spin):
    """signal = 0
    N = 1
    assert N<7
    modes = [[2, 2, 0], [2, -2, 0], [3, 2, 0], [3, -2, 0], [4, 2, 0], [4, -2, 0]]
    for qnmI in range(N):
        l, m, n = modes[qnmI]
        LUT = load_QNM_mode_LUT(n, l, m)
        signal += amps * np.exp(-1j*freq_for_mode(LUT, l=l, m=m, n=n, mass=mass, spin=spin, method="LUT")*t
                                      + phases)
    return signal"""
    freq = freq_for_mode(l=2, m=2, n=0, mass=mass, spin=spin, method="qnmpy")#mass=0.954, spin=0.686, method="qnmpy")
    return amps * np.exp(np.imag(freq) * t + phases) * np.cos(np.real(freq) * t + phases)
    #return np.real(amps * np.exp(-1j * freq * t + phases))

def BCPmimic(signal, N=0):
    crop_ind = 900
    popt, pcov = curve_fit(fit_func, signal.t[crop_ind:]-signal.t[crop_ind],
                           np.real(signal.data[crop_ind:]),
                           bounds=([0, 0, 0.9, 0.5], [np.inf, 2*np.pi, 1, 1]))
    plt.figure(0)
    plt.plot(signal.t, np.real(signal.data))
    plt.plot(signal.t[800:], fit_func(signal.t[800:]-signal.t[crop_ind], *popt), 'g--')
    plt.axvline(x = signal.t[crop_ind], color='r')
    plt.show(block=True)
    plt.close()


def init_system():
    sxs.write_config(download=True, cache=True)
    qnm.download_data()

def main(bDebug=False, wavefunction_type='rhOverM', extrapolation_order=-1):
    catalog = sxs.load("catalog")

    # Extrapolation order can get any of the orders N2 to N4 (2-4) or OutermostExtractions (-1)
    # SXS:BBH:0305 represents GW150914
    # Outermost Extraction is the most accurate for the ringdown according to (cite the right paper from SXS references)

    assert wavefunction_type in ['rMPsi4', 'rhOverM'],\
        'Invalid wavefunction type, should be either a strain (rhOverM) or a Penrose scalar (rMPsi4), currently ' \
        f'{wavefunction_type:s}.\n'
    assert extrapolation_order in [-1, 2, 3, 4], \
        f'Invalid extrapolation order, should be a numerical in [-1,2,3,4], currently {extrapolation_order:d}.\n'

    QNM_mode_frequency_coefficients = init_QNM_mode_frequency_coeffs_analytic_database()

    sxs_event = "SXS:BBH:0305"  # usually 0305
    gw_sxs_bbh = sxs.load(f"{sxs_event:s}/Lev/{wavefunction_type:s}_Asymptotic_GeometricUnits_CoM.h5", extrapolation_order=extrapolation_order)

    remnant_mass = catalog.simulations[sxs_event].remnant_mass
    remnant_spin_mag = np.linalg.norm(catalog.simulations[sxs_event].remnant_dimensionless_spin)

    modes = [[2,2,0], [2,1,0], [3,3,0], [4,4,0]]
    signal = []
    ranges = []
    freq_num = []
    remnant_freq = []
    mass = []
    spin = []

    # Extraction of the leading l=m=2 mode

    for mode in modes:
        l, m, n = mode
        signalI = gw_sxs_bbh[:, gw_sxs_bbh.index(l, m)]

        s, e, t_peak = calc_range_offset(signalI)

        BCPmimic(signalI[s:e])

        freq_numI = freq_from_num_data(signalI[s:e], bDebug=False)

        remnant_freqI = freq_for_mode(QNM_mode_frequency_coefficients, l=l, m=m, n=n, mass=remnant_mass,
                                     spin=remnant_spin_mag, method='qnmpy')

        freq_filt = clean_noise_freq(freq_numI, 0.05)
        massI, spinI = props_from_freq(freq_filt, l=l, m=m, n=n, method="LUT")

        signal.append(signalI)
        ranges.append([s, e, t_peak])
        freq_num.append(freq_numI)
        remnant_freq.append(remnant_freqI)
        mass.append(massI)
        spin.append(spinI)

    flux = total_flux(gw_sxs_bbh, w_type=wavefunction_type)
    # amps = fit_mode_amp_to_num_data(1, gw_sxs_bbh_0305_2_2[ind_peak:ind_peak+ind_noise], QNM_mode_frequency_coefficients, t0=0, l=2, m=2, mass=remnant_mass, spin=remnant_spin_mag)
    # calc_err(gw_sxs_bbh_0305_2_2, amps, np.array([freq_remnant]), t0=t_peak, T=t_peak + 90)

    Goverc3s = 4.92703806e-6  # seconds / solar mass
    c = sxs.speed_of_light
    Mtot = 66.2 #solar masses

    #for modeI in range(len(modes)):
    for modeI in range(len(modes)):
        l, m, n = modes[modeI]
        s, e, t_peak = ranges[modeI]
        signalI = signal[modeI]
        freq_numI = freq_num[modeI]
        remnant_freqI = remnant_freq[modeI]
        massI = mass[modeI]
        spinI = spin[modeI]

        # Plotting the numeric wavefunction, extracted around the merger.
        plt.figure(1)
        plt.plot(signalI[s:e].t - t_peak, signalI[s:e].real, label=f"$r\, h_{{({l:d},{m:d})}}/M_{{tot}}$")
        #plt.plot((signalI[s-1500:e].t - t_peak)*Goverc3s*Mtot, signalI[s-1500:e].real, label=f"$r\, h_{{({l:d},{m:d})}}/M_{{tot}}$")

        # Plotting the frequency, extracted from the numeric wavefunction.
        plt.figure(2)
        plt.plot(freq_numI.t - t_peak, freq_numI.real, label=f"$\omega^R_{{{l:d},{m:d}}}[M_{{tot}}^{{-1}}]$")
        plt.plot(freq_numI.t - t_peak, remnant_freqI.real * np.ones((freq_numI.t.shape[0], 1)))
        freq_filt = clean_noise_freq(freq_numI, 0.05)
        plt.plot(freq_numI.t - t_peak, np.real(freq_filt))

        # Plotting the decay rate, extracted from the numeric wavefunction.
        plt.figure(3)
        plt.plot(freq_numI.t - t_peak, freq_numI.imag, label=f"$\omega^I_{{{l:d},{m:d}}}[M_{{tot}}^{{-1}}]$")
        plt.plot(freq_numI.t - t_peak, -remnant_freqI.imag * np.ones((freq_numI.t.shape[0], 1)))
        plt.plot(freq_numI.t - t_peak, freq_filt.imag)

        # Plotting the mass, extracted from the frequency.
        plt.figure(4)
        plt.plot(freq_numI.t - t_peak, massI, label=f"$r\, M_{{({l:d},{m:d})}}[M_{{tot}}]$")
        plt.plot(freq_numI.t - t_peak, remnant_mass * np.ones((freq_numI.t.shape[0], 1)))

        # Plotting the spin, extracted from the frequency and the decay rate.
        plt.figure(5)
        plt.plot(freq_numI.t - t_peak, spinI, label=f"$r\, \hat{{a}}_{{({l:d},{m:d})}}$")
        plt.plot(freq_numI.t - t_peak, remnant_spin_mag * np.ones((freq_numI.t.shape[0], 1)))

    plt.figure(1)
    plt.xlabel("$t-t_{peak}[M_{tot}]$")
    plt.ylabel('rh/M')
    plt.title(f"{sxs_event:s} (GW150914)")
    plt.legend()

    plt.figure(2)
    plt.xlabel("$t-t_{peak}[M_{tot}]$")
    plt.xlim((-75, 60))
    plt.ylim((-0.1, 0.6))
    plt.title(f"{sxs_event:s} $\omega_R$ (GW150914)")
    plt.legend()

    plt.figure(3)
    plt.xlabel("$t-t_{peak}[M_{tot}]$")
    plt.xlim((-75, 60))
    plt.ylim((-0.01, 0.095))
    plt.title(f"{sxs_event:s} $\omega_I$ (GW150914)")
    plt.legend()

    plt.figure(4)
    plt.xlabel(r"$t-t_{peak}[M_{tot}]$")
    plt.title(f"{sxs_event:s} remnant mass (GW150914)")
    plt.ylim((-0.01, 2))
    plt.legend()

    plt.figure(5)
    plt.xlabel(r"$t-t_{peak}[M_{tot}]$")
    plt.title(f'{sxs_event:s} remnant spin (GW150914)')
    plt.legend()

    # Plotting the flux to infinity, using the numerical integration of the strain derivative norm at infinity
    # (or equivalent using the Penrose scalar Psi4).
    t_peak = ranges[0][-1]
    plt.figure(6)
    plt.plot(flux.t - t_peak, flux)
    plt.xlabel(r"$t-t_{peak}[M_{tot}]$")
    plt.ylabel("$F[AU]$")
    #plt.xlim((spinI.t[0] - t_peak, spinI.t[-1] - t_peak))
    plt.ylim((-0.005 / (16 * np.pi), 0.06 / (16 * np.pi)))
    plt.title(f'{sxs_event:s} Energy Flux (GW150914)')

    # Plotting the flux to infinity, using the numerical integration of the strain derivative norm at infinity
    # (or equivalent using the Penrose scalar Psi4).

    t_peak = ranges[0][-1]
    plt.figure(8)
    plt.plot((flux.t[s-1500:e] - t_peak)*Goverc3s*Mtot, flux[s-1500:e]*Mtot, label=r"$F[M_{\odot}c^2]$")
    plt.plot((signalI[s - 1500:e].t - t_peak) * Goverc3s * Mtot, signalI[s - 1500:e].real*0.1+0.1,
             label=f"$r\, h_{{({l:d},{m:d})}}/M_{{tot}}$")
    plt.xlabel(r"$t-t_{peak}[sec]$")
    plt.ylabel("$F[M_{\odot}c^2]$")
    #plt.xlim((spinI.t[0] - t_peak, spinI.t[-1] - t_peak))
    #plt.ylim((-0.005 / (16 * np.pi), 0.06 / (16 * np.pi)))
    plt.title(f'{sxs_event:s} Energy Flux (GW150914)')
    plt.legend()

    plt.figure(7)
    plt.plot(flux.t - t_peak, flux.derivative())
    plt.xlabel(r"$t-t_{peak}[M_{tot}]$")
    plt.ylabel("$dM/dt$")
    plt.xlim((20, 37))
    plt.title(f"{sxs_event:s} Energy Flux (GW150914)")

    plt.show(block=True)
    plt.close()

if __name__ == "__main__":
    main()
