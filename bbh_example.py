import numpy as np
from matplotlib import pyplot as plt
import sxs
from QNM_package import *
from overtone_analysis_package import *

# For playing audio in the notebook
from IPython.display import Audio

def total_flux(w, max_ell = 8, w_type = 'rMPsi4',  bDebug = False):
    """
    Calculate the fluxes at a given timestep (currently only energy and Penrose scalar rMPsi4, momo - todo angular momentum components and strain h)
    Based on formulas 3.7-3.8 in arxiv:0707.4654 or 17, B2-B3 in arxiv:0610122
    """

    assert w_type in ['rMPsi4', 'rhOverM']
    assert max_ell < 9

    if(bDebug): #for debugging we compare the for loop to the built-in .norm function which sums over all available (l,m) modes.
        max_ell = w.ell_max
    timediff = w.t[1:]-w.t[:-1]
    flux = 0

    if w_type == 'rMPsi4':
        for ell in range(2, max_ell+1):
            for m in range(-ell, ell+1):
                Dlm = (w[:,w.index(ell, m)][1:]*timediff).cumsum()
                flux += Dlm.abs**2
    elif w_type == 'rhOverM':
        for ell in range(2, max_ell+1):
            for m in range(-ell, ell+1):
                Hlm_der = w[:,w.index(ell, m)].derivative()
                flux += Hlm_der.abs**2
        if(bDebug):
            flux_dbg = w.derivative().norm**2
            plt.figure(1)
            plt.plot(flux.t, flux)
            plt.plot(flux_dbg.t, flux_dbg)
            plt.show(block=True)
            plt.close()
    else:
        print(f'Invalid wavefunction type, should be either a strain (rhOverM) or a Penrose scalar (rMPsi4), currently {w_type:s}.\n')
        assert False
    return flux/(16*np.pi)

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
    return sf.SWSH(qt.from_spherical_coords(theta, phi), s, lm_list)*np.exp(1j*s*psi)


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
            waveform_modes[:, t]*swsh(s, lm_list, theta, phi, psi))
    return waveform_at_point

def init_system():
    sxs.write_config(download=True, cache=True)

def main(bDebug = False, wavefunction_type = 'rhOverM', extrapolation_order = -1):
    catalog = sxs.load("catalog")

    # Extrapolation order can get any of the orders N2 to N4 (2-4) or OutermostExtractions (-1)
    # SXS:BBH:0305 represents GW150914
    # Outermost Extraction is the most accurate for the ringdown according to (cite the right paper from SXS references)

    assert wavefunction_type in ['rMPsi4', 'rhOverM'], f'Invalid wavefunction type, should be either a strain (rhOverM) or a Penrose scalar (rMPsi4), currently {wavefunction_type:s}.\n'
    assert extrapolation_order in [-1, 2, 3, 4], f'Invalid extrapolation order, should be a numerical in [-1,2,3,4], currently {extrapolation_order:d}.\n'


    gw_sxs_bbh_0305 = sxs.load(f"SXS:BBH:0305/Lev/{wavefunction_type:s}", extrapolation_order=extrapolation_order)

    QNM_mode_frequency_coefficients = init_QNM_mode_frequency_coeffs_analytic_database()
    remnant_mass = catalog.simulations["SXS:BBH:0305"].remnant_mass
    remnant_spin_mag = np.linalg.norm(catalog.simulations["SXS:BBH:0305"].remnant_dimensionless_spin)

    #Extraction of the leading l=m=2 mode
    l, m = 2, 2
    gw_sxs_bbh_0305_ell_m = gw_sxs_bbh_0305[:, gw_sxs_bbh_0305.index(l, m)]

    # Find the time of the merger (momo - todo, check the git page for built-in function for it).
    ind_peak = gw_sxs_bbh_0305_ell_m.argmax()
    t_peak = gw_sxs_bbh_0305_ell_m.t[ind_peak]
    phase_der = -gw_sxs_bbh_0305_ell_m[ind_peak:].arg_unwrapped.derivative(2)

    # Find the earliest point in the tail where numerical noise appears for the frequency (assume it is where the frequency begins to decrease).
    ind_noise = ind_peak+np.argmax(phase_der < 0)
    freq_num = freq_from_num_data(gw_sxs_bbh_0305_ell_m[ind_peak-800:ind_noise+800])
    remnant_freq = freq_for_mode(QNM_mode_frequency_coefficients, l=l, m=m, n=0, mass=remnant_mass,
                                 spin=remnant_spin_mag)
    remnant_freq_LUT = freq_for_mode(QNM_mode_frequency_coefficients, l=l, m=m, n=0, mass=remnant_mass,
                                 spin=remnant_spin_mag, method='LUT')
    mass, spin = props_from_freq(QNM_mode_frequency_coefficients, freq_num, l=l, m=m, n=0, method="LUT")
    flux = total_flux(gw_sxs_bbh_0305, w_type=wavefunction_type)
    #amps = fit_mode_amp_to_num_data(1, gw_sxs_bbh_0305_2_2[ind_peak:ind_peak+ind_noise], QNM_mode_frequency_coefficients, t0=0, l=2, m=2, mass=remnant_mass, spin=remnant_spin_mag)
    #calc_err(gw_sxs_bbh_0305_2_2, amps, np.array([freq_remnant]), t0=t_peak, T=t_peak + 90)

    if(bDebug):
        total_mass_SXS0305 = 66.2 # m_sun
        Goverc2m = sxs.m_sun_in_meters # meters / solar mass
        Goverc3s = sxs.m_sun_in_seconds # seconds / solar mass
        #Plot graphs is SI (i.e. seconds and meters)
        plt.figure(3)
        plt.plot(freq.t*Goverc3s*total_mass_SXS0305, freq.real/(Goverc3s*total_mass_SXS0305*2*np.pi))
        plt.xlabel("time[sec]")
        plt.ylabel("frequency [Hz]")
        plt.figure(4)
        plt.plot(freq.t*Goverc3s*total_mass_SXS0305, 1/freq.imag*Goverc3s)
        plt.xlabel("time[sec]")
        plt.ylabel("decay time [sec]")

    # Ploting the numric wavefunction, extracted around the merger.
    plt.figure(1)
    plt.plot(gw_sxs_bbh_0305_ell_m[ind_peak-800:ind_noise+800].t - t_peak,
             gw_sxs_bbh_0305_ell_m[ind_peak-800:ind_noise+800].real)
    plt.xlabel("$t-t_{peak}[M_{tot}]$")
    plt.ylabel(f"$r\, h_{{({l:d},{m:d})}}/M_{{tot}}$")
    plt.title("SXS:BBH:0305 (GW150914)")

    # Ploting the frequency, extracted from the numeric wavefunction.
    plt.figure(2)
    plt.plot(freq_num.t - t_peak, freq_num.real)
    plt.plot(freq_num.t - t_peak, remnant_freq_LUT.real* np.ones((freq_num.t.shape[0], 1)))
    plt.plot(freq_num.t - t_peak, remnant_freq.real*np.ones((freq_num.t.shape[0], 1)))
    plt.xlabel("$t-t_{peak}[M_{tot}]$")
    plt.ylabel(f"$\omega^R_{{{l:d},{m:d}}}[M_{{tot}}^{{-1}}]$")
    plt.xlim((-75,60))
    plt.ylim((-0.1,0.6))
    plt.title("SXS:BBH:0305 $\omega_R$ (GW150914)")

    # Ploting the decay rate, extracted from the numeric wavefunction.
    plt.figure(3)
    plt.plot(freq_num.t - t_peak, freq_num.imag)
    plt.plot(freq_num.t - t_peak, -remnant_freq_LUT.imag*np.ones((freq_num.t.shape[0], 1)))
    plt.plot(freq_num.t - t_peak, -remnant_freq.imag*np.ones((freq_num.t.shape[0], 1)))
    plt.xlabel("$t-t_{peak}[M_{tot}]$")
    plt.ylabel(f"$\omega^I_{{{l:d},{m:d}}}[M_{{tot}}^{{-1}}]$")
    plt.xlim((-75,60))
    plt.ylim((-0.01,0.095))
    plt.title("SXS:BBH:0305 $\omega_I$ (GW150914)")

    # Plotting the mass, extracted from the frequency.
    plt.figure(4)
    plt.plot(mass.t - t_peak, mass)
    plt.plot(mass.t - t_peak, remnant_mass*np.ones((mass.t.shape[0], 1)))
    plt.xlabel(r"$t-t_{peak}[M_{tot}]$")
    plt.ylabel(r"$M_{rem}[M_{tot}]$")
    plt.xlim(20,37)
    plt.ylim(0.95,1.05)
    plt.title("SXS:BBH:0305 remnant mass (GW150914)")

    # Ploting the spin, extracted from the frequency and the decay rate.
    plt.figure(5)
    plt.plot(spin.t - t_peak, spin)
    plt.plot(spin.t - t_peak, remnant_spin_mag*np.ones((spin.t.shape[0], 1)))
    plt.xlabel(r"$t-t_{peak}[M_{tot}]$")
    plt.ylabel(r"$\hat{a}_{rem}$")
    plt.xlim(-5,37)
    plt.ylim(0.6,1.1)
    plt.title("SXS:BBH:0305 remnant spin (GW150914)")

    # Ploting the flux to infinity, using the numerical integration of the strain derivative norm at infinity
    # (or equivalent using the Penrose scalar Psi4).
    plt.figure(6)
    plt.plot(flux.t - t_peak, flux)
    plt.xlabel(r"$t-t_{peak}[M_{tot}]$")
    plt.ylabel("$dE/dt$")
    plt.xlim((spin.t[0] - t_peak,spin.t[-1] - t_peak))
    plt.ylim((-0.005/(16*np.pi),0.06/(16*np.pi)))
    plt.title("SXS:BBH:0305 Energy Flux (GW150914)")


    plt.figure(7)
    plt.plot(flux.t - t_peak, flux.derivative())
    plt.xlabel(r"$t-t_{peak}[M_{tot}]$")
    plt.ylabel("$dM/dt$")
    plt.xlim((20,37))
    plt.title("SXS:BBH:0305 Energy Flux (GW150914)")

    plt.show(block=True)
    plt.close()

    """
    ret_time = gw_sxs_bbh_0305[:,0]
    waveform_modes = np.array([gw_sxs_bbh_0305[:,1] + 1j*gw_sxs_bbh_0305[:,2]])
    plt.figure(2)
    plt.plot(freq[ind_peak:ind_peak + ind_noise].t - freq[ind_peak + 800].t,
             freq[ind_peak:ind_peak + ind_noise].real)lm_list = [(2,2)]

    for theta in [0,0.25,0.4,0.5]:
        plt.plot(ret_time, np.real(get_at_point(waveform_modes, lm_list, theta*np.pi, 0.0*np.pi)), label='$theta='+str(theta)+'\pi$')
        plt.title('$rh_{+}$ Extrapolated Waveform, N=4')
        plt.xlabel('$(t_{corr} - r^{*})/M$')
        plt.ylabel('$rh$')
        plt.legend()
        plt.show()

    for theta in [0,0.25,0.4,0.5]:
        plt.plot(ret_time, np.imag(get_at_point(waveform_modes, lm_list, theta*np.pi, 0.0*np.pi)), label='$theta='+str(theta)+'\pi$')
        plt.title('$rh_x$ Extrapolated Waveform, N=4')
        plt.xlabel('$(t_{corr} - r^{*})/M$')
        plt.ylabel('$rh$')
        plt.legend()
        plt.show()


    sxs_bbh_0305_init_mass_high = total_mass["SXS:BBH:0305"] * (metadata_high_res["SXS:BBH:0305"]["reference_mass1"] + metadata_high_res["SXS:BBH:0305"]["reference_mass2"])
    sxs_bbh_0305_final_mass_high = total_mass["SXS:BBH:0305"] * (metadata_high_res["SXS:BBH:0305"]["remnant_mass"])
    print("High resolution: Initial mass (solar masses): " + str(sxs_bbh_0305_init_mass_high))
    print("High resolution: Initial mass (solar masses): " + str(sxs_bbh_0305_final_mass_high))

    t305 = Goverc3s * total_mass["SXS:BBH:0305"] * gw_sxs_bbh_0305[:,0]
    h305 = 1.0e21 * Goverc2m * (total_mass["SXS:BBH:0305"] / distance["SXS:BBH:0305"]) * gw_sxs_bbh_0305[:,1]
    maxh305 = max(h305)

    from scipy.interpolate import interp1d

    sample_rate = 1e4 #Hz
    h305interp = interp1d(t305, h305)(np.arange(t305[0], t305[-1], 1.0 / sample_rate))

    from scipy.io import wavfile
    wavfile.write("SXS_BBH_0305.wav", int(sample_rate), h305interp / maxh305)

    Audio("SXS_BBH_0305.wav")
    """

if __name__ == "__main__":
    main()