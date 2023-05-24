import numpy as np
import pandas as pd
import os
import json

def init_QNM_mode_frequency_coeffs_analytic_database():
    assert os.path.isfile('QNM_mode_frequency_coefficients.jsonc'), \
        "QNM_mode_frequency_coefficients.jsonc does not exist" \
        " in {0:s}.\n".format(os.path.abspath(__file__))
    with open('QNM_mode_frequency_coefficients.jsonc') as json_file:
        QNM_mode_frequency_coefficients = json.load(json_file)
    return QNM_mode_frequency_coefficients

def load_QNM_mode_LUT(n=0,l=2,m=2):
    assert abs(m) <= l, "m ({0:d}) has to be in the range [-{1:d},{1:d}].\n".format(m,l)
    assert os.path.isdir('QNMSpecLUT'), "QNMSpecLUT folder does not exist in {0:s}.\n".format(os.path.abspath(__file__))

    fn = "n{0:d}l{1:d}m{2:d}.dat".format(n+1, l, m).replace("-","m")
    assert os.path.isfile('QNMSpecLUT/{0:s}'.format(fn)), \
        "File {1:s} does not exist in {1:s}.\n".format(fn, os.path.abspath(__file__)+"/QNMSpecLUT")
    entries = ["a/M", "MwR", "MwI", "Re(Alm)", "Im(Alm)"]
    LUT = pd.read_csv('QNMSpecLUT/{0:s}'.format(fn), delim_whitespace=True, names=entries)
    return LUT

def freq_for_mode(db, l=2, m=2, n=0, mass=1, spin=0, method="analytic"):
    """
    This function extracts frequencies for a given mode using various methods
    1. Analytic interpolation presented in arXiv:gr-qc/0512160, eq. E1-E2.
    2. Using look-up tables given in http://www.phy.olemiss.edu/ berti/qnms.html to the 0.1% level and interpolating in between.
    3. Using qnm Python package presented in arXiv:1908.10377
    """
    assert method in ["analytic", "LUT", "qnmpy"], "Method {0:s} is invalid.\n".format(method) # nana todo - add support for Leo Stein's code
    assert np.all(spin >= 0) and np.all(spin <= 1), "Black hole spin is not physical. It is {0:f}.\n".format(spin)

    if method == "analytic":
        f1, f2, f3, perc_re, q1, q2, q3, perc_im = db[f"{l}"][f"{m}"][f"{n}"]
        freq_re = (f1 + f2*np.power(1-spin, f3))/mass
        Q = q1 + q2*np.power(1-spin, q3)
        freq_im = -freq_re/(2*Q)
    elif method == "LUT":
        LUT = load_QNM_mode_LUT(n, l, m)
        freq_re = np.interp(spin, LUT['a/M'].values, LUT['MwR'].values)/mass
        freq_im = np.interp(spin, LUT['a/M'].values, LUT['MwI'].values)/mass
    else:
        assert False, "todo - add support for Leo Stein's code.\n"
    return freq_re + 1j*freq_im

def props_from_freq(db, freq, l=2, m=2, n=0, method="LUT"):
    """
    This function extracts frequencies for a given mode using various methods
    1. Analytic interpolation presented in arXiv:gr-qc/0512160, eq. E1-E2.
    2. Using look-up tables given in http://www.phy.olemiss.edu/ berti/qnms.html to the 0.1% level and interpolating.
    3. Using qnm Python package presented in arXiv:1908.10377
    """
    assert method in ["analytic", "LUT", "qnmpy"], "Method {0:s} is invalid.\n".format(method) # nana todo - add support for Leo Stein's code

    if method == "analytic":
        f1, f2, f3, perc_re, q1, q2, q3, perc_im = db[f"{l}"][f"{m}"][f"{n}"]
        Q_factor = np.real(freq)/(2*np.abs(np.imag(freq)))
        spin = 1 - np.power((Q_factor-q1)/q2, 1/q3)
        mass = (f1 + f2*np.power(1-spin, f3))/np.real(freq)
    elif method == "LUT":
        """
        1. Calculate the quality factor, i.e. Q=wR/2|wI|.
        2. Assert it is monotonously rising and no entry appears twice. # nana - todo, add assert
        3. Find the nearest rows in the LUT and calculate a/M.
        4. Calculate MwR similarly to 3 and extract M.
        """
        Q = np.real(freq)/(2*np.abs(np.imag(freq)))
        LUT = load_QNM_mode_LUT(n, l, m)
        LUT["Q"] = LUT["MwR"]/(2*np.abs(LUT["MwI"]))
        assert len(LUT["Q"]) == len(LUT["Q"].unique()), "Quality factor in database has repeating values."
        assert np.all((LUT["Q"].diff()>0)[1:]), "Quality factor in database is not monotonous."

        spin = Q*1
        spin[:] = np.interp(Q, LUT["Q"].values, LUT['a/M'].values)
        mass = Q*1
        mass[:] = np.interp(Q, LUT["Q"].values, LUT["MwR"].values)/np.real(freq)
    else:
        assert False, "todo - add support for Leo Stein's code.\n"
    #assert np.all(np.abs(freq-freq_for_mode(db, l=l, m=m, n=n, mass=mass, spin=spin)) < 1e-10)
    return mass, spin

def freq_from_num_data(num_data, bDebug=False):
    amplitude = num_data.abs
    freq_im = -amplitude.derivative()/amplitude
    phase = num_data.arg_unwrapped
    freq_re = -phase.derivative()

    if(bDebug):
        h_plus = num_data.real
        h_cross = num_data.imag
        freq_im = (h_plus.derivative(2)*h_cross - h_cross.derivative(2)*h_plus)/(2*(h_cross.derivative()*h_plus - h_plus.derivative()*h_cross))
        osc_plus = h_plus/np.exp(-freq_im*freq_im.t)
        freq_re = np.sqrt(-osc_plus.derivative(2)/osc_plus)
    return freq_re + 1j*freq_im

def fit_mode_amp_to_num_data(overtone_num, num_data, db, t0=0, l=2, m=2, mass=1, spin=0):
    freqs = np.empty((overtone_num))
    freqs[:] = np.nan
    A_mat = np.zeros((num_data.t.shape[0], overtone_num))
    b_vec = np.array(num_data.data, ndmin=2).T
    for overtoneI in range(overtone_num):
        freqs[overtoneI] = freq_for_mode(db, l=l, m=m, n=overtoneI, mass=mass, spin=spin)
        A_mat[:, overtoneI] = np.exp(-1j*freqs[overtoneI]*(num_data.t-num_data.t[0]+t0))
    assert not np.any(np.isnan(freqs))
    amps = np.linalg.inv(A_mat.T @ A_mat) @ A_mat.T @ b_vec
    return amps

def clean_noise_freq(freq, fc, method="butterlowpass"):
    # Clean numerical noise in the tail of the frequency (both real and imaginary parts

    assert method in ["steplowpass", "butterlowpass", "running average"]

    freq_re = np.real(freq)
    freq_im = np.imag(freq)

    fs = np.average(1/np.diff(freq.t))

    # todo - make adaptive filter cutoff
    if method=="butterlowpass":
        b, a = scipy.signal.butter(8, fc, fs=fs, btype='low', analog=False)
        freq_re_filt = scipy.signal.lfilter(b, a, freq_re)
        freq_im_filt = scipy.signal.lfilter(b, a, freq_im)

    freq_filtered = freq_re_filt + 1j*freq_im_filt

    return freq_filtered
