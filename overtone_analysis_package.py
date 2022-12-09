import numpy as np

def waveform_dotprod(x, y, dt):
    assert len(x)==len(y)==len(dT)
    dT = x.t[ind_t0+1:ind_T+1]-x.t[ind_t0:ind_T]
    return sum(x.data[ind_t0:ind_T]*np.conj(y.data[ind_t0:ind_T])*dT)

def calc_err(h_NR_init, amps, freqs, t0=0, T=90):
    assert len(amps) == len(freqs)
    ind_t0 = np.argmax(h_NR_init.t>=t0)
    ind_T = np.argmax(h_NR_init.t>T)
    dt = h_NR_init.t[ind_t0+1:ind_T+1]-h_NR_init.t[ind_t0:ind_T]
    h_NR = h_NR_init[ind_t0:ind_T]
    h_fit = h_NR
    h_fit.data = np.sum(amps*np.exp(-1j*freqs*(h_NR.t-t0)), axis=0)
    norm_h_NR = waveform_dotprod(h_NR, h_NR, dt)
    assert len(norm_h_NR) == 1
    norm_h_fit = waveform_dotprod(h_fit, h_fit, dt)
    assert len(norm_h_fit) == 1
    err = 1 - waveform_dotprod(h_NR, h_fit, dt)/np.sqrt(norm_h_fit*norm_h_NR)
    return err