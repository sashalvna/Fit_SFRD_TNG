import numpy as np
from scipy.integrate import trapezoid
import h5py
#from bilby.core.result import read_in_result
from popsummary.popresult import PopulationResult
from scipy.integrate import cumulative_trapezoid, trapezoid

def setup(): 

    from matplotlib import rc, rcParams

    rc_params = {
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "font.size": 12,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "text.usetex": True,
        "savefig.dpi": 300,
    }
    rcParams.update(rc_params)
    rc("text", usetex=True)
    rc("axes", linewidth=0.5)
    rcParams["xtick.major.pad"] = "6"
    rcParams["ytick.major.pad"] = "6"



def get_params(dataset, param, rate = True):
    param = [param] if not isinstance(param, list) else param
    for p in param:
        dat = dataset.get_rates_on_grids(p)
        x = dat[0][0]
        pdf = dat[1]
    if rate:
        lamb = dataset.get_hyperparameter_samples(hyperparameters = 'lamb')
        pdf = pdf * (1 + 0.2)**lamb.reshape(-1,1)
    return x, pdf



def plot_90CI(ax, x, pdf, fill = True, label = '', color = 'r', fill_alpha = 0.3, lw = 2, ls = '-', secondary_ls = '--',
        plot_kwargs=dict(rasterized=True), fill_kwargs=dict(rasterized=True), median=True, mean=False):
    med = np.median(pdf, axis=0) if median else np.mean(pdf, axis=0)
    low = np.percentile(pdf, 5, axis=0)
    high = np.percentile(pdf, 95, axis=0)

    if median:
        ax.plot(x, med, **plot_kwargs, color = color, label = label, lw = lw, ls = ls)
        label = None
    if fill:
        ax.fill_between(x, low, high, color = color, alpha = fill_alpha, **fill_kwargs, label = label)
        label = None
    else:
        ax.plot(x, low, **plot_kwargs, color = color, lw = lw, ls = secondary_ls, label = label)
        ax.plot(x, high, **plot_kwargs, color = color, lw = lw, ls = secondary_ls)


def get_03b_plp_ppds(path, mass_1 = True, mass_ratio = False):
    o3b_result = read_in_result(path + '/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')
    o3b_hyper = o3b_result.posterior.copy()
    rate_o2 = (1 + 0.2)**np.mean(o3b_hyper['lamb'].values)
    with h5py.File(path + '/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5','r') as f:
        ppd = f['ppd'][:] * rate_o2
        mlines = f['lines']['mass_1'][:] * rate_o2
        qlines = f['lines']['mass_ratio'][:] * rate_o2
    m1 = np.linspace(2,100,1000)
    q = np.linspace(0.1,1,500)
    mu = trapezoid(ppd, q, axis = 0) if mass_1 else trapezoid(ppd, m1, axis = 1)
    lines = mlines if mass_1 else qlines
    low = np.percentile(lines, 5, axis = 0)
    hi = np.percentile(lines, 95, axis = 0)
    x = m1 if mass_1 else q
    return x, low, mu, hi



def setup_mass_plot(ax, xrange=(2,100), yrange=(1e-3,20), yscale = 'log', xscale = 'linear', label_kwargs = {}, grid_kwargs={}):
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.set_ylim(*yrange)
    ax.set_xlim(*xrange)
    ax.set_xlabel(r"$m_1 \left[ \mathrm{M}_\odot \right]$", **label_kwargs)
    ax.set_ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1 \left[ \mathrm{Gpc}^{-3} \, \mathrm{yr}^{-1} \, \mathrm{M}_\odot^{-1} \right]$', **label_kwargs)
    ax.grid(**grid_kwargs)


def setup_mass_ratio_plot(ax, xrange=(0,1), yrange=(4e-2,3e2), yscale = 'log', xscale = 'linear', label_kwargs = {}, grid_kwargs={}):
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.set_ylim(*yrange)
    ax.set_xlim(*xrange)
    ax.set_xlabel(r"$q$", **label_kwargs)
    ax.set_ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}q \left[ \mathrm{Gpc}^{-3} \, \mathrm{yr}^{-1} \right]$', **label_kwargs)
    ax.grid(**grid_kwargs)


def darken(hex_color, factor = 0.8):
    """Darkens a given hex color by a specified factor.

    Args:
        hex_color: The hex color string (e.g., "#FF0000").
        factor: A float between 0 and 1, where 0 is fully black and 1 is no change.

    Returns:
        A darkened hex color string.
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)

    return f"#{r:02x}{g:02x}{b:02x}"

class GetBSplineMacroData(PopulationResult):

    def __init__(self, fname=None, iid = False, isopeak = False, hyperparameters=None, hyperparameter_descriptions=None, hyperparameter_latex_labels=None, references=None, model_names=None, events=None, event_waveforms=None, event_sample_IDs=None, event_parameters=None):
        super().__init__(fname, hyperparameters, hyperparameter_descriptions, hyperparameter_latex_labels, references, model_names, events, event_waveforms, event_sample_IDs, event_parameters)
        self.isopeak = isopeak
        if iid:
            self.rate_keys = ['rate_vs_mass_1_at_z0-2', 'rate_vs_mass_ratio_at_z0-2', 'p(a)', 'p(cos_tilt)', 'rate_vs_redshift']
            self.params = ['mass_1', 'mass_ratio', 'a', 'cos_tilt', 'redshift',]
        else:
            self.rate_keys = ['rate_vs_mass_1_at_z0-2', 'rate_vs_mass_ratio_at_z0-2', 'p(a_1)', 'p(a_2)', 'p(cos_tilt_1)', 'p(cos_tilt_2)', 'rate_vs_redshift']
            self.params = ['mass_1', 'mass_ratio', 'a_1', 'a_2', 'cos_tilt_1', 'cos_tilt_2', 'redshift']
        if self.isopeak:
            keys = self.rate_keys.copy()
            keys.remove('rate_vs_redshift')
            self.subpop_keys = {sub: [sub + '_' + key for key in keys] for sub in ['peak', 'continuum']}
        
        self.pdfs = self.get_dict_from_popsum()
        self.macros = self.generate_macros()
        

    def get_ppd(self, pxs, rate = None):
        if rate is not None:
            ppd = np.mean(pxs / rate, axis = 0)
        else:
            ppd = np.mean(pxs, axis = 0)
        return ppd

    def get_dict_from_popsum(self):

        pdf_dict = {}

        if self.isopeak:
            for sub in self.subpop_keys.keys():
                keys = self.subpop_keys[sub]
                for idx, key in enumerate(keys):
                    x, px = self.get_rates_on_grids(grid_key = key)
                    x = x[0]
                    
                    pdf_dict[sub + '_' + self.params[idx] + '_pdfs'] = px
                    pdf_dict[sub + '_' + self.params[idx]] = x

                    if sub == 'peak':
                        z, pz = self.get_rates_on_grids(grid_key = 'rate_vs_redshift')
                        z = z[0]
                        pdf_dict['redshift_pdfs'] = pz
                        pdf_dict['redshift'] = z

        else:
            for idx, key in enumerate(self.rate_keys):
                x, px = self.get_rates_on_grids(grid_key = key)
                x = x[0]

                pdf_dict[self.params[idx] + '_pdfs'] = px
                pdf_dict[self.params[idx]] = x

        return pdf_dict

    def get_max_loc(self, param, xrange = None):
    
        y = self.pdfs[param + '_pdfs']
        x = self.pdfs[param]
    
        if xrange is not None:
            y = y[:,xrange[0]:xrange[1]]
            x = x[xrange[0]:xrange[1]]
        
        return x[np.argmax(y, axis = 1)]

    def get_zero_slope_loc(self, param):
        y = self.pdfs[param + '_pdfs']
        x = self.pdfs[param]
        diff = y - np.roll(y,1, axis = 1) > 0
        idxs = [np.max(np.where(~diff[i] & np.roll(diff[i], 1))[0]) for i in range(diff.shape[0])]
        return x[idxs]

    def get_cred_vals(self, x, axis = 0):
        med = np.median(x, axis = axis)
        low = np.percentile(x, 5, axis = axis)
        hi = np.percentile(x, 95, axis = axis)
        return low, med, hi

    def get_ppd_percentile(self, param, perc, rate = None):
        pdfs = self.pdfs[param + '_pdfs']
        xs = self.pdfs[param]
        ppd = self.get_ppd(pdfs, rate=rate)
        i = ppd.shape[0]
        cumulative_prob = cumulative_trapezoid(ppd, initial = 0)
        init_prob = cumulative_prob[-1]
        prob = init_prob
        final_prob = init_prob * perc / 100.0
        while prob > final_prob:
            i -= 1
            prob = cumulative_prob[i]
        return xs[i]
    
    def get_ppd_cred_values(self, param):
        low = self.get_ppd_percentile(param, 5)
        med = self.get_ppd_percentile(param, 50)
        hi = self.get_ppd_percentile(param, 95)
        return low, med, hi
    
    def record_cred_vals(self, x, decimals = 2):
        
        return {
                'median': str(np.round(x[1], decimals = decimals)),
                'error plus': str(np.round(x[2] - x[1], decimals = decimals).astype(str)),
                'error minus': str(np.round(x[1] - x[0], decimals = decimals).astype(str)),
                '5th percentile': str(np.round(x[0], decimals = decimals).astype(str)),
                '95th percentile': str(np.round(x[2], decimals = decimals).astype(str))
            }
    def generate_macros(self):

        macros = {}

        if self.isopeak:

            peak_mu = self.get_hyperparameter_samples(hyperparameters='peak_mu').T[0]
            peak_mu_cred = self.get_cred_vals(peak_mu)
            peak_logsigp = self.get_hyperparameter_samples(hyperparameters='peak_logsig').T[0]
            peak_logsigp_cred = self.get_cred_vals(peak_logsigp)
            macros['mass_1'] = {'peak': {}}
            macros['mass_1']['peak']['peak_mu'] = self.record_cred_vals(peak_mu_cred)
            macros['mass_1']['peak']['peak_logsigp'] = self.record_cred_vals(peak_logsigp_cred)
            
            macros['mass_ratio'] = {'peak': {}, 'continuum': {}}
            macros['a'] = {'peak': {}, 'continuum': {}}
            macros['cos_tilt'] = {'peak': {}, 'continuum': {}}

            for sub in self.subpop_keys.keys():

                q_peak = self.get_max_loc(sub + '_mass_ratio')
                q_peak_cred = self.get_cred_vals(q_peak)
                macros['mass_ratio'][sub]['peak_location'] = self.record_cred_vals(q_peak_cred)

                for param in ['a', 'cos_tilt']:

                    peak = self.get_max_loc(sub + '_' + param)
                    peak_cred = self.get_cred_vals(peak)
                    macros[param][sub]['peak_location'] = self.record_cred_vals(peak_cred)
        else:
            for param in self.params:
                macros[param] = {}

                if param == 'mass_1':

                    peak_1 = self.get_max_loc('mass_1')
                    peak_1_cred = self.get_cred_vals(peak_1)

                    peak_2 = self.get_max_loc('mass_1', xrange = [68, -1])
                    peak_2_cred = self.get_cred_vals(peak_2)

                    macros['mass_1']['peak_1_location'] = self.record_cred_vals(peak_1_cred)
                    macros['mass_1']['peak_2_location'] = self.record_cred_vals(peak_2_cred)
                
                elif param == 'redshift':

                    idx = np.sum(self.pdfs['redshift'] < 0.2)
                    z02 = self.pdfs['redshift_pdfs'][:,idx]
                    z02_cred = self.get_cred_vals(z02)
                    macros['rate_at_z_0-2'] = self.record_cred_vals(z02_cred)
        

                else:
                    peak = self.get_max_loc(param)
                    peak_cred = self.get_cred_vals(peak)
                    macros[param]['peak'] = self.record_cred_vals(peak_cred)


                ppd_cred = self.get_ppd_cred_values(param)
                macros[param]['ppd'] = self.record_cred_vals(ppd_cred)

        return macros
    
