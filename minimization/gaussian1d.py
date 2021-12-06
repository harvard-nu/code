import numpy as np
from scipy.stats import norm
from scipy.special import loggamma

def make_nll(x_bin_edges, counts):
    """
    make_nll(x_bin_edges, counts)

    Closure of a negative log-likelihood function for a normal (Gaussian)
    distribution under the assumption that the probability of measuring a
    single value is given by the Poisson probability mass function.

    Parameters
    ----------
    x_bin_edges : array_like of floats
        The bin edges along the first dimension.
    counts : array_like of floats
        Single-dimensional histogram.

    Returns
    -------
    nll : function

    """
    x_bin_centers = 0.5*(x_bin_edges[1:]+x_bin_edges[:-1])
    def nll(mu, sigma, a):
        _nll = 0.
        for idx in range(len(counts)):
            x = x_bin_centers[idx]
            z = counts[idx]
            if z == 0 or z < 1e-5:
                continue
            f = a * norm.pdf(x, mu, sigma)
            _nll += f - z*np.log(f) + loggamma(z+1)
        return _nll
    return nll

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator
    from iminuit import Minuit

    #//////////////////////////////////////////////////////////////////////////
    # configuration
    #//////////////////////////////////////////////////////////////////////////

    # seed for rng
    seed = 137

    # number of points
    n = 100000

    # mean and standard deviation of gaussian
    mu = 0
    sigma = 1

    # binning for 1d histogram
    x_bin_lower, x_bin_upper, x_bin_width = -6, 6, 0.1

    # plot aesthetics
    ax_font_size = 18

    # plot labels
    x_unit = 'cm'
    x_label = '$x$ [{}]'.format(x_unit)
    y_label = 'entries per {} {}'.format(x_bin_width, x_unit)

    #//////////////////////////////////////////////////////////////////////////
    # generate sample
    #//////////////////////////////////////////////////////////////////////////

    #--------------------------------------------------------------------------
    # requires numpy.__version__ >= 1.17.0 (new version)
    #--------------------------------------------------------------------------
    rng = np.random.default_rng(seed=seed)
    X = rng.normal(mu, sigma, n)
    #--------------------------------------------------------------------------
    # for numpy.__version__ < 1.17.0 (legacy version)
    #--------------------------------------------------------------------------
    # np.random.seed(seed=seed)
    # X = np.random.normal(mu, sigma, n)
    #--------------------------------------------------------------------------

    #//////////////////////////////////////////////////////////////////////////
    # get 2D histogram of sample
    #//////////////////////////////////////////////////////////////////////////

    x_bins = np.arange(x_bin_lower, x_bin_upper+x_bin_width, x_bin_width)
    counts, x_edges = np.histogram(X, bins=x_bins)

    #//////////////////////////////////////////////////////////////////////////
    # run minuit
    #//////////////////////////////////////////////////////////////////////////

    # construct minuit object
    minuit = Minuit(
        fcn=make_nll(x_edges, counts),
        mu=0,
        sigma=1,
        a=200,
        )

    # set step sizes for minuit's numerical gradient estimation
    minuit.errors = (1e-5, 1e-5, 1e-1)

    # set limits for each parameter
    minuit.limits = [ None, (0, None), (0, None) ]

    # set errordef for a negative log-likelihood (NLL) function
    minuit.errordef = Minuit.LIKELIHOOD
    # minuit.errordef = Minuit.LEAST_SQUARES  # for a least-squares cost function

    # run migrad minimizer
    minuit.migrad(ncall=1000000)

    # print estimated parameters
    print('minuit.values:')
    print(minuit.values)

    # run hesse algorithm to compute asymptotic errors
    minuit.hesse()

    # print estimated errors on estimated parameters
    print('minuit.errors:')
    print(minuit.errors)

    # run minos algorithm to compute confidence intervals
    minuit.minos()

    # print estimated parameters
    print('minuit.params:')
    print(minuit.params)

    # print estimated errors on estimated parameters
    print('minuit.merrors:')
    print(minuit.merrors)

    #--------------------------------------------------------------------------
    # get Gaussian function with estimated parameters from minimization
    #--------------------------------------------------------------------------
    x = 0.5*(x_edges[1:]+x_edges[:-1])
    y = minuit.values['a'] * norm.pdf(x, minuit.values['mu'], minuit.values['sigma'])

    #--------------------------------------------------------------------------
    # chi2
    #--------------------------------------------------------------------------
    mask = counts == 0
    y_fit = y[~mask]

    chi2 = ((y_fit - counts[~mask])**2 / y_fit).sum()
    dof = len(y_fit) - len(minuit.values)

    # reduced chi2
    print('chi2 / dof = {} / {} = {}'.format(chi2, dof, chi2/dof))

    #//////////////////////////////////////////////////////////////////////////
    # plot histogram of the sample and estimated Gaussian function
    #//////////////////////////////////////////////////////////////////////////

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)

    #--------------------------------------------------------------------------
    # plot the histogram
    #--------------------------------------------------------------------------
    # multiple ways of plotting the histogram
    ax.fill_between(x_edges[:-1], counts, step='post', color='C0')
    ax.step(x_edges[:-1], counts, where='post', color='C2', linewidth=3)
    ax.hist(X, bins=x_bins, color='k', histtype='step', hatch='//')

    #--------------------------------------------------------------------------
    # use the estimated parameters plot the Gaussian function
    #--------------------------------------------------------------------------
    ax.plot(x, y, c='C1')

    #--------------------------------------------------------------------------
    # set plot aesthetics
    #--------------------------------------------------------------------------

    # set labels
    ax.set_xlabel(x_label, horizontalalignment='right', x=1.0,
                  fontsize=ax_font_size)
    ax.set_ylabel(y_label, horizontalalignment='right', y=1.0,
                  fontsize=ax_font_size)

    # configure tick properties
    ax.tick_params(axis='both', which='major', labelsize=ax_font_size)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # set grid lines
    ax.grid(True, which='both', axis='both', color='k', linestyle=':',
            linewidth=1, alpha=0.2)

    #--------------------------------------------------------------------------
    # plot
    #--------------------------------------------------------------------------
    plt.tight_layout()
    plt.show()

