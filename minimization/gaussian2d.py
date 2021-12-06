import numpy as np
from scipy.special import loggamma

def gaussian2d(x, y, mu_x, mu_y, sigma_x, sigma_y, a, corr=0):
    """
    gaussian2d(x, y, mu_x, mu_y, sigma_x, sigma_y, a, corr=0)

    Bivariate normal (Gaussian) function

    Parameters
    ----------
    x : float or array_like of floats
        Input float or array for the first dimension (x-axis).
    y : float or array_like of floats
        Input float or array for the second dimension (y-axis).
    mu_x : float
        Mean ("centre") of the distribution along the x-axis.
    mu_y : float
        Mean ("centre") of the distribution along the y-axis.
    sigma_x : float
        Standard deviation (spread or "width") of the distribution along the
        x-axis. Must be non-negative.
    sigma_y : float
        Standard deviation (spread or "width") of the distribution along the
        y-axis. Must be non-negative.
    a : float
        Amplitude ("height") of the distribution along the z-axis. Must be
        non-negative.
    corr : float
        Correlation between x and y. Must be in [-1, 1].

    Returns
    -------
    out : ndarray or scalar
        Parameterized bivariate normal (Gaussian) function.

    Notes
    -----
    The probability density function for the bivariate Gaussian distribution is

    .. math:: p(x,y) = \frac{1}{2 \pi \sigma_x \sigma_y \sqrt{1-\rho^2}}
                       \exp
                         \left( -\frac{1}{2(1-\rho^2)}\left[
                           \left(\frac{x-\mu_x}{\sigma_x}\right)^2 -
                           2\rho\left(\frac{x-\mu_x}{\sigma_x}\right)
                           \left(\frac{y-\mu_y}{\sigma_y}\right) +
                           \left(\frac{y-\mu_y}{\sigma_y}\right)^2 
                           \right]
                         \right)

    where :math:`\mu_x` is the mean of :math:`x`, :math:`\sigma_x` is the
    standard deviation of :math:`x`, :math:`\mu_y` is the mean of :math:`y`,
    :math:`\sigma_y` is the standard deviation of :math:`y`, and :math:`\rho`
    is the correlation between :math:`x` and :math:`y`.

    References
    ----------
    .. [1] Wikipedia, "Multivariate normal distribution",
           https://en.wikipedia.org/wiki/Multivariate_normal_distribution

    """
    if corr > 1 or corr < -1:
        return np.inf
    a_ = a / (2*np.pi*sigma_x*sigma_y*np.sqrt(1-corr**2))
    x_ = (x-mu_x)/sigma_x
    y_ = (y-mu_y)/sigma_y
    arg = -0.5/(1-corr**2) * (x_**2-2*corr*x_*y_+y_**2)
    z = a_ * np.exp(arg)
    return z

def make_nll(x_bin_edges, y_bin_edges, counts):
    """
    make_nll(x_bin_edges, y_bin_edges, counts)

    Closure of a negative log-likelihood function for a bivariate normal
    (Gaussian) distribution under the assumption that the probability of
    measuring a single value is given by the Poisson probability mass
    function.

    Parameters
    ----------
    x_bin_edges : array_like of floats
        The bin edges along the first dimension.
    y_bin_edges : array_like of floats
        The bin edges along the second dimension.
    counts : array_like of floats
        Bi-dimensional histogram.

    Returns
    -------
    nll : function

    """
    x_bin_centers = 0.5*(x_bin_edges[1:]+x_bin_edges[:-1])
    y_bin_centers = 0.5*(y_bin_edges[1:]+y_bin_edges[:-1])
    def nll(mu_x, mu_y, sigma_x, sigma_y, a, corr):
        _nll = 0.
        rows, cols = counts.shape
        for row in range(rows):
            for col in range(cols):
                x = x_bin_centers[row]
                y = y_bin_centers[col]
                z = counts[row][col]
                if z == 0 or z < 1e-5:
                    continue
                f = gaussian2d(x, y, mu_x, mu_y, sigma_x, sigma_y, a, corr)
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

    # mean of bivariate Gaussian
    mu_x, mu_y = 0, 0
    mean = [ mu_x, mu_y ]

    # covariance of bivariate Gaussian
    sigma_x, sigma_y, corr = 1.5, 1.5, -0.5
    cov = [
        [ sigma_x**2, corr*sigma_x*sigma_y ],
        [ corr*sigma_y*sigma_x, sigma_y**2 ],
    ]

    # binning for 2d histogram
    x_bin_lower, x_bin_upper, x_bin_width = -6, 6, 0.1
    y_bin_lower, y_bin_upper, y_bin_width = -6, 6, 0.1

    # plot aesthetics
    ax_font_size = 18
    z_font_size = 14

    # plot labels
    x_unit = 'cm'
    y_unit = 'cm'
    x_label = '$x$ [{}]'.format(x_unit)
    y_label = '$y$ [{}]'.format(y_unit)
    z_label = 'entries per {} {} per {} {}'.format(x_bin_width, x_unit, y_bin_width, y_unit)

    #//////////////////////////////////////////////////////////////////////////
    # generate sample
    #//////////////////////////////////////////////////////////////////////////

    #--------------------------------------------------------------------------
    # requires numpy.__version__ >= 1.17.0 (new version)
    #--------------------------------------------------------------------------
    rng = np.random.default_rng(seed=seed)
    X = rng.multivariate_normal(mean, cov, n)
    #--------------------------------------------------------------------------
    # for numpy.__version__ < 1.17.0 (legacy version)
    #--------------------------------------------------------------------------
    # np.random.seed(seed=seed)
    # X = np.random.multivariate_normal(mean, cov, n)
    #--------------------------------------------------------------------------

    #//////////////////////////////////////////////////////////////////////////
    # get 2D histogram of sample
    #//////////////////////////////////////////////////////////////////////////

    # x and y points
    x_data, y_data = X[:, 0], X[:, 1]

    # bins for the 2D histogram
    x_bins = np.arange(x_bin_lower, x_bin_upper+x_bin_width, x_bin_width)
    y_bins = np.arange(y_bin_lower, y_bin_upper+y_bin_width, y_bin_width)

    # 2D histogram
    counts, x_edges, y_edges = np.histogram2d(x_data, y_data, bins=(x_bins, y_bins))

    #//////////////////////////////////////////////////////////////////////////
    # run minuit
    #//////////////////////////////////////////////////////////////////////////

    # construct minuit object
    minuit = Minuit(
        fcn=make_nll(x_edges, y_edges, counts),
        mu_x=0,
        mu_y=0,
        sigma_x=1,
        sigma_y=1,
        a=10000,
        corr=0,
        )

    # set step sizes for minuit's numerical gradient estimation
    minuit.errors = (1e-5, 1e-5, 1e-5, 1e-5, 1e-1, 1e-5)

    # set limits for each parameter
    minuit.limits = [None, None, (0, None), (0, None), (0, None), (-1, 1)]

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
    # chi2
    #--------------------------------------------------------------------------
    x_bin_centers = 0.5*(x_edges[1:]+x_edges[:-1])
    y_bin_centers = 0.5*(y_edges[1:]+y_edges[:-1])
    x_, y_ = np.meshgrid(x_bin_centers, y_bin_centers)
    z_ = gaussian2d(
        x_, y_,
        minuit.values['mu_x'],
        minuit.values['mu_y'],
        minuit.values['sigma_x'],
        minuit.values['sigma_y'],
        minuit.values['a'],
        minuit.values['corr']
        )

    mask = counts == 0
    z_fit = z_[~mask]

    chi2 = ((z_fit - counts[~mask])**2 / z_fit).sum()
    dof = len(z_fit) - len(minuit.values)

    # reduced chi2
    print('chi2 / dof = {} / {} = {}'.format(chi2, dof, chi2/dof))

    #//////////////////////////////////////////////////////////////////////////
    # plot histogram of the sample and estimated bivariate Gaussian function
    #//////////////////////////////////////////////////////////////////////////

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)

    #--------------------------------------------------------------------------
    # plot the 2D histogram
    #--------------------------------------------------------------------------
    x, y = np.meshgrid(x_bins, y_bins)
    p = ax.pcolormesh(x, y, np.ma.masked_where(counts == 0, counts).T,
                      shading='auto')  # with mask
    # p = ax.pcolormesh(x, y, counts.T, shading='auto')  # without mask

    #--------------------------------------------------------------------------
    # create color bar for the 2D histogram
    #--------------------------------------------------------------------------
    cb = fig.colorbar(p)
    cb.ax.tick_params(labelsize=z_font_size)
    cb.ax.yaxis.offsetText.set_fontsize(z_font_size)
    cb.ax.yaxis.set_offset_position('left')
    cb.set_label(z_label, rotation=90, fontsize=z_font_size)  # z label

    #--------------------------------------------------------------------------
    # use the estimated parameters plot the bivariate Gaussian function
    #--------------------------------------------------------------------------
    z = gaussian2d(
        x, y,
        minuit.values['mu_x'],
        minuit.values['mu_y'],
        minuit.values['sigma_x'],
        minuit.values['sigma_y'],
        minuit.values['a'],
        minuit.values['corr']
        )

    # plot contour of the bivariate Gaussian function
    c = ax.contour(x, y, z, levels=7, colors='w', linewidths=1.5, alpha=1)
    ax.clabel(c, inline=True, fontsize=6)

    #--------------------------------------------------------------------------
    # set plot aesthetics
    #--------------------------------------------------------------------------
    # set aspect ratio to 1:1
    ax.set_aspect('equal')

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

