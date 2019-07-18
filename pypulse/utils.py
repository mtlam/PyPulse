'''
Michael Lam 2015

Useful mathematical commands
'''
import sys
import numpy as np
import scipy.fftpack as fft
import scipy.optimize as optimize
import scipy.stats as stats
import scipy.interpolate as interp
import scipy.special as special
from scipy.signal import fftconvolve, correlate
import matplotlib.pyplot as plt

from multiprocessing import Process, Pipe
try:
    from itertools import izip
except ImportError:
    izip = zip

if sys.version_info.major == 2:
    fmap = map
elif sys.version_info.major == 3:
    fmap = lambda x, *args: list(map(x, *args))
    xrange = range

'''
ACF
var=True: calculate variance, var=False, do not calculate. var=number: use as number
Include mean subtraction?
Include lagaxis function?
'''
def acf(array, var=False, norm_by_tau=True, lagaxis=False): #set lagaxis=True?
    array = np.array(array)
    N = len(array)
    if var:
        var = np.var(array)
    elif not var:
        var = 1

    lags = np.arange(-(N-1), N, dtype=np.float)
    if norm_by_tau:
        taus = np.concatenate((np.arange(1, N+1), np.arange(N-1, 0, -1)))
        if lagaxis:
            return lags, np.correlate(array, array, "full")/(var*taus)
        return np.correlate(array, array, "full")/(var*taus)
    if lagaxis:
        return lags, np.correlate(array, array, "full")/(var*N)
    return np.correlate(array, array, "full")/(var*N)

#Do not provide bins but provide edges?
#error bars?
def lagfunction(func, t, x, e=None, dtau=1, tau_edges=None, mirror=False):
    length = len(x)
    if tau_edges is None:
        num_lags = np.ceil((np.max(t) - np.min(t))/dtau) + 1 #+1?
        taus = np.arange(num_lags) * dtau
        tau_edges = (taus[:-1] + taus[1:])/2.0
        tau_edges = np.hstack((tau_edges, [tau_edges[-1]+dtau]))
        N_taus = np.zeros(num_lags)
        retval = np.zeros(num_lags)
        variance = np.zeros(num_lags)
    else:
        dtau = np.median(np.diff(tau_edges)) #not quite working
        taus = tau_edges - dtau
        #taus = np.concatenate((taus, [taus[-1]+dtau]))
        N_taus = np.zeros(len(tau_edges))#-1)
        retval = np.zeros(len(tau_edges))#-1) #this should just be "mean"
        variance = np.zeros(len(tau_edges))

    weighted = False
    if e is not None:
        weighted = True

    # this could be sped up several ways
    I = list(range(length))
    for i in I:
        for j in I:
            dt = np.abs(t[i]-t[j])
            index = np.where(dt < tau_edges)[0] #<=?
            if len(index) == 0:
                continue

            index = index[0] #get the lowest applicable lag value

            N_taus[index] += 1
            #Replace this with online algorithm?
            retval[index] += func(x[i], x[j])
    #    print N_taus

    #divide by zero problem!, only with one-pass algorithm
    retval = retval / N_taus

    if mirror: #fix this
    #mirror each:
        taus = np.concatenate((-1*taus[::-1][:-1], taus))
        retval = np.concatenate((retval[::-1][:-1], retval))
        #retval /= 2 #double counting, can speed this up! why no division by 2?
    #return tau_edges, retval
    return taus, retval
    #return tau_edges, retval #BAD

def acf2d(array, speed='fast', mode='full', xlags=None, ylags=None):
    if speed == 'fast' or speed == 'slow':
        ones = np.ones(np.shape(array))
        norm = fftconvolve(ones, ones, mode=mode) #very close for either speed
        if speed == 'fast':
            return fftconvolve(array, np.flipud(np.fliplr(array)), mode=mode)/norm
        else:
            return correlate(array, array, mode=mode)/norm
    elif speed == 'exact':
        #NOTE: (r, c) convention is flipped from (x, y), also that increasing c is decreasing y
        LENX = len(array[0])
        LENY = len(array)
        if xlags is None:
            xlags = np.arange(-1*LENX+1, LENX)
        if ylags is None:
            ylags = np.arange(-1*LENY+1, LENY)
        retval = np.zeros((len(ylags), len(xlags)))
        for i, xlag in enumerate(xlags):
            print(xlag)
            for j, ylag in enumerate(ylags):
                if ylag > 0 and xlag > 0:
                    A = array[:-1*ylag, xlag:] #the "stationary" array
                    B = array[ylag:, :-1*xlag]
                elif ylag < 0 and xlag > 0:
                    A = array[-1*ylag:, xlag:]
                    B = array[:ylag, :-1*xlag]
                elif ylag > 0 and xlag < 0:#optimize later via symmetries
                    A = array[:-1*ylag, :xlag]
                    B = array[ylag:, -1*xlag:]
                elif ylag < 0 and xlag < 0:
                    A = array[-1*ylag:, :xlag]
                    B = array[:ylag, -1*xlag:]
                else: #one of the lags is zero
                    if ylag == 0 and xlag > 0:
                        A = array[-1*ylag:, xlag:]
                        B = array[:, :-1*xlag]
                    elif ylag == 0 and xlag < 0:
                        A = array[-1*ylag:, :xlag]
                        B = array[:, -1*xlag:]
                    elif ylag > 0 and xlag == 0:
                        A = array[:-1*ylag, :]
                        B = array[ylag:, -1*xlag:]
                    elif ylag < 0 and xlag == 0:
                        A = array[-1*ylag:, :]
                        B = array[:ylag, -1*xlag:]
                    else:
                        A = array[:, :]
                        B = array[:, :]
                        #print xlag, ylag, A, B
                C = A*B
                C = C.flatten()
                goodinds = np.where(np.isfinite(C))[0] #check for good values
                retval[j, i] = np.mean(C[goodinds])
        return retval

def lagaxis(arg, dtau=1):
    if isinstance(arg, (list, np.ndarray)): #generate a lag axis based on a time axis
        length = len(arg)
        dtau = np.mean(np.diff(arg))
        return np.arange(-1*length+1, length)*dtau
    else: # Returns a generic lag axis
        half = arg//2 #arg should be odd
        return np.arange(-1*half, half+1)*dtau

#Taken from diagnostics.py, set default threshold=3
def zct(series, threshold=3, full=False, meansub=False):
    count = 0
    N = len(series)
    current = np.sign(series[0])
    if meansub:
        series -= np.mean(series)
    for i in range(1, N):
        #print np.sign(series[i])
        if np.sign(series[i]) != current:
            count += 1 #Crossed zero, add to count
            current *= -1 #Flip sign
    average_zw = float(N-1)/2
    sigma_zw = np.sqrt(N-1)/2
    if (average_zw - threshold*sigma_zw) <= count <= (average_zw + threshold*sigma_zw):
        if full:
            return True, abs(count-average_zw)/sigma_zw, count
        return True
    else:
        if full:
            return False, abs(count-average_zw)/sigma_zw, count
        return False

'''
Decimate the data
Be careful with window_len!
if remainder: include the remainder?
'''
def decimate(x, window_len, error=False):#, mean=True, remainder=False):
    if window_len == 1:
        return x
    length = len(x)
    retval = np.zeros(length/window_len)
    #counts = np.zeros_like(retval)
    if error:
        errorretval = np.zeros_like(retval)
        for i in range(len(retval)):
            win = x[i*window_len:(i+1)*window_len]
            retval[i] = np.mean(win)
            errorretval[i] = np.std(win)/np.sqrt(window_len)
        return retval, errorretval
    else:
        for i in range(window_len):
            retval += x[i:length:window_len]
        return retval/window_len

def imshow(x, ax=None, origin='lower', interpolation='nearest', aspect='auto', **kwargs):
    if ax is not None:
        im = ax.imshow(x, origin=origin, interpolation=interpolation, aspect=aspect, **kwargs)
    else:
        im = plt.imshow(x, origin=origin, interpolation=interpolation, aspect=aspect, **kwargs)
    return im

'''
Histogram

Allow for intervals or number of bins
bins: Provide an array of bins
'''
def histogram(values, interval=1.0, bottom=None, full=False, bins=None,
              plot=False, show=True, horizontal=False, normalize=False,
              logbins=False, **kwargs):
    if bins is None:
        factor = 1.0/interval
        if bottom is None:
            minval = (np.fix(factor*min(values))-1)/factor
        else:
            minval = bottom
        maxval = (np.ceil(factor*max(values))+1)/factor
        bins = np.arange(minval, maxval+interval, interval)
    else:
        minval = bins[0]
        maxval = bins[-1]
    hist, bins = np.histogram(values, bins=bins)
    if logbins: #need to adjust the centers
        center = np.sqrt(bins[:-1]*bins[1:]) #geometric mean = log-average
    else:
        center = (bins[:-1]+bins[1:])/2.0 #arithmetic mean

    if normalize:
        hist = np.array(hist, dtype=np.float)/(float(interval)*np.sum(hist))


    if plot:
        plothistogram(center, hist, interval, show=show, horizontal=horizontal,
                      logbins=logbins, **kwargs)
        return

    if full:
        return center, hist, bins, minval, maxval
    return center, hist

#Need to take into account logbins!
#Must be left aligned!
def plothistogram(center, hist, interval=1.0, bins=None, steps=False,
                  show=True, horizontal=False, logbins=False,
                  centerbin=False, ax=None, **kwargs):
    if steps or bins is not None:
        binsize = np.mean(np.diff(center))
        center = np.concatenate(([center[0]-binsize], center, [center[-1]+binsize]))
        if centerbin:
            center -= binsize/2.0
        hist = np.concatenate(([0], hist, [0]))
        if ax is None:
            p, = plt.plot(center, hist, drawstyle='steps-mid', **kwargs)
        else:
            p, = ax.plot(center, hist, drawstyle='steps-mid', **kwargs)
    else:
        if centerbin:
            binsize = np.mean(np.diff(center))
            center -= binsize/2.0

        if horizontal:
            if ax is None:
                p = plt.barh(center, hist, height=interval, align='center', **kwargs)
            else:
                p = ax.barh(center, hist, height=interval, align='center', **kwargs)
        else:
            if ax is None:
                p = plt.bar(center, hist, width=interval, align='center', **kwargs)
            else:
                p = ax.bar(center, hist, width=interval, align='center', **kwargs)
    if show:
        plt.show()
    return p

#Creates empirical cdf
def ecdf(values, sort=True):
    if sort:
        values = np.sort(values)
    return values, np.linspace(0, 1, len(values))

EPS = special.erf(1.0/np.sqrt(2))/2.0

def pdf_to_cdf(pdf, dt=1):
    return np.cumsum(pdf)*dt

def likelihood_evaluator(x, y, cdf=False, median=False, pm=True, values=None):
    """
    cdf: if True, x,y describe the cdf
    median: if True, use the median value, otherwise the peak of the pdf (assuming cdf=False
    pm: xminus and xplus are the plus/minus range, not the actual values

    Future: give it values to grab off the CDF (e.g. 2 sigma, 99%, etc)
    values: use this array
    """
    if not cdf:
        y = y/np.trapz(y, x=x)
        ycdf = pdf_to_cdf(y, dt=(x[1]-x[0]))
    else: #else given a cdf
        ycdf = y

    if not values:
        if median:
            yb = 0.50   #Now take the median!
        else:
            indb = np.argmax(y)
            yb = ycdf[indb]
        ya = yb - EPS
        yc = yb + EPS
        yd = 0.95

        inda = np.argmin(np.abs(ycdf - ya))
        if median:
            indb = np.argmin(np.abs(ycdf - yb))
        indc = np.argmin(np.abs(ycdf - yc))
        indd = np.argmin(np.abs(ycdf - yd))

        inds = np.arange(inda, indc+1) #including indc
        #print indc-inda, np.trapz(L[inds], x=Vrs[inds])
        xval = x[indb]
        if pm:
            xminus = x[indb] - x[inda]
            xplus = x[indc] - x[indb]
        else:
            xminus = x[inda]
            xplus = x[indc]
        x95 = x[indd]

        return xval, xminus, xplus, x95
    else:
        retval = np.zeros_like(values)
        for i, v in enumerate(values):
            indv = np.argmin(np.abs(ycdf - v))
            retval[i] = x[indv]
        return retval

'''
2D data saving
'''
def write2Dtxt(filename, array, x=None, y=None, info=True, **kwargs):
    if x is None and y is None:
        np.savetxt(filename, array, **kwargs)
    # what about one of them being None
    else:
        header = " ".join(fmap(str, x)) + "\n" + " ".join(fmap(str, y)) + "\n"
        # check if header is in kwargs
        np.savetxt(filename, array, comments='', header=header, **kwargs)

'''
Normalize an array to unit height
Below: normalize 
'''
def normalize(array, simple=False, minimum=None):
    if simple:
        return array/np.max(array)
    maximum = np.max(array)
    if minimum is None:
        minimum = np.min(array)
    return (array-minimum)/(maximum-minimum)
def normalize_area(array, x=None, full=False):
    if x is None:
        x = np.arange(len(array))
    area = np.trapz(array, x=x)
    if full:
        return array/area, area
    return array/area

'''
Center the maximum value of the array
Follows profiles.py
'''
def center_max(array, full=False):
    maxind = np.argmax(array)
    length = len(array)
    centerind = int(length/2)
    diff = centerind-maxind
    if full:
        return diff, np.roll(array, diff)
    return np.roll(array, diff)

#Follow profiles.py
#notcentered is very rudimentary
#have norm be simple
def FWHM(series, norm=True, simple=False, notcentered=False):
    return FW(series, value=0.5, norm=norm, simple=simple, notcentered=notcentered)

def FW(series, value=0.5, norm=True, simple=False, notcentered=False):
    if norm:
        series = normalize(series) #assumes these are floats, not integers!
    y = np.abs(series-value)

    N = len(series)
    half = N//2

    wL = 0
    wR = N-1

    #initial solution
    if notcentered:
        series = center_max(series)
#        half=np.argmax(series)
    iL = np.argmin(y[:half])
    iR = np.argmin(y[half:])+half
    if not simple:
        x = np.arange(len(series))
        f = interp.interp1d(x, series-value)

        negindsL = np.where(np.logical_and(series < value, x < half))[0]
        negindsR = np.where(np.logical_and(series < value, x > half))[0]
        iL = optimize.brentq(f, negindsL[-1], negindsL[-1]+1)#half)
        iR = optimize.brentq(f, negindsR[0]-1, negindsR[0])#half, wR)
    return iR-iL

def subdivide(tdata, ydata, noise, rms=True, minsep=16, maxsep=64, fac=1.25):
    """ Subdivide an array and determine where knots should be placed in spline smoothing """
    N = len(ydata)
    if N <= minsep or N <= 4: # minimum is 4
        return []

    '''
    if rms:
        localrms = RMS(ydata)
        if localrms<fac*noise and N <= maxsep:
            return []
    else:
        p = np.polyfit(tdata, ydata, 1)
        f = np.poly1d(p)
        if RMS(ydata-f(tdata))<fac*noise and N <= maxsep:
            return []
    '''
    #'''
    ks = np.arange(1, 5)
    chisqs = np.zeros(len(ks))
    for i, k in enumerate(ks):
        p = np.polyfit(tdata, ydata, k)
        f = np.poly1d(p)
        resids = ydata-f(tdata)
        chisqs[i] = np.sum(resids**2)/(N-k-1)

    #print chisqs
    if np.argmin(chisqs) < 2 and N <= maxsep:
        #p = np.polyfit(tdata, ydata, np.argmin(chisqs))
        #f = np.poly1d(p)
        #resids = ydata-f(tdata)
        return []
    #'''

    # Test new knot at the midpoint
    half = N/2
    tdataL = tdata[:half]
    tdataR = tdata[half:]
    ydataL = ydata[:half]
    ydataR = ydata[half:]
    knotsL = subdivide(tdataL, ydataL, noise, rms=rms, minsep=minsep, maxsep=maxsep, fac=fac)
    knotsR = subdivide(tdataR, ydataR, noise, rms=rms, minsep=minsep, maxsep=maxsep, fac=fac)
    '''
    # if the left and right sides are disproportionate, re-run with lower minsep
    lenL = len(knotsL)
    lenR = len(knotsR)
    if (lenL == 0 and lenR >= 2) or (lenL != 0 and lenR/float(lenL) < 4):
        knotsL = subdivide(tdataL, ydataL, noise, rms=rms, minsep=4, maxsep=maxsep, fac=fac)
        print len(knotsL), len(knotsR)
    if (lenR == 0 and lenL >= 2) or (lenR != 0 and lenL/float(lenR) < 4):
        knotsR = subdivide(tdataR, ydataR, noise, rms=rms, minsep=4, maxsep=maxsep, fac=fac)
        print len(knotsL), len(knotsR)
    '''
    return np.concatenate((knotsL, knotsR, [half+tdata[0]]))

def fit_components(xdata, ydata, mode='gaussian', N=1, allownegative=False):
    nbins = len(xdata)
    if allownegative:
        imax = np.argmax(np.abs(ydata))
    else:
        imax = np.argmax(ydata)
    if mode == 'gaussian':
        pinit = np.array([ydata[imax], xdata[imax], 0.02*nbins]) #2% duty cycle
    elif mode == 'vonmises':
        pinit = np.array([ydata[imax], xdata[imax], nbins])
    fitter = eval(mode)

    # perform this fit iteratively
    for n in range(1, N+1):
        def fitfunc(p, x):
            retval = np.zeros(len(x))
            for i in range(n):
                retval += fitter(x, p[3*i], p[3*i+1], p[3*i+2])
            return retval
        def errfunc(p, x, y):
            return y - fitfunc(p, x)

        out = optimize.leastsq(errfunc, pinit, args=(xdata, ydata), full_output=True)
        if n == N:
            break
        # Re-define initial conditions for next round
        pfit = out[0]

        resids = ydata-fitfunc(pfit, xdata)
        if allownegative:
            imax = np.argmax(np.abs(resids))
        else:
            imax = np.argmax(resids)
        if mode == 'gaussian':
            pinitprime = np.array([resids[imax], xdata[imax], 0.02*nbins]) #2% duty cycle
        elif mode == 'vonmises':
            pinitprime = np.array([resids[imax], xdata[imax], nbins])#1.0/(0.02*nbins)])#need sqrt?
        pinit = np.concatenate((pfit, pinitprime))

    s_sq = (errfunc(out[0], xdata, ydata)**2).sum()/(len(ydata)-len(pinit)-1) #-1 included here!
    return fitfunc, errfunc, out[0], out[1], s_sq

def fit_gaussians(xdata, ydata, N=1):
    return fit_components(xdata, ydata, mode='gaussian', N=N)

def fit_vonmises(xdata, ydata, N=1):
    return fit_components(xdata, ydata, mode='vonmises', N=N)

def gaussian(x, amp, mu, sigma):
    return amp*np.exp(-0.5*((x-mu)/sigma)**2)

def vonmises(x, amp, mu, kappa):
    #return amp*np.exp(kappa*np.cos(x-mu))/(2*np.pi*special.iv(0, kappa))
    '''
    # More numerically stable:
    ive(v, z) = iv(v, z) * exp(-abs(z.real)), z here must be positive number here
    therefore
    iv(v, z) = ive(v, z) / exp(-z)
    log(iv(v,z)) = log(ive(v, z) / exp(-z)) = log(ive(v,z)) - log(exp(-z)) = log(ive(v,z)) + z
    '''
    numer = kappa*np.cos(x-mu)
    denom = np.log(2*np.pi) + np.log(special.ive(0, kappa)) + kappa
    y = np.exp(numer - denom)
    #y /= np.max(y)
    # Allow for negatives
    y /= np.max(np.abs(y))
    return amp*y

def pbf_clean(t, y, g=None, taud=1.0, opw=None, gamma=0.05, m=1.0, x=1.5, stop=1.5):
    '''
    gamma - loop gain
    m = factor that determines relative strength of Gamma versus f_r
    x =
    stop = stopping criterion sigma. Should be the same as x?
    '''

    N = len(t)
    if g is None:
        def g(t, tmax=0, taud=1.0):
            retval = np.exp(-t/taud)/taud
            retval = shiftit(retval, -tmax) #shiftit of zero introducing baseline oscillation?
            return retval
    if opw is None:
        sigma_opw = RMS(y)
    else:
        sigma_opw = RMS(y[opw])

    N_components = 0

    Dy = np.copy(y)

    i_components = []
    y_components = []


    oldrms = 0.0
    n = 0
    while True:
        imax = np.argmax(Dy)
        tmax = t[imax]
        ymax = Dy[imax]
        i_components.append(imax)
        y_components.append(ymax*gamma)
        N_components += 1
        Dy -= y_components[-1]*g(t, tmax, taud=taud)
        rms = RMS(Dy)
        #if np.all(np.abs(Dy)<3.0*sigma_opw) or oldrms == rms:
        if rms <= stop*sigma_opw or oldrms == rms:
            if N_components == 1:
                stop -= 0.5
                if stop == 0.0:
                    break
                else:
                    continue
            else:
                break
        oldrms = rms
        n += 1

    i_components = np.array(i_components)
    y_components = np.array(y_components)
    t_components = np.zeros_like(y_components)

    C = np.zeros(len(t))

    for n in range(N_components):
        #c[i_components[n]] += y_components[n]
        C += gaussian(t, y_components[n], t[i_components[n]], 1.0) #what width to use?
        t_components[n] = t[i_components[n]]

    C /= np.max(C)

    # N_f metric
    inds = np.where(np.abs(C) < 3*sigma_opw)[0] #C?
    N_f = float(len(inds))/len(C)
    #inds = np.where(np.abs(Dy-np.mean(Dy))<3*sigma_opw)[0] #C?
    #N_f = float(len(inds))/len(Dy)

    # sigma_offc metric
    sigma_offc = RMS(Dy[opw])/sigma_opw

    # Gamma metric
    sumy = np.sum(y_components)
    tbar = np.sum(t_components*y_components)/sumy
    avgt = lambda n: np.sum(np.power((t_components-tbar), n)*y_components)/sumy
    Gamma = avgt(3)/np.power(avgt(2), 1.5)
    #print Gamma

    # f_r metric
    inds = np.where(Dy < -x*sigma_opw)[0] #the step function
    #print len(inds)
    f_r = (m/(N*sigma_opw**2)) * np.sum(Dy[inds]**2)

    return Dy, C, N_f, sigma_offc, Gamma, f_r

def pbf_fourier(t, y, g=None, taud=1.0, opw=None, m=1.0, x=1.5, **kwargs):
    N = len(t)

    if g is None:
        def g(t, taud=1.0):
            retval = np.exp(-t/taud)/taud
            return retval
    if opw is None:
        sigma_opw = RMS(y)
    else:
        sigma_opw = RMS(y[opw])

    t = np.array(t, dtype=np.float)

    Yf = np.fft.fft(y)
    dt = np.diff(t)[0]
    #f = np.fft.fftshift(np.fft.fftfreq(N, dt)) #shift?
    f = np.fft.fftfreq(N, dt)
    gt = g(t, taud=taud)
    Gf = np.fft.fft(gt)
    #rt = gaussian(t, 1.0, 0, 1.0)+gaussian(t, 1.0, N, 1.0)
    #Rf = np.fft.fft(rt).real
    #Rf = gaussian(f, np.sqrt(2*np.pi), 0, 1/(2*np.pi)) #FT of Gaussian of unit amplitude and width
    #print "gt area", np.trapz(gt, x=t)
    #print "Gf area", np.trapz(Gf, x=f), np.trapz(np.sqrt(np.abs(Gf)**2), x=f)
    #print "Rf area", np.trapz(Rf, x=f)
    #plt.plot(t, rt)
    #plt.plot(f, Rf, 'k.')
    Rf = gaussian(f, np.sqrt(2*np.pi), 0, 1/(2*np.pi)) #FT of Gaussian of unit amplitude and width
    Rf = np.fft.fftshift(Rf)
    #plt.plot(f, Rf, 'b.')
    #plt.show()
    #raise SystemExit
    Xf = Yf/(Gf*Rf)
    xt = np.fft.ifft(Xf).real

    # Rescale the deconvolved profile
    #Yprime = np.correlate(xt, gt[::-1], 'full')[:N]
    #Yprime = np.correlate(np.correlate(xt, gt[::-1], 'full')[:N], rt[::-1], 'full')[:N]
    #xt = xt *np.trapz(y, x=t)/ np.trapz(Yprime, x=t)
    #xt = xt *np.trapz(y, x=t)/ np.trapz(np.abs(xt), x=t)#?
    # scale by offpulse noise
    #xt = xt * sigma_opw / RMS(xt[opw])
    # scale by the peak?
    xt = xt * np.max(y)/np.max(xt)
    #plt.plot(t, y)
    #plt.plot(t, xt)
    #plt.show()

    # N_f metric
    #inds = np.where(np.abs(xt)<3*sigma_opw)[0] #C?
    #N_f = float(len(inds))/len(xt)
    N_f = 0 #no residuals

    # sigma_offc metric
    sigma_offc = RMS(xt[opw])/sigma_opw

    # Gamma metric
    inds = np.where(xt > x*sigma_opw)[0]
    #inds = np.where(xt>-100000)[0]

    sumx = np.sum(xt[inds])
    tbar = np.sum(t[inds]*xt[inds])/sumx
    avgt = lambda n: np.sum(np.power((t[inds]-tbar), n)*xt[inds])/sumx
    #print "tbar", tbar, avgt(3), avgt(2)**1.5
    Gamma = avgt(3)/np.power(avgt(2), 1.5)
    #Gamma = np.abs(Gamma)
    Gamma = -Gamma #meh
    #print Gamma

    # f_r metric
    # is this the correct modification?
    #sigma_opw = RMS(xt[opw])
    inds = np.where(xt < -x*sigma_opw)[0] #the step function
    #print len(inds)
    f_r = (m/(N*sigma_opw**2)) * np.sum(xt[inds]**2)
    #print Gamma, f_r

    return np.zeros(N), xt, N_f, sigma_offc, Gamma, f_r

'''
Return RMS
'''
def RMS(series, subtract_mean=False):
    if subtract_mean:
        series = series - np.mean(series)
    return np.sqrt(np.mean(np.power(series, 2)))

'''
Return weighted sample mean and std
http://en.wikipedia.org/wiki/Weighted_mean#Weighted_sample_variance
'''
def weighted_moments(series, weights, unbiased=False, harmonic=False):
    if len(series) == 1:
        return series, 1.0/np.sqrt(weights)
    series = np.array(series)
    weights = np.array(weights)
    weightsum = np.sum(weights)
    weightedmean = np.sum(weights*series)/weightsum
    weightedvariance = np.sum(weights*np.power(series-weightedmean, 2))
    if harmonic:
        return weightedmean, harmonic_mean(1.0/weights)
    elif unbiased:
        weightsquaredsum = np.sum(np.power(weights, 2))
        return weightedmean, np.sqrt(weightedvariance * weightsum / (weightsum**2 - weightsquaredsum))
    else:
        return weightedmean, np.sqrt(weightedvariance / weightsum)

### ==================================================
### Parallelization
### ==================================================

#http://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class/10525471#10525471
def spawn(f):
    def fun(ppipe, cpipe, x):
        ppipe.close()
        cpipe.send(f(x))
        cpipe.close()
    return fun

def parmap(f, X):
    pipe = [Pipe() for x in X]
    proc = [Process(target=spawn(f), args=(p, c, x)) for x, (p, c) in izip(X, pipe)]
    [p.start() for p in proc]
    ret = [p.recv() for (p, c) in pipe]
    [p.join() for p in proc]
    return ret

### ==================================================
### Optimizations of JMC's code
### ==================================================

def shiftit_old(y, shift):
    """
    shifts array y by amount shift (in sample numbers)
    uses shift theorem and FFT
    shift > 0  ==>  lower sample number (earlier)
    modeled after fortran routine shiftit
    Optimized from JMC's code
    """
    yfft = np.fft.fft(y)
    size = np.size(y) #saves time
    constant = (shift*2*np.pi)/float(size) #needs a negative here for the right direction, put it in?
    theta = constant*np.arange(size)
    c = np.cos(theta)
    s = np.sin(theta)
    work = np.zeros(size, dtype='complex')
    work.real = c * yfft.real - s * yfft.imag
    work.imag = c * yfft.imag + s * yfft.real
    # enforce hermiticity

    work.real[size//2:] = work.real[size//2:0:-1]
    work.imag[size//2:] = -work.imag[size//2:0:-1]
    work[size//2] = 0.+0.j
    workifft = np.fft.ifft(work)
    return workifft.real

def shiftit(y, shift):
    '''
    Speed-ups via Paul Baker, Ross Jennings
    '''
    if isinstance(shift,np.ndarray):
        shift = shift[...,np.newaxis]
    N = y.shape[-1]
    yfft = np.fft.rfft(y)
    fs = np.fft.rfftfreq(N)#, d=dt)
    phase = 1j*2*np.pi*fs*shift  #reversed from Paul's code so that this matches the previous convention
    yfft_sh = yfft * np.exp(phase)
    return np.fft.irfft(yfft_sh)

def find_fwhm(array):
    """
    Finds full width at half maximum in sample numbers via interpolation.
    """
    ninterp = 3                   # 3 pt linear interpolation
    # put maximum in center of array
    amax = np.max(array)
    amaxloc = np.argmax(array)
    shift = int(np.size(array)/2. - amaxloc)
    array = np.roll(array, shift)
    xvec = range(np.size(array))
    amaxloc = np.argmax(array) # Fix by MTL
    half1 = np.where(np.diff(np.sign(array[:amaxloc]-amax/2.)))[0][0]
    half2 = np.where(np.diff(np.sign(array[amaxloc:]-amax/2.)))[0][0]
    start1 = half1-(ninterp-1)//2
    start2 = amaxloc+half2-(ninterp-1)//2
    xinterp1 = xvec[start1:start1+ninterp]
    yinterp1 = array[start1:start1+ninterp]
    xinterp2 = xvec[start2:start2+ninterp]
    yinterp2 = array[start2:start2+ninterp]
    hwhm_minus = -np.interp(amax/2., yinterp1, xinterp1)+amaxloc
    hwhm_plus = np.interp(amax/2., yinterp2[::-1], xinterp2[::-1])-amaxloc
    fwhm = hwhm_minus+hwhm_plus
    return fwhm

def toa_errors_additive(tfft, b, sigma_t):
    """
    Calculates error in b = scale factor and tau = TOA due to additive noise.

    input:
        fft of template
        b = fit value for scale factor
	sigma_t = rms additive noise in time domain
    output:
  	sigma_b
        sigma_tau
    """
    Nfft = np.size(tfft)
    Nsum = Nfft // 2
    kvec = np.arange(1, Nsum)
    sigma_b = sigma_t*np.sqrt(float(Nfft) / (2.*np.sum(np.abs(tfft[1:Nsum])**2)))
    sigma_tau = (sigma_t*Nfft/(2.*np.pi*np.abs(b))) * np.sqrt(float(Nfft) / (2.*np.sum(kvec**2*np.abs(tfft[1:Nsum])**2)))
    return sigma_tau, sigma_b

def tfresids(params, tfft, pfft):
    """
    """
    b = params[0]
    tau = params[1]
    Nfft = np.size(pfft)
    Nsum = Nfft//2
    arg = (2.*np.pi*tau/float(Nfft)) * np.arange(0., Nfft, 1.)
    phasevec = np.cos(arg) - 1j*np.sin(arg)
    #resids = abs(pfft[1:Nsum] - b*tfft[1:Nsum]*phasevec[1:Nsum])
    resids = np.abs(pfft[1:Nsum] - b*tfft[1:Nsum]*phasevec[1:Nsum])
    return resids

def get_toa3(template, profile, sigma_t, dphi_in=0.1, snrthresh=0., nlagsfit=5, norder=2):
    """
    Calculates TOA and its error in samples (bins).
    Uses least-squares method in frequency domain, minimizing chi^2.
    Also calculates scale factor for template matching.
    Input: template = template file; if normalized to unity max,
                      the scale factor divided by the input sigma_t is
                      the peak to rms S/N.
           profile = average profile to process
           sigma_t = off pulse rms in same units as profile.
    Output:
    tauccf = TOA (bins) based on parabloic interpolation of CCF.
    tauhat = TOA (bins) using Fourier-domain fitting.,
    bhat = best-fit amplitude of pulse.
    sigma_tau = error on tauhat.
    sigma_b = error on bhat.
    snr = bhat/sigma_t.
    rho = cross correlation coefficient between template and centered profile.
    """
    # Some initial values:
    snr_coarse = np.max(profile)/sigma_t
    tauhat = 0.
    bhat = 0.
    sigma_tau = -1.
    sigma_b = -1.
    rho = -2.

    # find coarse estimates for scale factor and tau from CCF maximum
    #  (quadratically interpolated)
    ccf = np.correlate(template, profile, 'full')
    lags = np.arange(-np.size(profile)+1., np.size(profile), 1.)
    ccfmaxloc = ccf.argmax()
    ccffit = ccf[ccfmaxloc-(nlagsfit-1)//2:ccfmaxloc+(nlagsfit-1)//2+1]
    lagfit = lags[ccfmaxloc-(nlagsfit-1)//2:ccfmaxloc+(nlagsfit-1)//2+1]

    p = np.polyfit(lagfit, ccffit, norder)
    ccfhat = p[0] + p[1]*lagfit + p[2]*lagfit**2
    tauccf = p[1]/(2.*p[2])

    # roughly center the pulse to line up with the template:
    ishift = int(-tauccf)
    profile = np.roll(profile, ishift)

    bccf = sum(template*profile)/sum(template**2)

    # Search range for TOA using Fourier-domain method:
    # expect -fwhm/2 < tauhat < fwhm/2  since pulse has been centered

    # fwhm, taumin, taumax currently not used.  But should we do a
    # windowed TOA calculation?
    fwhm = find_fwhm(template)		# fwhm in samples (bins)
    taumin = -fwhm/2.
    taumax = fwhm/2.

    tfft = np.fft.fft(template)
    pfft = np.fft.fft(profile)
    bhat0 = bccf
    tauhat0 = tauccf+ishift
    paramvec0 = np.array((bhat0, tauhat0))

    paramvec = optimize.minpack.leastsq(tfresids, paramvec0, args=(tfft, pfft))
    bhat = paramvec[0][0]
    tauhat = paramvec[0][1]

    sigma_tau, sigma_b = toa_errors_additive(tfft, bhat, sigma_t)
    # snr = scale factor / sigma_t:
    snr = (bhat*np.max(template))/sigma_t

    # rho = correlation coefficient of template and shifted profile:
    profile_shifted = shiftit(profile, +tauhat)	# checked sign: this is correct

    # TBA: correct rho for off-pulse noise.
    # Two possibilities:
    #     1. subtract noise variance term  from sum(profile_shifted**2)
    #  or 2. calculate ACF of profile_shifted with a one or two sample lag.
    rho = np.sum(template*profile_shifted) / np.sqrt(np.sum(template**2)*np.sum(profile_shifted**2))
    tauhat = tauhat - ishift	# account for initial shift
    return tauccf, tauhat, bhat, sigma_tau, sigma_b, snr, rho
