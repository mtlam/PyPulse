'''
Michael Lam 2015

Useful mathematical commands
'''

import numpy as np
import scipy.fftpack as fft
import scipy.optimize as optimize
import scipy.stats as stats
import scipy.interpolate as interp
import scipy.special as special
from scipy.signal import fftconvolve,correlate
import matplotlib.pyplot as plt

from multiprocessing import Process, Pipe
from itertools import izip





'''
ACF
var=True: calculate variance, var=False, do not calculate. var=number: use as number
Include mean subtraction?
Include lagaxis function?
'''
def acf(array,var=False,norm_by_tau=True,lagaxis=False): #set lagaxis=True?
    array=np.array(array)
    N=len(array)
    if var==True:
        var=np.var(array)
    elif var==False:
        var=1
        
    lags = np.arange(-(N-1),N,dtype=np.float)
    if norm_by_tau:
        taus=np.concatenate((np.arange(1,N+1),np.arange(N-1,0,-1)))
        if lagaxis:
            return lags,np.correlate(array,array,"full")/(var*taus)
        return np.correlate(array,array,"full")/(var*taus)
    if lagaxis:
        return lags,np.correlate(array,array,"full")/(var*N)
    return np.correlate(array,array,"full")/(var*N)
    
#Do not provide bins but provide edges?
#error bars?
def lagfunction(func,t,x,e=None,dtau=1,tau_edges=None,mirror=False):
    length = len(x)
    if tau_edges is None:
        num_lags = np.ceil((np.max(t) - np.min(t))/dtau) + 1 #+1?
        taus = np.arange(num_lags) * dtau
        tau_edges = (taus[:-1] + taus[1:])/2.0
        tau_edges = np.hstack((tau_edges,[tau_edges[-1]+dtau]))
        N_taus = np.zeros(num_lags)
        retval = np.zeros(num_lags)
        variance = np.zeros(num_lags)
    else:
        dtau = np.median(np.diff(tau_edges)) #not quite working
        taus = tau_edges - dtau
        #taus = np.concatenate((taus,[taus[-1]+dtau]))
        N_taus = np.zeros(len(tau_edges))#-1)
        retval = np.zeros(len(tau_edges))#-1) #this should just be "mean"
        variance = np.zeros(len(tau_edges))
        
    weighted=False
    if e != None:
        weighted=True
    


    # this could be sped up several ways
    I = list(range(length))
    for i in I:
        for j in I:
            dt = np.abs(t[i]-t[j])
            index = np.where(dt < tau_edges)[0] #<=?
            if len(index)==0:
                continue
        
            index = index[0] #get the lowest applicable lag value

            N_taus[index] += 1
            #Replace this with online algorithm?
            retval[index] += func(x[i],x[j])
    #    print N_taus

    #divide by zero problem!, only with one-pass algorithm
    retval = retval / N_taus


    if mirror: #fix this
    #mirror each:
        taus = np.concatenate((-1*taus[::-1][:-1],taus))
        retval = np.concatenate((retval[::-1][:-1],retval))
        #retval /= 2 #double counting, can speed this up! why no division by 2?
    #return tau_edges,retval
    return taus,retval
    #return tau_edges,retval #BAD





def acf2d(array,speed='fast',mode='full',xlags=None,ylags=None):
    if speed == 'fast' or speed == 'slow':
        ones = np.ones(np.shape(array))
        norm = fftconvolve(ones,ones,mode=mode) #very close for either speed
        if speed=='fast':
            return fftconvolve(array,np.flipud(np.fliplr(array)),mode=mode)/norm
        else:
            return correlate(array,array,mode=mode)/norm
    elif speed == 'exact':
        #NOTE: (r,c) convention is flipped from (x,y), also that increasing c is decreasing y
        LENX = len(array[0])
        LENY = len(array)
        if xlags is None:
            xlags = np.arange(-1*LENX+1,LENX)
        if ylags is None:
            ylags = np.arange(-1*LENY+1,LENY)
        retval = np.zeros((len(ylags),len(xlags)))
        for i,xlag in enumerate(xlags):
            print(xlag)
            for j,ylag in enumerate(ylags):
                if ylag > 0 and xlag > 0:
                    A = array[:-1*ylag,xlag:] #the "stationary" array
                    B = array[ylag:,:-1*xlag]
                elif ylag < 0 and xlag > 0:
                    A = array[-1*ylag:,xlag:]
                    B = array[:ylag,:-1*xlag]
                elif ylag > 0 and xlag < 0:#optimize later via symmetries
                    A = array[:-1*ylag,:xlag]
                    B = array[ylag:,-1*xlag:]
                elif ylag < 0 and xlag < 0:
                    A = array[-1*ylag:,:xlag]
                    B = array[:ylag,-1*xlag:]
                else: #one of the lags is zero
                    if ylag == 0 and xlag > 0:
                        A = array[-1*ylag:,xlag:]
                        B = array[:,:-1*xlag]
                    elif ylag == 0 and xlag < 0:
                        A = array[-1*ylag:,:xlag]
                        B = array[:,-1*xlag:]
                    elif ylag > 0 and xlag == 0:
                        A = array[:-1*ylag,:]
                        B = array[ylag:,-1*xlag:]
                    elif ylag < 0 and xlag == 0:
                        A = array[-1*ylag:,:]
                        B = array[:ylag,-1*xlag:]
                    else:
                        A = array[:,:]
                        B = array[:,:]
                        #print xlag,ylag,A,B
                C = A*B
                C = C.flatten()
                goodinds = np.where(np.isfinite(C))[0] #check for good values
                retval[j,i] = np.mean(C[goodinds])            
        return retval


def lagaxis(arg,dtau=1):
    if type(arg)==type([]) or type(arg)==np.ndarray: #generate a lag axis based on a time axis
        length = len(arg)
        dtau = np.mean(np.diff(arg))
        return np.arange(-1*length+1,length)*dtau
    else: # Returns a generic lag axis
        half = arg//2 #arg should be odd
        return np.arange(-1*half,half+1)*dtau



#Taken from diagnostics.py, set default threshold=3
def zct(series,threshold=3,full=False,meansub=False):
    count=0
    N=len(series)
    current=np.sign(series[0])
    if meansub:
        series-=np.mean(series)
    for i in range(1,N):
        #print np.sign(series[i])
        if np.sign(series[i]) != current:
            count+=1 #Crossed zero, add to count
            current*=-1 #Flip sign
    average_zw=float(N-1)/2
    sigma_zw=np.sqrt(N-1)/2
    if (average_zw - threshold*sigma_zw) <= count <= (average_zw + threshold*sigma_zw):
        if full:
            return True,abs(count-average_zw)/sigma_zw,count
        return True
    else:
        if full:
            return False,abs(count-average_zw)/sigma_zw,count
        return False 




'''
Decimate the data
Be careful with window_len!
if remainder: include the remainder?
'''
def decimate(x,window_len,error=False):#,mean=True,remainder=False):
    if window_len==1:
        return x
    length = len(x)
    retval = np.zeros(length/window_len)
    counts = np.zeros_like(retval)
    if error:
        errorretval = np.zeros_like(retval)
        for i in range(len(retval)):
            win = x[i*window_len:(i+1)*window_len]
            retval[i] = np.mean(win)
            errorretval[i] = np.std(win)/np.sqrt(window_len)
        return retval,errorretval
    else:
        for i in range(window_len):
            retval+=x[i:length:window_len]
        return retval/window_len



def imshow(x,ax=None,origin='lower',interpolation='nearest',aspect='auto',**kwargs):
    if ax is not None:
        im=ax.imshow(x,origin=origin,interpolation=interpolation,aspect=aspect,**kwargs)
    else:
        im=plt.imshow(x,origin=origin,interpolation=interpolation,aspect=aspect,**kwargs)
    return im

'''
Histogram

Allow for intervals or number of bins
bins: Provide an array of bins
'''
def histogram(values,interval=1.0,bottom=None,full=False,bins=None,plot=False,show=True,horizontal=False,normalize=False,logbins=False,**kwargs):
    if bins is None:
        factor=1.0/interval
        if bottom is None:
            minval=(np.fix(factor*min(values))-1)/factor
        else:
            minval=bottom
        maxval=(np.ceil(factor*max(values))+1)/factor
        #bins=np.arange(minval,maxval,interval)
        bins=np.arange(minval,maxval+interval,interval)

    else:
        minval=bins[0]
        maxval=bins[-1]
    hist,bins=np.histogram(values,bins=bins)
    if logbins: #need to adjust the centers
        center=np.sqrt(bins[:-1]*bins[1:]) #geometric mean = log-average
    else:
        center=(bins[:-1]+bins[1:])/2.0 #arithmetic mean

    if normalize:
        hist = np.array(hist,dtype=np.float)/(float(interval)*np.sum(hist))
        

    if plot:
        plothistogram(center,hist,interval,show=show,horizontal=horizontal,logbins=logbins,**kwargs)
        return

    if full:
        return center,hist,bins,minval,maxval
    return center,hist

#Need to take into account logbins!
#Must be left aligned!
def plothistogram(center,hist,interval=1.0,bins=None,steps=False,show=True,horizontal=False,logbins=False,centerbin=False,ax=None,**kwargs):
    if steps or bins is not None:
        binsize = np.mean(np.diff(center))
        center = np.concatenate(([center[0]-binsize],center,[center[-1]+binsize]))
        if centerbin:
            center -= binsize/2.0
        hist = np.concatenate(([0],hist,[0]))
        if ax is None:
            p,=plt.plot(center,hist,drawstyle='steps-mid',**kwargs)
        else:
            p,=ax.plot(center,hist,drawstyle='steps-mid',**kwargs)
    else:
        if centerbin:
            binsize = np.mean(np.diff(center))
            center -= binsize/2.0

        if horizontal:
            if ax is None:
                p=plt.barh(center,hist,height=interval,align='center',**kwargs)
            else:
                p=ax.barh(center,hist,height=interval,align='center',**kwargs)
        else:
            if ax is None:
                p=plt.bar(center,hist,width=interval,align='center',**kwargs)
            else:
                p=ax.bar(center,hist,width=interval,align='center',**kwargs)
    if show:
        plt.show()
    return p


#Creates empirical cdf
def ecdf(values,sort=True):
    if sort:
        values = np.sort(values)
    return values,np.linspace(0,1,len(values))

EPS = special.erf(1.0/np.sqrt(2))/2.0
def pdf_to_cdf(pdf,dt=1):
    return np.cumsum(pdf)*dt
def likelihood_evaluator(x,y,cdf=False,median=False,pm=True,values=None):
    """
    cdf: if True, x,y describe the cdf
    median: if True, use the median value, otherwise the peak of the pdf (assuming cdf=False
    pm: xminus and xplus are the plus/minus range, not the actual values

    Future: give it values to grab off the CDF (e.g. 2 sigma, 99%, etc)
    values: use this array
    """
    if not cdf:
        y = y/np.trapz(y,x=x)
        ycdf = pdf_to_cdf(y,dt=(x[1]-x[0]))
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

        inds = np.arange(inda,indc+1) #including indc    
        #print indc-inda,np.trapz(L[inds],x=Vrs[inds])
        xval = x[indb]
        if pm:
            xminus = x[indb] - x[inda]
            xplus = x[indc] - x[indb]
        else:
            xminus = x[inda]
            xplus = x[indc]
        x95 = x[indd]

        return xval,xminus,xplus,x95
    else:
        retval = np.zeros_like(values)
        for i,v in enumerate(values):
            indv = np.argmin(np.abs(ycdf - v))
            retval[i] = x[indv]
        return retval




'''
Normalize an array to unit height
Below: normalize 
'''
def normalize(array,simple=False,minimum=None):
    if simple:
        return array/np.max(array)
    maximum=np.max(array)
    if minimum is None:
        minimum=np.min(array)
    return (array-minimum)/(maximum-minimum)
def normalize_area(array,x=None,full=False):
    if x is None:
        x=np.arange(len(array))
    area=np.trapz(array,x=x)
    if full:
        return array/area,area
    return array/area


'''
Center the maximum value of the array
Follows profiles.py
'''
def center_max(array,full=False):    
    maxind=np.argmax(array)
    length=len(array)
    centerind=int(length/2)
    diff=centerind-maxind
    if full:
        return diff,np.roll(array,diff)
    return np.roll(array,diff)


#Follow profiles.py
#notcentered is very rudimentary
#have norm be simple
def FWHM(series,norm=True,simple=False,notcentered=False):
    if norm:
        series=normalize(series) #assumes these are floats, not integers!
    y=np.abs(series-0.5)
    
    N=len(series)
    half=N//2

    wL = 0
    wR = N-1

    
    #initial solution
    if notcentered:
        series = center_max(series)
#        half=np.argmax(series)
    iL=np.argmin(y[:half])
    iR=np.argmin(y[half:])+half
    if not simple:
        x=np.arange(len(series))
        f=interp.interp1d(x,series-0.5)

        negindsL = np.where(np.logical_and(series<0.5,x<half))[0]
        negindsR = np.where(np.logical_and(series<0.5,x>half))[0]
        iL=optimize.brentq(f,negindsL[-1],negindsL[-1]+1)#half)
        iR=optimize.brentq(f,negindsR[0]-1,negindsR[0])#half,wR)
    return iR-iL



def subdivide(tdata,ydata,noise,rms=True,minsep=16,maxsep=64,fac=1.25):
    """ Subdivide an array and determine where knots should be placed in spline smoothing """
    N = len(ydata)
    if N <= minsep:
        return []

    if rms:
        localrms = RMS(ydata)
        if localrms<fac*noise and N <= maxsep:
            return []
    else:
        p = np.polyfit(tdata,ydata,1)
        f = np.poly1d(p)
        if RMS(ydata-f(tdata))<fac*noise and N <= maxsep:
            return []

    # Test new knot at the midpoint
    half = N/2
    tdataL = tdata[:half]
    tdataR = tdata[half:]
    ydataL = ydata[:half]
    ydataR = ydata[half:]
    knotsL = subdivide(tdataL,ydataL,noise,rms=rms,minsep=minsep,maxsep=maxsep,fac=fac)
    knotsR = subdivide(tdataR,ydataR,noise,rms=rms,minsep=minsep,maxsep=maxsep,fac=fac)
    return np.concatenate((knotsL,knotsR,[half+tdata[0]]))


'''
Return RMS
'''
def RMS(series,subtract_mean=False):
    if subtract_mean:
        series = series - np.mean(series)
    return np.sqrt(np.mean(np.power(series,2)))


'''
Return weighted sample mean and std
http://en.wikipedia.org/wiki/Weighted_mean#Weighted_sample_variance
'''
def weighted_moments(series,weights,unbiased=False,harmonic=False):
    if len(series)==1:
        return series,1.0/np.sqrt(weights)
    series=np.array(series)
    weights=np.array(weights)
    weightsum=np.sum(weights)
    weightedmean = np.sum(weights*series)/weightsum
    weightedvariance = np.sum(weights*np.power(series-weightedmean,2))
    if harmonic:
        return weightedmean, harmonic_mean(1.0/weights)
    elif unbiased:
        weightsquaredsum=np.sum(np.power(weights,2))
        return weightedmean, np.sqrt(weightedvariance * weightsum / (weightsum**2 - weightsquaredsum))
    else:
        return weightedmean, np.sqrt(weightedvariance / weightsum)



### ==================================================
### Parallelization
### ==================================================


#http://stackoverflow.com/questions/3288595/multiprocessing-how-to-use-pool-map-on-a-function-defined-in-a-class/10525471#10525471
def spawn(f):
    def fun(ppipe, cpipe,x):
        ppipe.close()
        cpipe.send(f(x))
        cpipe.close()
    return fun

def parmap(f,X):
    pipe=[Pipe() for x in X]
    proc=[Process(target=spawn(f),args=(p,c,x)) for x,(p,c) in izip(X,pipe)]
    [p.start() for p in proc]
    ret = [p.recv() for (p,c) in pipe]
    [p.join() for p in proc]
    return ret





### ==================================================
### Optimizations of JMC's code
### ==================================================

def shiftit(y, shift):
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




def find_fwhm(array):
    """
    Finds full width at half maximum in sample numbers via interpolation.
    """
    ninterp=3                   # 3 pt linear interpolation
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
    kvec = np.arange(1,Nsum)
    sigma_b = sigma_t*np.sqrt(float(Nfft) / (2.*np.sum(np.abs(tfft[1:Nsum])**2)))
    sigma_tau = (sigma_t*Nfft/(2.*np.pi*np.abs(b))) * np.sqrt(float(Nfft) / (2.*np.sum(kvec**2*np.abs(tfft[1:Nsum])**2)))
    return sigma_tau, sigma_b


def tfresids(params, tfft, pfft):
    """
    """
    b=params[0]
    tau=params[1]
    Nfft = np.size(pfft)
    Nsum = Nfft//2
    arg=(2.*np.pi*tau/float(Nfft)) * np.arange(0., Nfft, 1.)
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
