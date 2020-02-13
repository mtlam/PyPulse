'''
Michael Lam 2015

Define interpulse alignment as putting the peak value at len/4.
Interpulse will be roughly at 3*len/4

Figure out way to add/average SPs.
'''
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import scipy.stats as stats
import scipy.signal as signal
import pypulse.utils as u

get_toa = u.get_toa3 #try this one

#ACF=lambda p: np.correlate(p, p, "full") #no longer used

'''
data : data array
mpw : Main pulse window
ipw : Interpulse window
opw : off pulse window
prepare : Process data according to some metrics (deprecated)
align : Roll pulse by this amount
period : Set pulse period
windowsize: Override mpw, ipw, opw, define an offpulse window about the minimum of this size
'''


class SinglePulse(object):
    def __init__(self, data, mpw=None, ipw=None, opw=None, prepare=False,
                 align=None, period=None, windowsize=None):
        if len(np.shape(data)) != 1:
            raise IndexError("SinglePulse received incorrect data shape")

        self.data = np.array(data)
        if mpw is not None:
            self.mpw = np.array(mpw)
        else:
            self.mpw = None
        if ipw is not None:
            self.ipw = np.array(ipw)
        else:
            self.ipw = None
        #Define off pulse
        self.nbins = self.getNbins()
        self.bins = np.arange(self.nbins)
        self.phases = np.arange(self.nbins, dtype=np.float)/self.nbins

        if windowsize is not None:
            self.calcOffpulseWindow(windowsize)

        elif opw is None:
            if self.mpw is None and self.ipw is None:
                self.opw = None #do not define any windows
            elif self.ipw is None:
                self.opw = self.bins[np.logical_not(np.in1d(self.bins, mpw))]
            elif self.mpw is None:
                self.opw = self.bins[np.logical_not(np.in1d(self.bins, ipw))]
            else:
                self.opw = self.bins[np.logical_not(np.logical_or(np.in1d(self.bins, mpw), np.in1d(self.bins, ipw)))]
        else:
            self.opw = np.array(opw)
            if self.mpw is None:
                self.mpw = self.bins[np.logical_not(np.in1d(self.bins, opw))]

        if self.mpw is None and self.ipw is None and self.opw is None:
            self.mpw = np.arange(self.nbins)

        if align:
            if align != 0:
                self.data = np.roll(self.data, align)
                #prepare=True #? #keep this for 1937?
            #self.shiftit(align, save=True)

        if prepare: #change this for jitter (prepare set to False here)
            self.interpulse_align()
            #self.normalize() #do not do this

        self.period = period

        self.null = False
        if np.all(self.data == self.data[0]) or np.all(np.isnan(self.data)):
            self.null = True

    def interpulse_align(self):
        """
        Align the pulse such that the main pulse is at phase=0.25
        and the interpulse is at phase = 0.75
        """
        self.data = np.roll(u.center_max(self.data), -len(self.data)//4)
        return self

    def center_align(self):
        """
        Align the pulse such that the peak is in the center
        """
        self.data = u.center_max(self.data)
        return self

    def normalize(self, area=False):
        """
        Normalize the pulse so that the peak has a value of 1.0
        """
        if area:
            self.data = u.normalize_area(self.data)
        else:
            minimum = np.mean(self.getOffpulse())
            self.data = u.normalize(self.data, minimum=minimum)
        return self

    def getFWHM(self, simple=False, timeunits=True):
        """
        Get the full width at half maximum of the main component of the pulse
        """
        #remove baseline? what if no offpulse window?
        dbin = u.FWHM(self.data, notcentered=True)#, window=800)
        factor = 1
        if timeunits and self.getPeriod() is not None:
            factor = self.getPeriod()/self.getNbins()
        return factor*dbin

    def getFW(self, value=0.5, simple=False, timeunits=True):
        """
        Get the full width at some value (e.g., 0.5 = FWHM) of the main component of the pulse
        """
        #remove baseline? what if no offpulse window?
        dbin = u.FW(self.data, value=value, notcentered=True)#, window=800)
        factor = 1
        if timeunits and self.getPeriod() is not None:
            factor = self.getPeriod()/self.getNbins()
        return factor*dbin

    def getWeff(self, fourier=False, sumonly=False, timeunits=True):
        """
        Calculate the effective width of the pulse
        """
        if not timeunits or self.getPeriod() is None:
            return None
        P = self.getPeriod()
        N = self.getNbins()
        U = u.normalize(self.data, simple=True) #remove baseline?

        tot = np.sum(np.power(U[1:]-U[:-1], 2))
        if sumonly:
            return tot
        self.weff = P/np.sqrt(N*tot)
        return self.weff

    def getUscale(self):
        """
        Following Lam et al. 2018a Optimal Frequencies approach, equation 11.
        """
        Uobsbar = np.mean(self.data)
        return 1.0/Uobsbar

    def getSN(self):
        """
        Calculate a very crude S/N
        """
        return np.max(self.data)/self.getOffpulseNoise()

    def remove_baseline(self, save=True):
        """
        Subtract the baseline of the pulse so that it will be set to zero mean
        """
        if self.opw is None:
            print("No Offpulse") #do this?
            return
        opmean = np.mean(self.getOffpulse())
        if save:
            self.data = self.data - opmean
            return self
        return self.data - opmean
    removeBaseline = remove_baseline

    def calcOffpulseWindow(self, windowsize=None):
        # Find minimum in the area
        if windowsize is None:
            windowsize = self.nbins/8

        integral = np.zeros_like(self.data)
        for i in self.bins:
            win = np.arange(i-windowsize//2, i+windowsize//2) % self.nbins
            integral[i] = np.trapz(self.data[win])
        minind = np.argmin(integral)
        self.opw = np.arange(minind-windowsize//2, minind+windowsize//2+1)
        self.opw = self.opw % self.nbins
        self.mpw = self.bins[np.logical_not(np.in1d(self.bins, self.opw))]
        return self.opw

    ### Get each of the pulse components, if they exist
    def getMainpulse(self):
        if self.mpw is None:
            return None
        return self.data[self.mpw]

    def getInterpulse(self):
        if self.ipw is None:
            return None
        return self.data[self.ipw]

    def getOffpulse(self):
        if self.opw is None:
            return None
        return self.data[self.opw]

    def getAllpulse(self):
        return self.getMainpulse(), self.getInterpulse(), self.getOffpulse()

    def getPulse(self, ind=None):
        if ind is None:
            return self.data
        else:
            return self.data[ind]
    getData = getPulse

    ### Get each of the auto-correlation functions of the pulse components, if they exist
    def getMainpulseACF(self):
        mp = self.getMainpulse()
        return u.acf(mp, var=False, norm_by_tau=True)

    def getInterpulseACF(self):
        if self.ipw is None:
            return None
        ip = self.getInterpulse()
        return u.acf(ip, var=False, norm_by_tau=True)

    def getOffpulseACF(self):
        if self.opw is None:
            return None
        op = self.getOffpulse()
        return u.acf(op, var=False, norm_by_tau=True)

    def getAllACF(self):
        return self.getMainpulseACF(), self.getInterpulseACF(), self.getOffpulseACF()

    def getACF(self):
        return u.acf(self.getData(), var=False, norm_by_tau=True)

    def getOffpulseNoise(self, mean=False, full=False):
        """
        Return the off-pulse noise statistics
        """
        if self.opw is None:
            return None
        op = self.getOffpulse()
        if mean:
            return np.mean(op)
        if full:
            return np.mean(op), np.std(op)
        return np.std(op)

    def getOffpulseZCT(self):
        """
        Perform a zero-crossing test of the offpulse noise
        """
        return u.zct(self.getOffpulse(), full=True, meansub=True)

    # todo: include a goodness-of-fit flag (gof) as a measure of the residuals.
    def fitPulse(self, template, fixedphase=False, rms_baseline=None):
        """
        Returns tauccf, tauhat, bhat, sigma_Tau, sigma_b, snr, rho
        """
        if isinstance(template, SinglePulse):
            template = template.data
        if self.null or len(template) != self.getNbins():
            return None
        if rms_baseline is None:
            self.remove_baseline()
        if fixedphase: #just return S/N
            p0 = [np.max(self.data)]
            #conversion to np.float64 fixes bug with Jacobian inversion
            p1, cov, infodict, mesg, ier = \
            optimize.leastsq(lambda p, x, y: np.abs(p[0])*x - y, p0[:],
                             args=(np.asarray(template, np.float64),
                                   np.asarray(self.data, np.float64)),
                             full_output=True)
            noise = self.getOffpulseNoise()
            return np.abs(p1[0])/noise#, np.sqrt(cov[0][0])/noise
        if self.opw is None:
            if rms_baseline is not None:
                try:
                    return get_toa(template, self.data, rms_baseline)
                except:
                    print(self.data)
                    plt.plot(self.data)
                    plt.show()
                    raise SystemExit
            return get_toa(template, self.data, 1)
        try:
            #occasional failure at very low S/N
            return get_toa(template, self.data, self.getOffpulseNoise())
        except:
            return None

    #define this so a positive shift is forward
    # Implement shifts of mpw, opw?
    def shiftit(self, shift, save=False):
        """
        Shift the pulse by some phase.
        """
        x = u.shiftit(self.data, -1*shift)
        if save:
            self.data = x
            return self
        return x

    def spline_smoothing(self, sigma=None, lam=None, **kwargs):
        """ Cubic Spline Interplation
        sigma: phase bin error bars
        lam: (0, 1], 1 = maximize fitting through control points (knots),
                     0 = minimize curvature of spline
        """

        tdata = self.bins
        ydata = u.normalize(self.data, simple=True)
        N = len(ydata)

        noise = self.getOffpulseNoise()
        if lam is None or (lam > 1 or lam <= 0):
            lam = 1-noise**2
        mu = 2*float(1-lam)/(3*lam)

        ### Define knot locations

        # Rotate the pulse so the peak is at the edges of the spline
        shift = -np.argmax(ydata)
        yshift = np.roll(ydata, shift)

        # Add periodic point
        tdata = np.concatenate((tdata, [tdata[-1] + np.diff(tdata)[0]]))
        yshift = np.concatenate((yshift, [yshift[0]]))

        knots = u.subdivide(tdata, yshift, noise, **kwargs)
        knots = np.array(np.sort(knots), dtype=np.int)
        knots = np.concatenate(([0], knots, [N])) #Add endpoints

        setsigma = False
        if sigma is None:
            setsigma = True

        #for passnum in range(2):
        while True:
            Nknots = len(knots)
            Narcs = Nknots-1
            t = np.array(tdata[knots], dtype=np.float)

            # Determine the knot y-values.
            y = np.zeros_like(t)
            y[0] = yshift[0]
            y[-1] = yshift[-1]
            for i in range(1, len(knots)-1):
                knotL = knots[i-1]
                knotR = knots[i+1]
                dt = tdata[knotL:knotR]
                dy = yshift[knotL:knotR]
                p = np.polyfit(dt, dy, 3) #Fit a preliminary cubic over the data to place the point.
                f = np.poly1d(p)
                y[i] = f(tdata[knots[i]])

            if setsigma:
                sigma = np.ones(len(y), dtype=np.float)
            Sigma = np.diag(sigma[:-1]) #matrix

            # Smoothing with Cubic Splines by D.S.G. Pollock 1999
            h = t[1:]-t[:-1]
            r = 3.0/h

            f = np.zeros_like(h)
            p = np.zeros_like(h)
            #q = np.zeros_like(h)

            p[0] = 2*(h[0] + h[-1])
            #q[0] = 3*(y[1] - y[0])/h[0] - 3*(y[-1] - y[-2])/h[-1] #note the indices
            f[0] = -(r[-1]+r[0])
            for i in range(1, Narcs):
                p[i] = 2*(h[i] + h[i-1])
                #q[i] = 3*(y[i+1] - y[i])/h[i] - 3*(y[i] - y[i-1])/h[i-1]
                f[i] = -(r[i-1]+r[i])

            # Build projection matrices
            R = np.zeros((Narcs, Narcs))
            Qp = np.zeros((Narcs, Narcs))

            for i in range(Narcs):
                #for j in range(Narcs):
                #    if i == j:
                R[i, i] = p[i]
                Qp[i, i] = f[i]
                if i != Narcs -1:
                    R[i+1, i] = h[i]
                    R[i, i+1] = h[i]
                    Qp[i+1, i] = r[i]
                    Qp[i, i+1] = r[i]
            R[0, -1] = h[-1]
            R[-1, 0] = h[-1]
            Qp[0, -1] = r[-1]
            Qp[-1, 0] = r[-1]
            Q = np.transpose(Qp)

            A = mu*np.dot(np.dot(Qp, Sigma), Q) + R

            b = np.linalg.solve(A, np.dot(Qp, y[:-1]))
            d = y[:-1] - mu*np.dot(np.dot(Sigma, Q), b)
            a = np.zeros(Narcs)
            c = np.zeros(Narcs)

            i = Narcs-1
            a[i] = (b[0] - b[i])/(3*h[i])
            for i in range(Narcs-1):
                a[i] = (b[i+1] - b[i])/(3*h[i])

            i = Narcs-1
            c[i] = (d[0] - d[i])/h[i] - a[i]*h[i]**2 - b[i]*h[i]
            for i in range(Narcs-1):
                c[i] = (d[i+1] - d[i])/h[i] - a[i]*h[i]**2 - b[i]*h[i]

            # Build polynomials
            S = []
            for i in range(Narcs):
                S.append(np.poly1d([a[i], b[i], c[i], d[i]]))

            ytemp = np.zeros_like(yshift)
            for i in range(Narcs):
                ts = np.arange(t[i], t[i+1])
                hs = ts-t[i]
                yS = S[i](hs)
                ytemp[int(t[i]):int(t[i+1])] = yS

            ytemp[-1] = ytemp[0]
            resids = yshift-ytemp
            #print resids, noise
            rms_resids = u.RMS(resids)
            #inds = np.where(np.abs(resids)>4*rms_resids)[0]

            #require the intensity to be "significant", and the residuals
            inds = np.where(np.logical_and(yshift > 3*noise, np.abs(resids) > 4*rms_resids))[0]
            if len(inds) == 0:
                break

            #newinds = np.sort(np.abs(resids))
            newinds = np.argsort(np.abs(resids))
            #print np.argmax(np.abs(resids))
            #print newinds
            #raise SystemExit
            #newind = np.argmax(np.abs(resids))

            newind = newinds[-1]
            i = 1
            addknot = True
            while newind in knots and np.any(np.abs(knots-newind) <= 4):
                if i == N:
                    addknot = False
                    break
                i += 1
                newind = newinds[-i]

            '''
            for newind in newinds:
                if newind in knots:
                    continue
                elif np.all(np.abs(newind-knots)<=1):
                    continue
                print newind
                break
            '''
            if addknot:
                #print newind
                knots = np.sort(np.concatenate((knots, [newind])))
            else:
                break

        resids = yshift-ytemp
        rms_resids = u.RMS(resids)

        tdata = tdata[:-1]
        #yshift = yshift[:-1]
        ytemp = ytemp[:-1]

        #yshift = np.roll(yshift, -shift)
        ytemp = np.roll(ytemp, -shift)
        ytemp /= np.max(ytemp)

        return ytemp
    splineSmoothing = spline_smoothing

    def component_fitting(self, mode='gaussian', nmax=10, full=False,
                          minamp=None, alpha=0.05, allownegative=False,
                          verbose=False, save=False):
        '''
        Fitting to phases is much more numerically stable for von mises function
        '''

        fitter = lambda x, y, n: u.fit_components(x, y, mode, n, allownegative=allownegative)

        if minamp is not None:
            MAX = np.max(self.data)*minamp
        N = self.getNbins()

        # Fit first component
        n = 1
        nparamA = 3
        fitfunc, errfunc, pfit, perr, s_sq = fitter(self.phases, self.data, n)
        residsA = self.data - fitfunc(pfit, self.phases)
        RSS_funcA = np.sum(residsA**2)

        def doreturn(save=False):
            if save:
                self.data = fitfunc(pfit, self.phases)
            if full:
                return fitfunc(pfit, self.phases), pfit, n
            return fitfunc(pfit, self.phases)

        for n in range(2, nmax+1):
            if verbose:
                print("Fitting component: %i"%n)
            nparamB = 3*n
            fitfunc, errfunc, pfit, perr, s_sq = fitter(self.phases, self.data, n)
            residsB = self.data - fitfunc(pfit, self.phases)
            RSS_funcB = np.sum(residsB**2)

            if minamp is not None and np.all(residsB < MAX):
                return doreturn(save=save)

            # F-test
            F = ((RSS_funcA-RSS_funcB)/(nparamB-nparamA))/(RSS_funcB/(N-nparamB-1))

            p_value = stats.f.sf(F, nparamB-nparamA, N-nparamB-1) #cdf?
            #print F, RSS_funcA, RSS_funcB, nparamA, nparamB, p_value
            if p_value > alpha: # if p_value < alpha,  then B is significant, so keep going
            # Replace old values
                return doreturn(save=save)
            nparamA = nparamB
            residsA = residsB
            RSS_funcA = RSS_funcB
        return doreturn(save=save)

        '''
        n = 1
        chisq = 10000.0


        while True:
            #print s_sq
            if s_sq < chisq:
                chisq = s_sq
            else:
                break
            if n == nmax:
                break
            n += 1

        #n -= 1
        if n <= 0:
            n = 1
        fitfunc, errfunc, pfit, perr, s_sq = fitter(self.phases, self.data, n)
        if full:
            return fitfunc(pfit, self.phases), n
        return fitfunc(pfit, self.phases)
        '''
        #return fitfunc, errfunc, pfit, perr, s_sq, n
    componentFitting = component_fitting

    def gaussian_smoothing(self, **kwargs):
        """ Smooth the data with Gaussian functions """
        return self.component_fitting(mode='gaussian', **kwargs)
    def vonmises_smoothing(self, **kwargs):
        """ Smooth the data with von Mises functions """
        return self.component_fitting(mode='vonmises', **kwargs)
    vonMises_smoothing = vonmises_smoothing

    def savgol_smoothing(self, window_length=11, polyorder=5):
        """
        Savitzky-Golay filter smoothing
        """
        return signal.savgol_filter(self.data, window_length, polyorder)

    def smooth(self, mode='vonmises', sigma=None, lam=None,
               window_length=11, polyorder=5, **kwargs):
        """
        Generic smoothing caller
        """
        if mode == "vonmises":
            return self.vonmises_smoothing(**kwargs)
        elif mode == "gaussian":
            return self.gaussian_smoothing(**kwargs)
        elif mode == "spline":
            return self.spline_smoothing(sigma=sigma, lam=lam, **kwargs)
        elif mode == "savgol":
            return self.savgol_smoothing(window_length=window_length, polyorder=polyorder)
        return None

    def estimateScatteringTimescale(self, searchtauds=None, ntauds=25,
                                    name=None, fourier=False, **kwargs):
        if searchtauds is None:
            fwhm = self.getFWHM(timeunits=False) #in bin units
            tauds = np.linspace(fwhm/4, fwhm, ntauds)

            tauds = np.linspace(2, self.getNbins()/4, ntauds)
            tauds = np.linspace(2, fwhm, ntauds)
        else:
            tauds = np.copy(searchtauds)

        bins = np.array(self.bins, dtype=np.float)

        N_fs = np.zeros_like(tauds)
        sigma_offcs = np.zeros_like(tauds)
        Gammas = np.zeros_like(tauds)
        f_rs = np.zeros_like(tauds)
        for i, taud in enumerate(tauds):
            #print i, taud
            if fourier:
                Dy, C, N_f, sigma_offc, Gamma, f_r = \
                u.pbf_fourier(bins, self.data, taud=taud, opw=self.opw, **kwargs)
            else:
                Dy, C, N_f, sigma_offc, Gamma, f_r = \
                u.pbf_clean(bins, self.data, taud=taud, opw=self.opw, **kwargs)
            N_fs[i] = N_f
            sigma_offcs[i] = sigma_offc
            Gammas[i] = Gamma
            f_rs[i] = f_r

        f_cs = (Gammas+f_rs)/2.0

        fig = plt.figure(figsize=(8, 12))
        ax = fig.add_subplot(511)
        ax.plot(tauds, N_fs)
        ax.set_ylabel(r'$N_f/N_{\rm tot}$')
        ax = fig.add_subplot(512)
        ax.plot(tauds, sigma_offcs)
        ax.set_ylabel(r'$\sigma_{\rm offc}/\sigma_{\rm off}$')
        ax = fig.add_subplot(513)
        ax.plot(tauds, Gammas)
        ax.set_ylabel(r'$\Gamma$')
        ax = fig.add_subplot(514)
        ax.plot(tauds, f_rs)
        ax.set_ylabel(r'$f_r$')
        ax = fig.add_subplot(515)
        ax.plot(tauds, f_cs)
        ax.set_xlabel(r'$\tau_{\rm d}$')
        ax.set_ylabel(r'$f_c$')
        if name is None:
            plt.show()
        else:
            np.savez("cleandata/%s_clean_data.npz"%name,
                     tauds=(tauds*self.getTbin()*1000), N_fs=N_fs,
                     sigma_offcs=sigma_offcs, Gammas=Gammas, f_rs=f_rs,
                     f_cs=f_cs)
            plt.savefig("cleanimages/%s_clean_stats.png"%name)
            plt.close()

        ind = np.argmin(f_cs)
        #ind = np.argmin(Gammas)
        #print tauds[ind], tauds[ind]*self.getTbin()
        if fourier:
            Dy, C, N_f, sigma_offc, Gamma, f_r = \
            u.pbf_fourier(bins, self.data, taud=tauds[ind], opw=self.opw, **kwargs)
        else:
            Dy, C, N_f, sigma_offc, Gamma, f_r = \
            u.pbf_clean(bins, self.data, taud=tauds[ind], opw=self.opw, **kwargs)
        #Dy, C, N_f, sigma_offc, Gamma, f_r=clean(t, y, taud=10)
        #print "gamma", Gamma

        fig = plt.figure()
        ax = fig.add_subplot(111)
        #ax.plot(self.bins, self.data/np.max(self.data), 'b')
        ax.plot(bins, Dy/np.max(self.data), '0.50')
        if fourier:
            ax.plot(bins, C/np.max(self.data), 'r')
        else:
            ax.plot(bins, C, 'r')
        ax.plot(bins, self.data/np.max(self.data), 'k')
        ax.set_xlim(0, bins[-1])
        ax.text(0.7, 0.8,
                r'$\tau_d$ = %0.2f ms'%(tauds[ind]*self.getTbin()*1000),
                transform=ax.transAxes, fontsize=14)
        if name is None:
            plt.show()
        else:
            plt.savefig("cleanimages/%s_clean.png"%name)
            plt.close()

    def getPeriod(self):
        """ Return the pulse period """
        return self.period

    def getNbins(self):
        """ Return the number of bins """
        return len(self.data)
    getNbin = getNbins

    def getTbin(self):
        """ Return the time unit of one phase bin """
        if self.getPeriod() is not None:
            return self.getPeriod()/self.getNbins()

    def plot(self, show=True):
        """
        Simple plot
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.bins, self.data, 'k')
        ax.set_xlim(self.bins[0], self.bins[-1])
        dy = np.ptp(self.data)
        ax.set_ylim(np.min(self.data)-0.05*dy, np.max(self.data)+0.05*dy)
        ax.set_xlabel("Phase Bins")
        ax.set_ylabel("Intensity")
        if show:
            plt.show()

    def plot_windows(self, show=True):
        """
        Diagnostic plot of the main-, inter-, and off-pulse regions
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(self.bins, self.data, 'k')
        MIN = np.min(self.data)

        diffs = np.abs(np.diff(self.mpw))
        if np.any(diffs > 1):
            M = np.argmax(diffs) + 1
            ax.fill_between(self.mpw[:M], np.zeros_like(self.mpw[:M])+MIN,
                            self.data[self.mpw[:M]], facecolor='b', alpha=0.5)
            ax.fill_between(self.mpw[M:], np.zeros_like(self.mpw[M:])+MIN,
                            self.data[self.mpw[M:]], facecolor='b', alpha=0.5)
        else:
            ax.fill_between(self.mpw, np.zeros_like(self.mpw)+MIN,
                            self.data[self.mpw], facecolor='b', alpha=0.5)



        if self.ipw != None:
            diffs = np.abs(np.diff(self.ipw))
            if np.any(diffs > 1):
                M = np.argmax(diffs) + 1
                ax.fill_between(self.ipw[:M], np.zeros_like(self.ipw[:M])+MIN,
                                self.data[self.ipw[:M]], facecolor='g', alpha=0.5)
                ax.fill_between(self.ipw[M:], np.zeros_like(self.ipw[M:])+MIN,
                                self.data[self.ipw[M:]], facecolor='g', alpha=0.5)
            else:
                ax.fill_between(self.ipw, np.zeros_like(self.ipw)+MIN,
                                self.data[svelf.ipw], facecolor='g', alpha=0.5)


        ax.set_xlim(self.bins[0], self.bins[-1])
        dy = np.ptp(self.data)
        ax.set_ylim(np.min(self.data)-0.05*dy, np.max(self.data)+0.05*dy)
        ax.set_xlabel("Phase Bins")
        ax.set_ylabel("Intensity")
        if show:
            plt.show()

 #Helper functions for fitPulse
# tauccf, tauhat, bhat, sigma_Tau, sigma_b, snr, rho
def get_fitPulse_TOA(retval):
    return retval[1]
get_tauhat = get_fitPulse_TOA

def get_fitPulse_ScaleFactor(retval):
    return retval[2]
get_bhat = get_fitPulse_ScaleFactor

def get_fitPulse_TOAerror(retval):
    return retval[3]
get_sigma_tau = get_fitPulse_TOAerror

def get_fitPulse_ScaleFactorerr(retval):
    return retval[4]
get_sigma_b = get_fitPulse_ScaleFactorerr

def get_fitPulse_SN(retval):
    return retval[5]
get_snr = get_fitPulse_SN
