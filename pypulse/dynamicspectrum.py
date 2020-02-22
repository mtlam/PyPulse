'''
Michael Lam 2015

To do: grid: set clim such that 0 is white, not blue,mask data,add zorder
'''

import sys
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import fftconvolve
import scipy.optimize as optimize
import scipy.interpolate as interpolate
import pypulse.utils as u
import pypulse.functionfit as ffit

if sys.version_info.major == 2:
    fmap = map
elif sys.version_info.major == 3:
    fmap = lambda x, *args: list(map(x, *args))
    xrange = range


class DynamicSpectrum(object):
    def __init__(self, data, offdata=None, errdata=None, mask=None,
                 F=None, T=None, extras=None, verbose=True,
                 Funit="arb.", Tunit="arb."):
        self.verbose = verbose
        if isinstance(data, str):
            name = data
            self.load(data)
            return
#            self.extras = dict()
#            self.extras['filename'] = data
        else:
            self.data = data
        #check if 1d array
        if len(self.shape()) >= 1: #why==?????

            self.offdata = offdata
            self.errdata = errdata
            self.mask = mask
            self.F = F#these are the edges
            self.T = T
            self.Fcenter = None
            self.Tcenter = None
            self.dF = None
            self.dT = None
            if F is not None:
                d = np.diff(self.F)
                self.Fcenter = d/2.0 + self.F[:-1] #is this correct?
                self.dF = np.mean(d)
            else:
                self.F = np.arange(self.shape()[0])
            if T is not None:
                d = np.diff(self.T)
                self.Tcenter = d/2.0 + self.T[:-1]
                self.dT = np.median(d)
            else:
                self.T = np.arange(self.shape()[1])
            self.Funit = Funit
            self.Tunit = Tunit
            if extras is None:
                self.extras = dict()
            else:
                self.extras = extras

        # Pre-define variables
        self.baseline_removed = False
        self.acf = None
        self.acfT = None
        self.acfF = None
        self.ss = None
        self.ssconjT = None
        self.ssconjF = None

    def getValue(self, f, t, df=1, dt=1, err=False, index=False):
        '''
        Returns value of dynamic spectrum
        if index==False, f,t are values to search for
        '''
        if index:# or self.F is None or self.T is None:
            if err:
                return self.errdata[f, t]
            return self.data[f, t]
        else:
            indsF = np.where(np.abs(self.Fcenter-f) <= df)[0]
            indsT = np.where(np.abs(self.Tcenter-t) <= dt)[0]
            if len(indsF) == 0 or len(indsT) == 0:
                return None
            if err:
                data = self.errdata
            else:
                data = self.data
            total = 0
            N = 0
            for indF in indsF:
                for indT in indsT:
                    total += data[indF, indT]
                    N += 1
            return total/float(N)

    def remove_baseline(self, function="gaussian", redo=False):
        """
        Attempts to remove the baseline amplitude from the dynamic spectrum
        """
        if not redo and self.baseline_removed:
            return self
        flatdata = self.data.flatten()
        if np.all(flatdata == np.median(flatdata)):#[0]:
            self.data -= np.median(flatdata)
            self.baseline_removed = True
            return self
        #~100 divisions, but bins to an even power of 10
        interval = np.power(10, np.floor(np.log10(np.ptp(flatdata/100))))
        center, hist = u.histogram(flatdata, interval=interval)

        if function == "gaussian":
            p1, err = ffit.gaussianfit(center, hist)
            y = ffit.funcgaussian(p1, center)
            peak = center[np.argmax(y)]
        elif function == "simple_DISS":
            area = np.trapz(hist, x=center)
            shift = -np.min(center)+1.0
            x = center + shift
            y = np.array(hist, dtype=np.float)/area
            p1, err = ffit.simpleDISSpdffit(x, y)
            y1 = ffit.funcsimpleDISSpdf(p1, x)*area
            peak = center[np.argmax(y1)]
        else:
            peak = 0.0

        self.data -= peak
        self.baseline_removed = True
        return self

    def stretch(self, nuref=None, index=(-22.0/5), save=True):
        # Modified from code by Glenn Jones, https://github.com/gitj/dynISM/blob/master/dynspec.py
        F = self.F
        if nuref is None:
            nuref = np.sqrt(np.max(F)*np.min(F))

        f2 = np.cumsum((F/nuref)**index) #stretched axis (non-interpolated)
        nfout = int(np.floor(np.max(f2))) #number of out frequencies
        retval = np.zeros((nfout, self.shape()[1]))
        x = np.arange(nfout)
        data = self.getData()
        for i in range(self.shape()[1]): #iterate over each subintegration
            retval[:, i] = np.interp(x, f2, data[:, i])

        fout = np.interp(x, f2, F) #stretched axis (interpolated)
        if save:
            self.data = retval
            self.offdata = None #can do the same thing as above
            self.errdata = None #need to figure this out
            self.mask = None#need to figure this out
            self.F = fout

            # this is not correct
            d = np.diff(self.F)
            self.Fcenter = d/2.0 + self.F[:-1] #is this correct?
            self.dF = np.mean(d)

        return retval, nuref, fout

    def acf2d(self, remove_baseline=True, speed='fast', mode='full', full_output=False):
        """
        Calculate the two-dimensional auto-correlation function of the dynamic spectrum
        """
        data = self.getData(remove_baseline=remove_baseline)


        # Have if statement to apply mask: set ones in the norm to 0.0
        ones = np.ones(np.shape(data))
        norm = fftconvolve(ones, np.flipud(np.fliplr(ones)), mode=mode)
        acf = fftconvolve(data, np.flipud(np.fliplr(data)), mode=mode)/norm


        # Replace the central noise spike with that of the next highest of its neighbors
        acfshape = np.shape(acf)
        centerrind = acfshape[0]//2
        centercind = acfshape[1]//2
        acf[centerrind, centercind] = 0.0
        acf[centerrind, centercind] = \
        np.max(acf[centerrind-1:centerrind+2, centercind-1:centercind+2])

        self.acf = acf
        self.acfT = np.concatenate((-1*self.T[:-1][::-1], self.T[1:-1]))
        self.acfF = np.concatenate((-1*self.F[:-1][::-1], self.F[1:-1]))
        if full_output:
            return self.acfT, self.acfF, acf
        return acf

        #return u.acf2d(self.data, speed=speed, mode=mode) #do more here

    def secondary_spectrum(self, remove_baseline=True, log=False, full_output=False):
        data = self.getData(remove_baseline=remove_baseline)

        ss = np.abs(np.fft.fftshift(np.fft.fft2(data)))**2

        if log:
            ss = np.log10(ss)

        self.ss = ss
        self.ssconjT = np.fft.fftshift(np.fft.fftfreq(len(self.T), d=self.dT))
        self.ssconjF = np.fft.fftshift(np.fft.fftfreq(len(self.F), d=self.dF))
        if full_output:
            return self.ssconjT, self.ssconjF, ss
        return ss

    # allow for simple 1D fitting
    def scintillation_parameters(self, plotbound=1.0, maxr=None, maxc=None,
                                 savefig=None, show=True, full_output=False,
                                 simple=False, eta=0.2, cmap=cm.binary,
                                 finitescintleerrors=True, diagnostic=False):
        if self.acf is None:
            self.acf2d()
        if self.dT is None:
            dT = 1
        else:
            dT = self.dT
        if self.dF is None:
            dF = 1
        else:
            dF = self.dF

        acfshape = np.shape(self.acf)
        centerrind = acfshape[0]//2
        centercind = acfshape[1]//2

        if simple: # Do 1D slices
            NF = len(self.F)
            Faxis = (np.arange(-(NF-1), NF, dtype=np.float)*np.abs(dF)) #why abs?
            NT = len(self.T)
            Taxis = (np.arange(-(NT-1), NT, dtype=np.float)*np.abs(dT))[1:-1] #???

            
            pout, errs = ffit.gaussianfit(Taxis[NT//2:3*NT//2], self.acf[centerrind, NT//2:3*NT//2], baseline=True)
            ft = interpolate.interp1d(Taxis, ffit.funcgaussian(pout, Taxis, baseline=True)-(pout[3]+pout[0]/np.e))
            try:
                delta_t_d = optimize.brentq(ft, 0, Taxis[-1])
            except ValueError:
                delta_t_d = np.nan

            pout, errs = ffit.gaussianfit(Faxis[NF//2:3*NF//2], self.acf[NF//2:3*NF//2, centercind], baseline=True)
            fnu = interpolate.interp1d(Faxis, ffit.funcgaussian(pout, Faxis, baseline=True)-(pout[3]+pout[0]/2))
            try:
                delta_nu_d = optimize.brentq(fnu, 0, Faxis[-1])
            except ValueError:
                delta_nu_d = np.nan


            #following Glenn's code and Cordes 1986 (Space Velocities...)
            # Errors from finite scintle effect:
            bw = self.getBandwidth()
            T = self.getTspan()
            if delta_t_d == 0.0:
                N_d = (1+eta * bw/delta_nu_d)
            elif delta_nu_d == 0.0:
                N_d = (1+eta*T/delta_t_d)
            else:
                N_d = (1+eta * bw/delta_nu_d) * (1+eta*T/delta_t_d)
            fse_nu_d = delta_nu_d/(2*np.log(2)*np.sqrt(N_d)) #log because of FWHM?
            fse_t_d = delta_t_d/(2*np.sqrt(N_d))

            err_nu_d = fse_nu_d
            err_t_d = fse_t_d #need to add in fitting errors

            if show or savefig is not None:
                fig = plt.figure()
                ax = fig.add_subplot(211)
                ax.plot(Taxis, self.acf[centerrind, :])
                ax.plot(Taxis, ft(Taxis))

                ax = fig.add_subplot(212)
                ax.plot(Faxis, self.acf[:, centercind])
                ax.plot(Faxis, fnu(Faxis))

                if savefig is not None:
                    plt.savefig(savefig)
                if show:
                    plt.show()
            
            if full_output:
                return delta_t_d, err_t_d, delta_nu_d, err_nu_d
            return delta_t_d, delta_nu_d

        # Look for the central peak in the ACF
        MIN = np.min(self.acf)
        if MIN < 0: #The min value is approximately from a gaussian distribution
            MIN = np.abs(MIN)
        else:
            #center, hist = u.histogram(acf.flatten(), interval=0.001) #relies on 0.001
            #MIN = center[np.argmax(hist)]
            MIN = u.RMS(self.acf.flatten())
        if maxr is None:
            rslice = self.acf[centerrind:, centercind]
            maxr = np.where(rslice <= MIN)[0][0]
            if maxr == 0:
                maxr = np.shape(self.acf)[0]//2
        if maxc is None:
            cslice = self.acf[centerrind, centercind:]
            maxc = np.where(cslice <= MIN)[0][0]
            if maxc == 0:
                maxc = np.shape(self.acf)[1]//2
                
        plotacf = self.acf[int(centerrind-plotbound*maxr+1):int(centerrind+plotbound*maxr),
                           int(centercind-plotbound*maxc+1):int(centercind+plotbound*maxc+1)]

        if diagnostic:
            print(np.shape(self.acf))
            print(centerrind-plotbound*maxr+1,centerrind+plotbound*maxr+1)
            print(centercind-plotbound*maxc+1,centercind+plotbound*maxc+1)
            u.imshow(plotacf)
            plt.show()
            return plotacf

        params, pcov = ffit.fitgaussian2d(plotacf) #pcov already takes into account s_sq issue
        SHAPE = np.shape(plotacf)

        fit = ffit.gaussian2d(*params)
        amplitude, center_x, center_y, width_x, width_y, rotation, baseline = params

        paramnames = ["amplitude", "center_x", "center_y", "width_x",
                      "width_y", "rotation", "baseline"]
        if pcov is not None:
            paramerrors = np.sqrt(np.diagonal(pcov))
        else:
            paramerrors = np.zeros_like(params)
        if self.verbose:
            for i, param in enumerate(params):
                print("%s: %0.2e+/-%0.2e"%(paramnames[i], param, paramerrors[i]))


        #Solve for scintillation parameters numerically

        try:
            delta_t_d = (optimize.brentq(lambda y: fit(SHAPE[0]//2, y)-baseline-amplitude/np.e,
                                         (SHAPE[1]-1)//2, SHAPE[1]*2)-(SHAPE[1]-1)//2)*dT #FWHM test
            if self.verbose:
                print("delta_t_d %0.3f %s"%(delta_t_d, self.Tunit))
        except ValueError:
            if self.verbose:
                print("ERROR in delta_t_d")
            delta_t_d = SHAPE[1]*dT
        if pcov is not None:
            err_t_d = paramerrors[3]*dT #assume no rotaton for now
        else:
            err_t_d = None

        try:
            delta_nu_d = (optimize.brentq(lambda x: fit(x, SHAPE[1]//2)-baseline-amplitude/2.0,
                                          (SHAPE[0]-1)//2, SHAPE[0])-(SHAPE[0]-1)//2)*dF
            if self.verbose:
                print("delta_nu_d %0.3f %s"%(delta_nu_d, self.Funit))
        except ValueError:
            if self.verbose:
                print("ERROR in delta_nu_d")
            delta_nu_d = SHAPE[0]*dF
        if pcov is not None:
            err_nu_d = paramerrors[4]*dF #assume no rotaton for now
        else:
            err_nu_d = None

        err_rot = paramerrors[5]

        #finite-scintle errors
        if finitescintleerrors:
            bw = self.getBandwidth()
            T = self.getTspan()
            if delta_t_d == 0.0:
                N_d = (1+eta * bw/delta_nu_d)
            elif delta_nu_d == 0.0:
                N_d = (1+eta*T/delta_t_d)
            else:
                N_d = (1+eta * bw/delta_nu_d) * (1+eta*T/delta_t_d)
            fse_nu_d = delta_nu_d/(2*np.log(2)*np.sqrt(N_d)) #log because of FWHM?
            fse_t_d = delta_t_d/(2*np.sqrt(N_d))

            fse_rot = rotation * np.sqrt((fse_nu_d/delta_nu_d)**2 + (fse_t_d/delta_t_d)**2)

            err_nu_d = np.sqrt(err_nu_d**2 + fse_nu_d**2)
            err_t_d = np.sqrt(err_t_d**2 + fse_t_d**2)
            err_rot = np.sqrt(err_rot **2 + fse_rot**2)

        if self.verbose:
            f = (dF/dT)*np.tan(rotation)
            df = (dF/dT)*np.cos(rotation)**2 * err_rot
            print("dnu/dt %0.3e+/-%0.3e %s/%s" % (f, df, self.Funit, self.Tunit))
            #((dF/dT)*np.tan(rotation))

        if show or savefig is not None:
            fig = plt.figure()
            ax = fig.add_subplot(211)
            u.imshow(self.data, cmap=cmap)
            ax = fig.add_subplot(212)

            u.imshow(plotacf, cmap=cmap)
            plt.colorbar()
            levels = (amplitude*np.array([1.0, 0.5, 1.0/np.e]))+baseline
            levels = (amplitude*np.array([0.5]))+baseline
            #print(levels)

            ax.contour(fit(*np.indices(plotacf.shape)), levels, colors='k')
            #ax.set_xlim(len(xs)-20, len(xs)+20)
            #ax.set_ylim(len(ys)-10, len(ys)+10)
            if savefig is not None:
                plt.savefig(savefig)
            if show:
                plt.show()
        if full_output:
            return delta_t_d, err_t_d, delta_nu_d, err_nu_d, rotation, err_rot
        return delta_t_d, delta_nu_d, rotation

    def imshow(self, err=False, cbar=False, ax=None, show=True, border=False,
               zorder=0, cmap=cm.binary, alpha=True, cdf=True, savefig=None,
               acf=False, ss=False, extent=None, log=False, xlim=None,
               ylim=None):
        """
        Basic plotting of the dynamic spectrum
        """
        if acf:
            if self.acf is None:
                self.acf2d()
            data = self.acf
        elif ss:
            if self.ss is None:
                self.secondary_spectrum(log=log)
            data = self.ss
        elif err:
            data = self.errdata
        else:
            data = self.getData()
        if extent is None:
            if acf:
                T = self.acfT
                F = self.acfF
            elif ss:
                T = self.ssconjT
                F = self.ssconjF
            else:
                if self.Tcenter is None:
                    T = self.T
                else:
                    T = self.Tcenter
                if self.Fcenter is None:
                    F = self.F
                else:
                    F = self.Fcenter

        if log and not ss:
            data = np.log10(data)

        if alpha and not (acf or ss): #or just ignore?
            cmap.set_bad(alpha=0.0)

            for i in range(len(data)):
                for j in range(len(data[0])):
                    if data[i, j] <= 0.0:# or self.errdata[i][j]>3*sigma:
                        data[i, j] = np.nan

        if cdf:
            xcdf, ycdf = u.ecdf(data.flatten())
            low, high = u.likelihood_evaluator(xcdf, ycdf, cdf=True, values=[0.01, 0.99])
            for i in range(len(data)):
                for j in range(len(data[0])):
                    if data[i, j] <= low:
                        data[i, j] = low
                    elif data[i, j] >= high:
                        data[i, j] = high
        if extent is None:
            minT = T[0]
            maxT = T[-1]
            minF = F[0]
            maxF = F[-1]
            extent = [minT, maxT, minF, maxF]
        #return np.shape(data)
        #raise SystemExit
#        print inds
#        raise SystemExit
#        spec[inds] = np.nan
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        cax = u.imshow(data, ax=ax, extent=extent, cmap=cmap, zorder=zorder)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        #border here?
        if border:# and self.extras['name']!='EFF I':
            plt.plot([T[0], T[-1]], [F[0], F[0]], '0.50', zorder=zorder+0.1)
            plt.plot([T[0], T[-1]], [F[-1], F[-1]], '0.50', zorder=zorder+0.1)
            plt.plot([T[0], T[0]], [F[0], F[-1]], '0.50', zorder=zorder+0.1)
            plt.plot([T[-1], T[-1]], [F[0], F[-1]], '0.50', zorder=zorder+0.1)

        if acf:
            plt.xlabel('Time Lag (%s)'%self.Tunit)
            plt.ylabel('Frequency Lag (%s)'%self.Funit)
        elif ss:
            plt.xlabel('Conjugate Time (1/%s)'%self.Tunit)
            plt.ylabel('Conjugate Frequency (1/%s)'%self.Funit)
        else:
            plt.xlabel('Time (%s)'%self.Tunit)
            plt.ylabel('Frequency (%s)'%self.Funit)

        if cbar:
            plt.colorbar(cax)
        #im.set_clim(0.0001, None)
        if savefig is not None:
            plt.savefig(savefig)
        if show:
            plt.show()

        return ax

    def load(self, filename):
        """
        Load the dynamic spectrum from a .npz file
        """
        if self.verbose:
            print("Dynamic Spectrum: Loading from file: %s" % filename)
        x = np.load(filename, allow_pickle=True)
        for key in x.keys():
            val = eval("x['%s']"%key)
            if np.ndim(val) == 0 and isinstance(val, np.ndarray): #loading of non-array values (e.g.,  a None) needs to be handled this way
                exec("self.%s=x['%s'].item()"%(key, key))
            else:
                exec("self.%s=x['%s']"%(key, key))
        exec("self.extras = dict(%s)"%self.extras)
        #Convert array(None) to None
        if not hasattr(self, 'offdata') or (self.offdata is not None and len(np.shape(self.offdata)) == 0):
            self.offdata = None
        if not hasattr(self, 'errdata') or (self.errdata is not None and len(np.shape(self.errdata)) == 0):
            self.errdata = None
        if not hasattr(self, 'mask') or (self.mask is not None and len(np.shape(self.mask)) == 0):
            self.mask = None
        if not hasattr(self, 'acf') or (self.acf is not None and len(np.shape(self.acf)) == 0):
            self.acf = None
        if not hasattr(self, 'ss') or (self.ss is not None and len(np.shape(self.ss)) == 0):
            self.ss = None

        # Patch
        if not hasattr(self, 'baseline_removed'):
            self.baseline_removed = False
        for elem in ['acfT', 'acfF', 'ssconjT', 'ssconjF']:
            if not hasattr(self, elem):
                exec("self.%s = None"%elem)
        for elem in ['Tunit', 'Funit']:
            if not hasattr(self, elem):
                exec("self.%s = \"arb.\""%elem)

        x.close()
        return

    def save(self, filename):
        """
        Save the dynamic spectrum to a .npz file
        """
        if self.verbose:
            print("Dynamic Spectrum: Saving to file: %s" % filename)
        np.savez(filename, data=self.data, offdata=self.offdata,
                 errdata=self.errdata, mask=self.mask, F=self.F, T=self.T,
                 Fcenter=self.Fcenter, Tcenter=self.Tcenter,
                 Tunit=self.Tunit, Funit=self.Funit,
                 baseline_removed=self.baseline_removed, acf=self.acf,
                 acfT=self.acfT, acfF=self.acfF, ss=self.ss,
                 ssconjT=self.ssconjT, ssconjF=self.ssconjF,
                 extras=self.extras, dT=self.dT, dF=self.dF)
        return

    def savetxt(self, filename, acf=False, ss=False):
        # Need to check if acf or ss exist
        if acf:
            u.write2Dtxt(filename, self.acf, self.acfT, self.acfF)
            return
        elif ss:
            u.write2Dtxt(filename, self.ss, self.ssconjT, self.ssconjF)
            return
        else:
            u.write2Dtxt(filename, self.ds, self.T, self.F)

    # Must be in time order!
    def add(self, ds, axis='T'):
        """
        Concatenates another dynamic spectrum with this one
        """
        if axis == 'T':
            self.T = np.concatenate((self.T, ds.T))
            if len(np.shape(ds.data)) == 1:
                ds.data = np.reshape(ds.data, [len(ds.data), 1])
                if ds.offdata is not None:
                    ds.offdata = np.reshape(ds.offdata, [len(ds.offdata), 1])
                if ds.errdata is not None:
                    ds.errdata = np.reshape(ds.errdata, [len(ds.errdata), 1])
                if ds.mask is not None:
                    ds.mask = np.reshape(ds.mask, [len(ds.mask), 1])

            self.data = np.hstack((self.data, ds.data))
            #if statements
            if self.offdata is None and ds.offdata is None:
                self.offdata = None
            else:
                if self.offdata is None and ds.offdata is not None:
                    self.offdata = np.zeros(self.shape())
                elif self.offdata is not None and ds.offdata is None:
                    ds.offdata = np.zeros(np.shape(ds.data))
                self.offdata = np.hstack((self.offdata, ds.offdata))
            if self.errdata is None and ds.errdata is None:
                self.errdata = None
            else:
                if self.errdata is None and ds.errdata is not None:
                    self.errdata = np.zeros(self.shape())
                elif self.errdata is not None and ds.errdata is None:
                    ds.errdata = np.zeros(np.shape(ds.data))
                self.errdata = np.hstack((self.errdata, ds.errdata))
            if self.mask is None and ds.mask is None:
                self.mask = None
            else:
                if self.mask is None and ds.mask is not None:
                    self.mask = np.zeros(self.shape())
                elif self.mask is not None and ds.mask is None:
                    ds.mask = np.zeros(np.shape(ds.data))
                self.mask = np.hstack((self.mask, ds.mask))

            #Regenerate Tcenter?
            #Add extras together?

    def shape(self):
        """Return the current shape of the data array"""
        return np.shape(self.data)

    def getData(self, remove_baseline=True):
        """Returns the data array"""
        if remove_baseline:
            self.remove_baseline()
        return np.copy(self.data)

    def getACF(self, remove_baseline=True):
        if self.acf is None:
            return self.acf2d(remove_baseline=remove_baseline)
        return np.copy(self.acf)

    def getSS(self, remove_baseline=True, log=False):
        if self.ss is None:
            return self.secondary_spectrum(remove_baseline=remove_baseline, log=log)
        return np.copy(self.ss)

    def getBandwidth(self):
        return np.abs(self.F[-1]-self.F[0])

    def getTspan(self):
        return np.abs(self.T[-1]-self.T[0])

