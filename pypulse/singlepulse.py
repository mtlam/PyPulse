'''
Michael Lam 2015

Define interpulse alignment as putting the peak value at len/4. Interpulse will be roughly at 3*len/4

Figure out way to add/average SPs.
'''

import matplotlib.pyplot as plt
import numpy as np
import pypulse.utils as u
import scipy.optimize as optimize

import sys
get_toa = u.get_toa3 #try this one

#ACF=lambda p: np.correlate(p,p,"full") #no longer used


'''
data : data array
mpw : Main pulse window
ipw : Interpulse window
opw : off pulse window
prepare : Process data according to some metrics (deprecated)
align : Roll pulse by this amount
period : Set pulse period
windowsize: Override mpw,ipw,opw, define an offpulse window about the minimum of this size


'''




class SinglePulse:
    def __init__(self,data,mpw=None,ipw=None,opw=None,prepare=False,align=None,period=None,windowsize=None):
        if len(np.shape(data)) != 1:
            raise IndexError("SinglePulse received incorrect data shape")

        self.data=np.array(data)
        if mpw is not None:
            self.mpw = np.array(mpw)
        else:
            self.mpw = None
        if ipw is not None:
            self.ipw = np.array(ipw)
        else:
            self.ipw = None
        #Define off pulse
        self.nbins = len(data)
        self.bins=np.arange(self.nbins)


        if windowsize is not None:
            # Find minimum in the area
            integral = np.zeros_like(self.data)
            for i in self.bins:
                win = np.arange(i-windowsize//2,i+windowsize//2) % self.nbins
                integral[i] = np.trapz(self.data[win])
            minind = np.argmin(integral)
            self.opw = np.arange(minind-windowsize//2,minind+windowsize//2+1)
            self.opw = self.opw % self.nbins
            self.mpw = self.bins[np.logical_not(np.in1d(self.bins,self.opw))]

        elif opw is None:
            if self.mpw is None and self.ipw is None:
                self.opw = None #do not define any windows
            elif self.ipw is None:
                self.opw = self.bins[np.logical_not(np.in1d(self.bins,mpw))]
            elif self.mpw is None:
                self.opw = self.bins[np.logical_not(np.in1d(self.bins,ipw))]
            else:
                self.opw = self.bins[np.logical_not(np.logical_or(np.in1d(self.bins,mpw),np.in1d(self.bins,ipw)))]
        else:
            self.opw = np.array(opw)

        if self.mpw is None and self.ipw is None and self.opw is None:
            self.mpw = np.arange(self.nbins)

        if align:
            if align!=0:
                self.data = np.roll(self.data,align)
                #prepare=True #? #keep this for 1937?
            #self.shiftit(align,save=True)



        if prepare: #change this for jitter (prepare set to False here)
            self.interpulse_align()
            #self.normalize() #do not do this

        self.period = period

        self.null = False
        if np.all(self.data==self.data[0]) or np.all(np.isnan(self.data)):
            self.null = True


    def interpulse_align(self):
        """
        Align the pulse such that the main pulse is at phase=0.25 and the interpulse is at phase = 0.75
        """
        self.data = np.roll(u.center_max(self.data),-len(self.data)//4)

    def center_align(self):
        """
        Align the pulse such that the peak is in the center
        """
        self.data = u.center_max(self.data)
        
    def normalize(self,area=False):
        """
        Normalize the pulse so that the peak has a value of 1.0
        """
        if area:
            self.data = u.normalize_area(self.data)
        else:
            minimum = np.mean(self.getOffpulse())
            self.data = u.normalize(self.data,minimum=minimum)
        

    def getFWHM(self,simple=False,timeunits=True):
        """
        Get the full width at half maximum of the main component of the pulse
        """
        #remove baseline? what if no offpulse window?
        dbin = u.FWHM(self.data,notcentered=True)#,window=800)
        factor=1
        if timeunits and self.period!=None:
            factor = self.period/self.nbins
        return factor*dbin
        


    def getWeff(self,fourier=False,sumonly=False,timeunits=True):
        """
        Calculate the effective width of the pulse
        """
        if not timeunits or self.period is None:
            return None
        P=self.period
        N=self.nbins
        U=u.normalize(self.data,simple=True) #remove baseline?
        
        tot=np.sum(np.power(U[1:]-U[:-1],2))
        if sumonly:
            return tot
        self.weff=P/np.sqrt(N*tot)
        return self.weff

    def getSN(self):
        """
        Calculate a very crude S/N
        """
        return np.max(self.data)/self.getOffpulseNoise()



    def remove_baseline(self,save=True):
        """
        Subtract the baseline of the pulse so that it will be set to zero mean
        """
        if self.opw is None:
            print("No Offpulse") #do this?
            return
        opmean = np.mean(self.getOffpulse())
        if save:
            self.data = self.data - opmean
            return self.data
        return self.data - opmean

    

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
        return self.getMainpulse(),self.getInterpulse(),self.getOffpulse()

    ### Get each of the auto-correlation functions of the pulse components, if they exist
    def getMainpulseACF(self):
        mp=self.getMainpulse()
        return u.acf(mp,var=False,norm_by_tau=True)
    def getInterpulseACF(self):
        if self.ipw is None:
            return None
        ip=self.getInterpulse()
        return u.acf(ip,var=False,norm_by_tau=True)
    def getOffpulseACF(self):
        if self.opw is None:
            return None
        op=self.getOffpulse()
        return u.acf(op,var=False,norm_by_tau=True)
    def getAllACF(self):
        return self.getMainpulseACF(),self.getInterpulseACF(),self.getOffpulseACF()
    
    
    def getOffpulseNoise(self,mean=False,full=False):
        """
        Return the off-pulse noise statistics
        """
        if self.opw is None:
            return None
        op=self.getOffpulse()
        if mean:
            return np.mean(op)
        if full:
            return np.mean(op),np.std(op)
        return np.std(op)

    def getOffpulseZCT(self):
        """
        Perform a zero-crossing test of the offpulse noise
        """
        return u.zct(self.getOffpulse(),full=True,meansub=True)



    # todo: include a goodness-of-fit flag (gof) as a measure of the residuals.
    def fitPulse(self,template,fixedphase=False,rms_baseline=None):
        """
        Returns tauccf, tauhat, bhat, sigma_Tau, sigma_b, snr, rho
        """
        if self.null:
            return None
        if rms_baseline is None:
            self.remove_baseline()
        if fixedphase: #just return S/N
            p0 = [np.max(self.data)]
            p1,cov,infodict,mesg,ier = optimize.leastsq(lambda p,x,y: np.abs(p[0])*x - y,p0[:],args=(np.asarray(template,np.float64),np.asarray(self.data,np.float64)),full_output=True) #conversion to np.float64 fixes bug with Jacobian inversion
            noise = self.getOffpulseNoise()
            return np.abs(p1[0])/noise#,np.sqrt(cov[0][0])/noise
        if self.opw is None:
            if rms_baseline is not None:
                try:
                    return get_toa(template,self.data,rms_baseline)
                except:
                    print(self.data)
                    plt.plot(self.data)
                    plt.show()
                    raise SystemExit
            return get_toa(template,self.data,1)
        return get_toa(template,self.data,self.getOffpulseNoise())

        
    #define this so a positive shift is forward
    def shiftit(self,shift,save=False):
        """
        Shift the pulse by some phase.
        """
        x = u.shiftit(self.data,-1*shift)
        if save:
            self.data = x
        return x



    def getPeriod(self):
        return self.period

    def getNBins(self):
        return len(self.data)



    def plot_windows(self,show=True):
        """
        Diagnostic plot of the main-, inter-, and off-pulse regions
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(self.bins,self.data)
        MIN = np.min(self.data)

        diffs = np.abs(np.diff(self.mpw))
        if np.any(diffs>1):
            M = np.argmax(diffs)+1
            ax.fill_between(self.mpw[:M],np.zeros_like(self.mpw[:M])+MIN,self.data[self.mpw[:M]],facecolor='b',alpha=0.5)
            ax.fill_between(self.mpw[M:],np.zeros_like(self.mpw[M:])+MIN,self.data[self.mpw[M:]],facecolor='b',alpha=0.5)
        else:
            ax.fill_between(self.mpw,np.zeros_like(self.mpw)+MIN,self.data[self.mpw],facecolor='b',alpha=0.5)



        if self.ipw != None:
            diffs = np.abs(np.diff(self.ipw))
            if np.any(diffs>1):
                M = np.argmax(diffs)+1
                ax.fill_between(self.ipw[:M],np.zeros_like(self.ipw[:M])+MIN,self.data[self.ipw[:M]],facecolor='g',alpha=0.5)
                ax.fill_between(self.ipw[M:],np.zeros_like(self.ipw[M:])+MIN,self.data[self.ipw[M:]],facecolor='g',alpha=0.5)
            else:
                ax.fill_between(self.ipw,np.zeros_like(self.ipw)+MIN,self.data[self.ipw],facecolor='g',alpha=0.5)


        ax.set_xlim(self.bins[0],self.bins[-1])
        dy = np.ptp(self.data)
        ax.set_ylim(np.min(self.data)-0.05*dy,np.max(self.data)+0.05*dy)
        if show:
            plt.show()
