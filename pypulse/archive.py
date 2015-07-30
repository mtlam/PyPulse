'''
Michael Lam 2015

EXAMPLE USAGE:

ar = Archive(filename)
ar.tscrunch()
ar.pavplot()


TODO:

Check POL_TYPE in the above to figure out how to pscrunch

Add emulate_psrchive mode

Allow for chopping of subints, frequencies, etc?
HISTORY may not exist for raw files

Flip order of arguments in scrunching?

pyfits.open() use memmap via lowmem flag

Check if time() still works.
'''

import numpy as np
import numpy.ma as ma
import gc as g
import matplotlib.pyplot as plt
import time
import utils as u
import singlepulse as SP
import par
Par = par.Par
import decimal as d
Decimal = d.Decimal
try:
    import astropy.io.fits as pyfits
except:
    import pyfits



class Archive:
    def __init__(self,filename,prepare=True,lowmem=False,verbose=True,weight=True,center_pulse=True,remove_baseline=True):
        ## Parse filename here?
        self.filename = str(filename) #fix unicode issue
        self.prepare = prepare
        self.lowmem = lowmem
        self.verbose = verbose
        self.center_pulse = center_pulse
        self.remove_baseline = remove_baseline
        if verbose:
            print "Loading: %s" % self.filename
            t0=time.time()

        self.load(self.filename,prepare=prepare,center_pulse=center_pulse,remove_baseline=remove_baseline,weight=weight)
        if not self.lowmem:
            self.data_orig = np.copy(self.data)
        if verbose:
            t1=time.time()
            print "Load time: %0.2f s" % (t1-t0)

        #self.reset(False) #put prepare into here?, copying to arch is done here

        #if prepare:
        #    self.pscrunch()

        #if verbose and prepare:
        #    t2=time.time()
        #    print "Prep time: %0.2f s" % (t2-t1)

    def __repr__(self):
        return "Archive(%r,prepare=%r,lowmem=%r,verbose=%r)" % (self.filename,self.prepare,self.lowmem,self.verbose)
    def __str__(self):
        return self.filename



    def load(self,filename,prepare=True,center_pulse=True,remove_baseline=True,weight=True):
        if filename is None: #Needed?
            filename = self.filename
        try:
            hdulist = pyfits.open(filename,ignore_missing_end=True)
        except IOError:
            print "Filename not found"
            raise SystemExit
        self.header = hdulist[0].header
        
        self.history = History(hdulist['HISTORY'])
        nsubint = self.history.getLatest("NSUB")
        npol = self.history.getLatest("NPOL")
        nchan = self.history.getLatest("NCHAN")
        nbin = self.history.getLatest("NBIN")

        self.params = Par(map(lambda x: x[0],hdulist['PSRPARAM'].data),numwrap=float)

        self.subintinfo = dict()
        for i,column in enumerate(hdulist['SUBINT'].columns[:-3]):#[:-5]):
            self.subintinfo[column.name] = hdulist['SUBINT'].data[column.name]
        self.subintheader = dict()
        for i,key in enumerate(hdulist['SUBINT'].header):
            self.subintheader[key] = hdulist['SUBINT'].header[key]


        self.durations = self.subintinfo['TSUBINT']

        DATA = hdulist['SUBINT'].data['DATA']
        #Definitions in Base/Formats/PSRFITS/ProfileColumn.C
        DAT_FREQ = hdulist['SUBINT'].data['DAT_FREQ']
        DAT_WTS = hdulist['SUBINT'].data['DAT_WTS']
        if not weight:
            DAT_WTS = np.ones(np.shape(DAT_WTS))
        DAT_SCL = hdulist['SUBINT'].data['DAT_SCL']#.flatten()
        DAT_OFFS = hdulist['SUBINT'].data['DAT_OFFS']#.flatten()

        # This guarantees reshape to (t,pol,freq,phase) dimensions but is extremely slow
        #DATA = np.reshape(DATA,(nsubint,npol,nchan,nbin),order='C')
        #DAT_SCL = np.reshape(DAT_SCL,(nsubint,npol,nchan,1),order='C')
        #DAT_OFFS = np.reshape(DAT_OFFS,(nsubint,npol,nchan,1),order='C')
        #self.data = DAT_SCL * np.array(DATA,dtype=np.float) + DAT_OFFS         
        #if weight:
        #    #DAT_WTS = u.normalize(DAT_WTS) #not needed?
        #    DAT_WTS = np.reshape(DAT_WTS,(nsubint,1,nchan,1),order='F')
        #    self.data = self.data * DAT_WTS


        self.data = np.zeros((nsubint,npol,nchan,nbin))
        data = np.zeros((nsubint,npol,nchan,nbin))
        
        I = range(nsubint)
        J = range(npol)
        K = range(nchan)
        
        
        if np.size(DAT_WTS) == 1:
            DAT_WTS[0] == 1.0
        else:
            #DAT_WTS = u.normalize(DAT_WTS) #close???
            DAT_WTS /= np.max(DAT_WTS) #close???

        #print DAT_WTS,DAT_OFFS
        if nsubint == 1 and npol == 1 and nchan == 1:
            self.data = (DAT_SCL*DATA+DAT_OFFS)*DAT_WTS
        elif nsubint == 1 and npol == 1:
            for k in K:
                self.data[0,0,k,:] = (DAT_SCL[0,k]*DATA[0,0,k,:]+DAT_OFFS[0,k])*DAT_WTS[0] #dat WTS[0]?
        elif nsubint == 1 and nchan == 1:               
            for j in J:
                self.data[0,j,0,:] = (DAT_SCL[0,j]*DATA[0,j,0,:]+DAT_OFFS[0,j])*DAT_WTS[0]
        elif npol == 1 and nchan == 1:
            for i in I:
                self.data[i,0,0,:] = (DAT_SCL[i,0]*DATA[i,0,0,:]+DAT_OFFS[i,0])*DAT_WTS[0]
        else: #if nsubint == 1 or npol == 1 or nchan == 1 this works, or all three are not 1, might want to split this up
            for i in I:
                for j in J:
                    jnchan = j*nchan
                    for k in K:
                        self.data[i,j,k,:] = (DAT_SCL[i,jnchan+k]*DATA[i,j,k,:]+DAT_OFFS[i,jnchan+k])*DAT_WTS[i,k]

        bw = self.getBandwidth()
        #if bw < 0:
        #    print "foo"
        #    tempdata = np.copy(self.data)
        #    MAX = K[-1]
        #    for k in K:
        #        self.data[:,:,k,:] = tempdata[:,:,MAX-k,:]
            
            
        if prepare:
            self.pscrunch()
            self.dedisperse()

        self.calculateAverageProfile()

        
        if center_pulse:
            self.center()

        if remove_baseline:
            self.removeBaseline()

        return
        
    #rename this?, not the same as psrchive's unload, which saves!
    def unload(self):
        self.gc()

    def gc(self):
        """
        Manually clear the data cube for python garbage collection
        """
        if self.verbose:
            t0=time.time()
        self.data = None
        self.data_orig = None
        if self.verbose:
            t1=time.time()
            print "Unload time: %0.2f s" % (t1-t0)
        g.collect()


    def shape(self,squeeze=True):
        """
        Return the current shape of the data array
        """
        return np.shape(self.getData(squeeze=squeeze))
    def reset(self,prepare=True):
        """
        Replace the arch with the original clone
        """
        if self.lowmem:
            self.data = self.load(self.filename,prepare=prepare)            
        else:
            self.data = np.copy(self.data_orig)
        self.durations = self.subintinfo['TSUBINT']
        #if prepare:
        #    self.scrunch()


    def scrunch(self,arg='Dp'):
        """
        average the data cube along different axes
        """
        if 'T' in arg:
            self.data[0,:,:,:] = np.mean(self.data,axis=0) 
            self.data = self.data[0:1,:,:,:] #resize
            self.durations = np.array([self.getDuration()])
        if 'p' in arg:
            self.pscrunch() #throw this the other way
        if 'F' in arg:
            self.data[:,:,0,:] = np.mean(self.data,axis=2)
            self.data = self.data[:,:,0:1,:]
        if 'D' in arg:
            self.dedisperse()
        if 'B' in arg:
            self.data[:,:,:,0] = np.mean(self.data,axis=3)
            self.data = self.data[:,:,:,0:1]
        return self

    def tscrunch(self,nsubint=None,factor=None):
        """
        average the data cube along the time dimension
        """
        if nsubint == 1 or (factor is None and nsubint is None):
            return self.scrunch('T')
        if factor == 1:
            return self
        if factor is None and nsubint is not None:
            factor = self.getNsubint()/nsubint
            if self.getNsubint()%nsubint != 0:
                factor += 1
    
        nsub = self.getNsubint()
        retval = np.zeros((len(np.r_[0:nsub:factor]),self.getNpol(),self.getNchan(),self.getNbin()))
        counts = np.zeros_like(retval)
        newdurations = np.zeros(np.shape(retval)[0])
        for i in range(factor):        
            arr = self.data[i:nsub:factor,:,:,:] 
            count = np.ones_like(arr)
            length = np.shape(arr)[0]
            retval[:length,:,:,:] += arr
            counts[:length,:,:,:] += count
            newdurations[:length] += self.durations[i:nsub:factor]
        retval = retval/counts
        self.data = retval
        self.durations = newdurations
        return self

    def pscrunch(self):
        """
        average the data cube along the polarization dimension
        Coherence data is:
        A = PP
        B = QQ
        C = Re[PQ]
        D = Im[PQ]
        Therefore, Stokes are:
        I = A+B
        Q = A-B
        U = 2C
        V = 2D

        Note: What about circular versus linear, npol==2, etc.?
        """
        #print self.subintheader['POL_TYPE']
        if self.subintheader['POL_TYPE'] == "AABBCRCI": #Coherence:
            A = self.data[:,0,:,:]
            B = self.data[:,1,:,:]
            self.data[:,0,:,:] = A+B
        elif self.subintheader['POL_TYPE'] == "IQUV": #Stokes
            I = self.data[:,0,:,:] #No need to average the other dimensions?
            self.data[:,0,:,:] = I
        self.data = self.data[:,0:1,:,:] #keeps the shape
        return self

    def fscrunch(self,nchan=None,factor=None):
        """
        average the data cube along the frequency dimension
        """
        if nchan == 1 or (factor is None and nchan is None):
            return self.scrunch('F')
        if factor == 1:
            return self
        if factor is None and nchan is not None:
            factor = self.getNchan()/nchan
            if self.getNchan()%nchan != 0:
                factor += 1

        nch = self.getNchan()
        retval = np.zeros((self.getNsubint(),self.getNpol(),len(np.r_[0:nch:factor]),self.getNbin()))
        counts = np.zeros_like(retval)
        for i in range(factor):        
            arr = self.data[:,:,i:nch:factor,:] 
            count = np.ones_like(arr)
            length = np.shape(arr)[2]
            retval[:,:,:length,:] += arr
            counts[:,:,:length,:] += count
        retval = retval/counts
        self.data = retval
        return self

    def bscrunch(self,nbins=None,factor=None):
        """
        average the data cube along the phase dimension
        """
        if nbins == 1 or (factor is None and nbins is None):
            return self.scrunch('B')
        if factor == 1:
            return self
        if factor is None and nchan is not None:
            factor = self.getNchan()/nchan
            if self.getNchan()%nchan != 0:
                factor += 1
        else:
            nch = self.getNchan()
            retval = np.zeros((self.getNsubint(),self.getNpol(),len(np.r_[0:nch:factor]),self.getNbin()))
            counts = np.zeros_like(retval)
            for i in range(factor):        
                arr = self.data[:,:,i:nch:factor,:] 
                count = np.ones_like(arr)
                length = np.shape(arr)[0]
                retval[:,:,:length,:] += arr
                counts[:,:,:length,:] += count
            retval = retval/counts
            self.data = retval
        return self



    def dedisperse(self,DM=None,barycentric=True,reverse=False,wcfreq=False):
        """
        De-disperse the pulses
        if DM is given, use this value to compute the time_delays
        """
        nchan = self.getNchan()
        if nchan == 1: #do not dedisperse
            return self
        Faxis = self.getAxis('F')
        nsubint = self.getNsubint()
        npol = self.getNpol()
        nbin = self.getNbin()
        if DM is None:
            DM = self.getDM()
        cfreq = self.getCenterFrequency(weighted=wcfreq)
        time_delays = 4.149e3*DM*(cfreq**(-2) - np.power(Faxis,-2)) #freq in MHz, delays in seconds
        bin_delays = (time_delays / self.getPeriod())*nbin
        bin_delays = bin_delays % nbin
        if reverse:
            sign = 1
        else:
            sign = -1

        J = range(nsubint)
        K = range(npol)
           
        for i,delay in enumerate(bin_delays):
            for j in J:
                for k in K:
                    self.data[j,k,i,:] = u.shiftit(self.data[j,k,i,:],sign*delay)
        return self
    def dededisperse(self,DM=None,barycentric=True): 
        """
        Remove the dedispersion of the pulses
        Note: Errors might propagate?
        """
        self.dedisperse(DM=DM,barycentric=barycentric,reverse=True)
        return self


    def calculateAverageProfile(self):
        self.average_profile = np.mean(np.mean(self.data,axis=2),axis=0)
        if np.shape(self.average_profile)[0] != 1:
            self.average_profile = self.average_profile[0,:] + self.average_profile[1,:] #polarization add
        else:
            self.average_profile = self.average_profile[0,:]
        self.calculateOffpulseWindow()

    def calculateOffpulseWindow(self):
        self.spavg = SP.SinglePulse(self.average_profile,windowsize=int(self.getNbin()/8))
        self.opw = self.spavg.opw


    def superprep(self):
        """
        This will do pscrunching, centering, and dedispersing all at once to avoid multiple loops?
        Does not work so far.
        """
        nsubint = self.getNsubint()
        npol = self.getNpol()
        nchan = self.getNchan()
        nbin = self.getNbin()

        center_bin = int(nbin*phase_offset)
        maxind = np.argmax(self.average_profile)
        diff = center_bin - maxind

        Faxis = self.getAxis('F')
        nsubint = self.getNsubint()
        npol = self.getNpol()
        nbin = self.getNbin()
        DM = self.getDM()
        cfreq = self.getCenterFrequency()
        time_delays = 4.149e3*DM*(cfreq**(-2) - np.power(Faxis,-2)) #DM in MHz, delays in seconds
        bin_delays = (time_delays / self.getPeriod())*nbin
        bin_delays = bin_delays % nbin
        if reverse:
            sign = 1
        else:
            sign = -1
           

        for i in xrange(nsubint):
            for j in xrange(npol):
                for k in xrange(nchan):
                    temp = self.data[i,j,k,:]
                    temp = u.shiftit(temp,sign*delay)
                    self.data[i,j,k,:] = np.roll(temp,diff)# - np.mean(temp[self.spavg.opw])
        self.average_profile -= np.mean(self.average_profile[self.spavg.opw])
        return self


        self.average_profile = np.roll(self.average_profile,diff)
        self.calculateOffpulseWindow()
        return self


    def center(self,phase_offset=0.5):
        """
        Center the peak of the pulse in the middle of the data arrays. 
        """
        nsubint = self.getNsubint()
        npol = self.getNpol()
        nchan = self.getNchan()
        nbin = self.getNbin()

        center_bin = int(nbin*phase_offset)
        maxind = np.argmax(self.average_profile)
        diff = center_bin - maxind

        for i in xrange(nsubint):
            for j in xrange(npol):
                for k in xrange(nchan):
                    self.data[i,j,k,:] = np.roll(self.data[i,j,k,:],diff)
        self.average_profile = np.roll(self.average_profile,diff)
        self.calculateOffpulseWindow()
        return self

    def removeBaseline(self):
        nsubint = self.getNsubint()
        npol = self.getNpol()
        nchan = self.getNchan()
        nbin = self.getNbin()

        for i in xrange(nsubint):
            for j in xrange(npol):
                for k in xrange(nchan):
                    baseline = np.mean(self.data[i,j,k,self.spavg.opw])
                    self.data[i,j,k,:] -= baseline
        self.average_profile -= np.mean(self.average_profile[self.spavg.opw])
        return self
    def remove_baseline(self): #for the psrchive naming convention
        return self.removeBaseline() 



    def getData(self,squeeze=True,setnan=None):
        """
        Returns the data array, fully squeezed
        """
        if squeeze:
            data = self.data.squeeze()
        else:
            data = self.data
        if setnan is not None:
            data[data==setnan]=np.nan
        return data


    def saveData(self,filename=None,ext='npy',ascii=False):
        """
        Save the data array to a different format
        """
        if filename is None:
            filename = self.filename
            filename = ".".join(filename.split(".")[:-1])+"."+ext
        if self.verbose:
            print "Saving: %s" % filename
        if ascii:
            shape = self.shape(squeeze=False)
            nsubint = self.getNsubint()
            npol = self.getNpol()
            nchan = self.getNchan()
            nbin = self.getNbin()
            output = ""
            if shape[0] == 1 and shape[1] == 1 and shape[2] == 1:
                np.savetxt(filename,self.getData())
                return
            elif ((shape[0] == 1 and shape[1] == 1) or
                  (shape[0] == 1 and shape[2] == 1) or
                  (shape[1] == 1 and shape[2] == 1)):
                np.savetxt(filename,self.getData())
                return
            for i in xrange(nsubint):
                for j in xrange(npol):
                    for k in xrange(nchan):
                        for l in xrange(nbin):
                            output += "%i %i %i %i %.18e\n" % (i,j,k,l,self.data[i,j,k,l])

            FILE = open(filename,'w')
            FILE.write(output)
            FILE.close()
        else:
            np.save(filename,self.getData())
        return
            


    def outputPulses(self,filename):
        """
        Write out a .npy file
        """
        np.save(filename,self.getData())
        return
    


    def getAxis(self,flag=None,edges=False,wcfreq=False):
        """
        Get F/T axes for plotting
        If edges: do not return centers for each. Better for imshow plotting because of extents.
        """
        if flag == 'T':
            durations = self.durations
            csum = np.cumsum(durations)
            edgearr = np.concatenate(([0],csum))
            if edges:
                return edgearr
            else: #centered
                return csum-np.diff(edgearr)/2.0
        elif flag == 'F':
            return self.subintinfo['DAT_FREQ'][0] #temporary replacement
            nchan = self.getNchan()
            fc = self.getCenterFrequency(weighted=wcfreq)
            bw = self.getBandwidth()
            df = np.abs(bw)/nchan
            if edges:
                arr = np.array((np.arange(nchan+1) - (nchan+1)/2.0 + 0.5)*df + fc)
            else:
                arr = np.array((np.arange(nchan) - nchan/2.0 + 0.5)*df + fc) #unweighted frequencies!
            if bw < 0.0:
                return arr[::-1] #???
            return arr

        else: #do both?
            pass


    #Assumes the shape of data is (t,f,b) (i.e. polarization scrunched)
    def getPulse(self,t,f=None):
        """
        Get pulse(t,f). If f==None, get pulse(t)
        """
        if f is None:
            if self.shape(squeeze=False)[2] == 1:
                return self.getData()[t,:]
            return np.mean(self.getData()[t,:,:],axis=0)
        return self.getData()[t,f,:]



    # Assumes it is calibrated
    # Better to replace with SinglePulse's fitPulse
    def getPeakFlux(self,t,f=None):
        """
        Return the maximum value of the pulses, not typically used
        """
        pulse = self.getPulse(t,f)
        return np.max(pulse)
    def getIntegratedFlux(self,t,f=None):
        """
        Return the integrated value of the pulses, not typically used
        """
        pulse = self.getPulse(t,f)
        return np.trapz(pulse)


    def getSinglePulses(self,func=None,windowsize=None,**kwargs): 
        """
        Efficiently wraps self.data with SP.SinglePulse
        """

        if func is None:
            func = lambda x: x
        newshape = self.shape()[:-1]
        data = self.getData() #properly weighted
        period = self.getPeriod()
        if newshape==():
            return SP.SinglePulse(func(data),period=period,**kwargs)
        retval = np.empty(newshape,dtype=np.object)
        for ind in np.ndindex(newshape):
            pulse = func(data[ind])
            retval[ind]=SP.SinglePulse(pulse,period=period,**kwargs)
        return retval





    #Given a list of numbers corresponding to the arguments returned
    def fitPulses(self,template,nums,flatten=False,func=None,windowsize=None,**kwargs):
        """
        Fit all of the pulses with a given template
        """
        nums = np.array(nums)
        if windowsize is not None:
            sptemp = SP.SinglePulse(template,windowsize=windowsize)
            opw = sptemp.opw
            kwargs["opw"] = opw #apply this windowing to alll single pulses
        sps = self.getSinglePulses(func=func,**kwargs)

        if np.shape(sps)==(): #single pulse
            x = np.array(sps.fitPulse(template))
            return x[nums]

        d = dict()
        for num in nums:
            d[num] = np.zeros(np.shape(sps))
        for ind in np.ndindex(np.shape(sps)):
            sp = sps[ind]
            x = sp.fitPulse(template)
            for num in nums:
                if x is None:
                    d[num][ind] = np.nan
                else:
                    d[num][ind] = x[num]
        retval = list()
        for num in nums:
            if flatten:
                retval.append(d[num].flatten())
            else:
                retval.append(d[num])
        return tuple(retval)


    #just bscrunch this?
    def getDynamicSpectrum(self,window=None,template=None,mpw=None,align=None,windowsize=None,weight=True,verbose=False,snr=False):
        """
        Return the dynamic spectrum
        window: return the dynamic spectrum using only a certain phase bins
        Should use a numpy array for this
        When thrown into imshow, tranpose puts frequency on the y-axis, time on the x
        """
        fullshape = self.shape(squeeze=False)
        if fullshape[0] != 1 and fullshape[1] == 1 and fullshape[2] != 1: #requires polarization scrunch for now
            bw = self.getBandwidth()
            data = self.getData()
            shape = self.shape()

            if bw < 0:
                wrapfunc = lambda x: np.transpose(x) #do not flipud?
            else:
                wrapfunc = lambda x: np.transpose(x)
            if template is not None and mpw is not None:
                gs = np.zeros((fullshape[0],fullshape[2]))
                offs = np.zeros((fullshape[0],fullshape[2]))
                sig_gs = np.zeros((fullshape[0],fullshape[2]))
                I = range(fullshape[0])
                J = range(fullshape[2])


                raise SystemExit

                if snr:
                    ind = -2
                else:
                    ind = 2

                if fullshape[0] == 1 or fullshape[2] == 1:
                    if fullshape[0] == 1: #only one subintegration
                        K = J
                    else: #only one frequency channel
                        K = I
                    for i in K:
                        sp = SP.SinglePulse(data[i],mpw=mpw,align=align)
                        baseline = sp.getOffpulseNoise(mean=True) #get mean value of offpulse
                        spfit = sp.fitPulse(template)
                        if spfit!=None:
                            gs[i] = spfit[ind] #bhat
                            offs[i] = baseline
                            sig_gs[i] = spfit[4]
                    return wrapfunc(gs),wrapfunc(baseline),wrapfunc(sig_gs)
                for i in I:
                    if verbose:
                        print i,I[-1]
                    for j in J:
                        sp = SP.SinglePulse(data[i,j],mpw=mpw,align=align)
                        baseline = sp.getOffpulseNoise(mean=True) #get mean value of offpulse
                        spfit = sp.fitPulse(template)
#                        if spfit==None:
#                            print i,j
#                            plot(self.data[i,j])
#                            show()
#                            raise SystemExit
                        if spfit!=None:
                            gs[i,j] = spfit[ind] #bhat
                            offs[i,j] = baseline
                            sig_gs[i,j] = spfit[4]  
                return wrapfunc(gs),wrapfunc(offs),wrapfunc(sig_gs)
                
            #kind of hard wired
            if window==None:
                return wrapfunc(np.mean(data,axis=2))
            else:
                return wrapfunc(np.mean(data[:,:,window],axis=2))



    def plot(self,ax=None,show=True):
        """
        Basic plotter of data
        """
        data = self.getData()
        if len(np.shape(data))==1:
            if ax is None:
                plt.plot(data)
            else:
                ax.plot(data)
            if show:
                plt.show()
        else:
            print "Invalid dimensions"
    def imshow(self,ax=None,cbar=False,mask=None,show=True):
        """
        Basic imshow of data
        """
        data = self.getData(setnan=0.0)
        if len(np.shape(data))==2:
            if mask is not None:
                u.imshow(ma.masked_array(data),ax=ax,mask=mask)
            else:
                u.imshow(data,ax=ax) 
            if cbar:
                plt.colorbar()
            if show:
                plt.show()
        else:
            print "Invalid dimensions"
        return ax


    def pavplot(self,ax=None,mode="GTpd",show=True,wcfreq=True):#,ax=None,mask=None,show=True):
        """
        Produces a pav-like plot for comparison
        """
        data = self.getData(setnan=0.0)
        if len(np.shape(data))==2:
            shape = self.shape(squeeze=False)
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111)
            if shape[0] == 1 and shape[1] == 1: #fix this to match mode
                Fedges = self.getAxis('F',edges=True) #is this true?
                cmap = plt.cm.afmhot
                cmap.set_bad(color='k',alpha=1.0)
                u.imshow(self.getData(),ax=ax,extent=[0,1,Fedges[0],Fedges[-1]],cmap=cmap)
                ax.set_xlabel("Pulse Phase")
                ax.set_ylabel("Frequency (MHz)")
                ax.set_title("%s %s\nFreq %0.3f MHz BW: %0.3f Length %0.3f S/N %0.3f"%(self.getName(),self.filename,self.getCenterFrequency(weighted=wcfreq),self.getBandwidth(),self.getDuration(),self.getSN()))#get the basename?
                ax2 = ax.twinx()
                ax2.set_ylim(0,self.getNchan())
                ax2.set_ylabel("Index")
                if show:
                    plt.show()
        else:
            print "Invalid dimensions"
        return ax


    def joyDivision(self,border=0.1,labels=False,album=True,**kwargs):
        return self.joy(border=border,labels=labels,album=album,**kwargs)
    def joy(self,offset=None,border=0,labels=True,album=False):
        """
        Joy Division plot of data, works like imshow
        Can be slow for many calls of plot!
        """
        data = self.getData()
        if len(np.shape(data))==2:
            if offset==None:
                offset = np.max(np.average(data,axis=0)) * 2.10 #?

            fig = plt.figure()           
            if album:
                bgcolor = 'black'
                ax=fig.add_subplot(111,axisbg=bgcolor)
                color='w'
            else: 
                bgcolor = 'white'
                ax=fig.add_subplot(111)
                color='k'
            
            XMIN = 0
            XMAX = len(data[0])-1
            YMIN = 0-offset
            YMAX = (1+len(data))*offset
            XLOW = XMIN-(XMAX-XMIN)*border
            XHIGH = (XMAX-XMIN)*border+XMAX
            YLOW = YMIN-(YMAX-YMIN)*border
            YHIGH = (YMAX-YMIN)*border+YMAX

            if album:
                x = np.arange(len(data[0]))
                lower_limit = np.ones(len(data[0]))*YLOW
                for i in range(len(data)-1,-1,-1):
                    y = self.data[i]+offset*i
                    ax.plot(y,color)

                    ax.set_xlim(XLOW,XHIGH)
                    ax.set_ylim(YLOW,YHIGH)
                    ax.fill_between(x,y,where=y>=YLOW,color="red")

            else:
                for i in range(len(data)):
                    ax.plot(data[i]+offset*i,color)

            ax.set_xlim(XLOW,XHIGH)
            ax.set_ylim(YLOW,YHIGH)

            if not labels:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            plt.show()
        else:
            print "Invalid dimensions"



            

        
    def time(self,template,filename,MJD=False,simple=False,**kwargs):
        """
        Times the pulses and outputs in the tempo2_IPTA format similar to pat.
        MJD: if True, return TOAs in MJD units, else in time units corresponding to a bin number
        """

        if isinstance(template,Archive):
            artemp = template
            tempname = artemp.filename
            template = artemp.getData()
        elif isinstance(template,str):
            tempname = template
            artemp = Archive(tempname)
            template = artemp.getData()
        elif isinstance(template,np.ndarray) or isinstance(template,list):
            tempname = "None"
        else:
            return

        template = u.center_max(u.normalize(template,simple=True)) #just to enforce
        
        

        tauhat,bhat,sigma_tau,sigma_b,snrs = self.fitPulses(template,[1,2,3,4,5],**kwargs)
        Taxis = self.getAxis('T')
        Faxis = self.getAxis('F')
        
        #Reshape if necessary
        tauhat = tauhat.reshape(len(Taxis),len(Faxis))
        bhat = bhat.reshape(len(Taxis),len(Faxis))
        sigma_tau = sigma_tau.reshape(len(Taxis),len(Faxis))
        sigma_b = sigma_b.reshape(len(Taxis),len(Faxis))
        snrs = snrs.reshape(len(Taxis),len(Faxis))


        telescope = self.getTelescope() #lowercase? This may cause tempo2 errors
        frontend = self.getFrontend()
        backend = self.getBackend()
        bw = np.abs(self.getBandwidth())
        nchan = self.getNchan()
        chanbw = bw / nchan
        nbin = self.getNbin()

        dt = self.getTbin()

        if MJD:
            shape = np.shape(tauhat)
            tauhatdec = np.reshape(np.array(map(Decimal,tauhat.flatten()),dtype=np.dtype(Decimal)),shape)
            tauhat = tauhatdec * Decimal(dt/86400.0) #day units
            checknan = lambda x: x.is_nan()
        else:
            tauhat *= (dt*1e6)
            checknan = lambda x: np.isnan(x)
        sigma_tau *= (dt*1e6)

        output = "FORMAT 1\n"

        t0 = 0.0
        for i,T in enumerate(Taxis):
            tobs = self.durations[i]
            if MJD:
                t0 = Decimal(integration.get_start_time().in_days())
            for j,F in enumerate(Faxis):
                if checknan(tauhat[i,j]):
                    continue

                toa = '{0:0.15f}'.format(tauhat[i,j]+t0)
                output += "%s %f %s   %0.3f  %s   -fe %s -be %s -bw %f -tobs %f -tmplt %s -nbin %i -nch %i -snr %0.2f -flux %0.2f -fluxerr %0.2f\n"%(self.filename,F,toa,sigma_tau[i,j],telescope,frontend,backend,chanbw,tobs,tempname,nbin,nchan,snrs[i,j],bhat[i,j],sigma_b[i,j])
                

        FILE = open(filename,'w')
        FILE.write(output)
        FILE.close()
        return




    ### ==============================
    ### Simple get functions
    ### ==============================


    def getNsubint(self):
        return self.shape(squeeze=False)[0]
    def getNpol(self):
        return self.shape(squeeze=False)[1]
    def getNchan(self): 
        return self.shape(squeeze=False)[2]
    def getNbin(self):
        return self.shape(squeeze=False)[3]
    def getPeriod(self):
        return self.params.getPeriod()

    # Best replacement for without PSRCHIVE
    def getValue(self,value):
        if value in self.header.keys():
            return self.header[value]
        if value in self.subintinfo.keys():
            return self.subintinfo[value]
        return self.params.get(value) #will return None if non-existent


    def getName(self):
        return self.header['SRC_NAME']
    def getMJD(self,full=False):
        if full:
            return self.header['STT_IMJD']+self.header['STT_SMJD']/86400.0
        return self.header['STT_IMJD']
    def getTbin(self): #get the time per bin
        return self.getPeriod() / self.getNbin()
    def getDM(self):
        return self.params.getDM()
    #def setDM(self,value):
    #    return self.arch.set_dispersion_measure(value)
    ### Get coords info in header
    def getCoords(self,parse=True): #use get/set coorinates? Use astropy?
        if parse:
            RA = tuple(map(lambda x: float(x),self.header['RA'].split(":")))
            dec = tuple(map(lambda x: float(x),self.header['DEC'].split(":")))
        else:
            RA = self.header['RA']
            dec = self.header['DEC']
        return RA,dec
    def getPulsarCoords(self,parse=True):
        return self.getCoords(parse=parse)
    def getTelescopeCoords(self):
        return self.header['ANT_X'],self.header['ANT_Y'],self.header['ANT_Z']


    def getBandwidth(self,header=False):
        if header:
            return self.header['OBSBW']
        else:
            return self.subintheader['CHAN_BW']*self.subintheader['NCHAN']
    def getDuration(self):
        return np.sum(self.subintinfo['TSUBINT']) #This is constant.
    def getCenterFrequency(self,weighted=False):
        if weighted:
            DAT_FREQ = self.subintinfo['DAT_FREQ']
            DAT_WTS = self.subintinfo['DAT_WTS']
            return np.sum(DAT_FREQ*DAT_WTS)/np.sum(DAT_WTS) 
        return self.history.getLatest("CTR_FREQ")

    def getTelescope(self):
        return self.header['TELESCOP']
    def getFrontend(self):
        return self.header['FRONTEND']
    def getBackend(self):
        return self.header['BACKEND']

    def getSN(self):
        return self.spavg.getSN()





# Takes hdulist['HISTORY']
class History:
    def __init__(self,history):
        self.dictionary = dict()
        for col in history.columns:
            self.dictionary[col.name] = list(col.array) #make a np.array?

    def getValue(self,field,num=None):
        if num is None:
            return self.dictionary[field]
        else:
            return self.dictionary[field][num]
        
    def getLatest(self,field):
        return self.getValue(field,-1)
