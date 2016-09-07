'''
Michael Lam 2015

PSRFITS specification: www.atnf.csiro.au/people/pulsar/index.html?n=Main.Psrfits

EXAMPLE USAGE:
ar = Archive(filename)
ar.tscrunch()
ar.pavplot()

TODO:

Check POL_TYPE in the above to figure out how to pscrunch

Add emulate_psrchive mode?

Allow for chopping of subints, frequencies, etc?

Flip order of arguments in scrunching?

Check if time() still works.
'''

import numpy as np
import numpy.ma as ma
import gc as g
import matplotlib.pyplot as plt
import time
import pypulse.utils as u
import pypulse.singlepulse as SP
import pypulse.par as par
Par = par.Par
import decimal as d
Decimal = d.Decimal
try:
    import astropy.io.fits as pyfits
except:
    import pyfits
import sys
if sys.version_info.major == 2:
    fmap = map    
elif sys.version_info.major == 3:
    fmap = lambda x,*args: list(map(x,*args))
    xrange = range

PSR = "PSR"
CAL = "CAL"
SEARCH = "SEARCH"




class Archive:
    def __init__(self,filename,prepare=True,lowmem=False,verbose=True,weight=True,center_pulse=True,baseline_removal=True):
        ## Parse filename here?
        self.filename = str(filename) #fix unicode issue
        self.prepare = prepare
        self.lowmem = lowmem
        self.verbose = verbose
        self.center_pulse = center_pulse
        self.baseline_removal = baseline_removal
        if verbose:
            print("Loading: %s" % self.filename)
            t0=time.time()

        self.load(self.filename,prepare=prepare,center_pulse=center_pulse,baseline_removal=baseline_removal,weight=weight)
        if not self.lowmem:
            self.data_orig = np.copy(self.data)
        if verbose:
            t1=time.time()
            print("Load time: %0.2f s" % (t1-t0))

        #self.reset(False) #put prepare into here?, copying to arch is done here

        #if prepare:
        #    self.pscrunch()

        #if verbose and prepare:
        #    t2=time.time()
        #    print("Prep time: %0.2f s" % (t2-t1))

    def __repr__(self):
        return "Archive(%r,prepare=%r,lowmem=%r,verbose=%r)" % (self.filename,self.prepare,self.lowmem,self.verbose)
    def __str__(self):
        return self.filename



    def load(self,filename,prepare=True,center_pulse=True,baseline_removal=True,weight=True):
        """
        Loads a PSRFITS file and processes
        http://www.atnf.csiro.au/people/pulsar/index.html?n=PsrfitsDocumentation.Txt
        """
        if filename is None: #Needed?
            filename = self.filename
        try:
            if self.lowmem:
                hdulist = pyfits.open(filename,ignore_missing_end=True,memmap=True)
            else:
                hdulist = pyfits.open(filename,ignore_missing_end=True)
        except IOError:
            print("Filename not found")
            raise SystemExit
        self.header = hdulist[0].header
        self.keys = fmap(lambda x: x.name,hdulist)
        tablenames = self.keys[:] #temporary list for checking other tables

        if 'HISTORY' in self.keys:
            tablenames.remove('HISTORY')
            self.history = History(hdulist['HISTORY'])
            nsubint = self.history.getLatest("NSUB")
            npol = self.history.getLatest("NPOL")
            nchan = self.history.getLatest("NCHAN")
            nbin = self.history.getLatest("NBIN")
        else:
            self.history = None
            nsubint = hdulist['SUBINT'].header['NAXIS2']
            nbin,nchan,npol,nsblk = fmap(int,hdulist['SUBINT'].columns[-1].dim[1:-1].split(","))

            
        if 'PSRPARAM' in self.keys:
            tablenames.remove('PSRPARAM')
            self.paramheaderlist = hdulist['PSRPARAM'].header.keys()
            self.paramheader = dict()
            for key in self.paramheaderlist:
                self.paramheader[key] = hdulist['PSRPARAM'].header[key]
            self.params = Par(fmap(lambda x: x[0],hdulist['PSRPARAM'].data),numwrap=float)
        elif 'PSREPHEM' in self.keys:
            tablenames.remove('PSREPHEM')
            paramkeys = fmap(lambda x: x.name,hdulist['PSREPHEM'].columns)
            paramvals = hdulist['PSREPHEM'].data[0]
            paramstrs = fmap(lambda x,y: "%s %s"%(x,y),paramkeys,paramvals)
            #self.paramheaderlist = hdulist['PSREPHEM'].header.keys()
            #self.paramheaderdict = dict()
            #for key in self.paramheaderlist:
            #    self.paramheaderdict[key] = hdulist['PSREPHEM'].header[key]
            self.params = Par(paramstrs,numwrap=float)
        else:
            self.params = None

        tablenames.remove('PRIMARY')
        if 'SUBINT' in tablenames:
            tablenames.remove('SUBINT')

        isFluxcal = False
        if 'FLUX_CAL' in tablenames: #need to account for is this appropriate?
            tablenames.remove('FLUX_CAL')
            isFluxcal = True
        self.tables = list()
        for tablename in tablenames: #remaining table names to store
            self.tables.append(hdulist[tablename].copy())
    


        #if self.header['OBS_MODE'] == 'PCM':
        if isFluxcal:


            
            raise SystemExit





        self.subintinfo = dict()
        self.subintinfolist = fmap(lambda x: x.name, hdulist['SUBINT'].columns[:-5])
        for i,column in enumerate(hdulist['SUBINT'].columns[:-5]):
            self.subintinfo[column.name] = (column.format,column.unit,hdulist['SUBINT'].data[column.name])
        self.subintheader = dict()
        self.subintheaderlist = hdulist['SUBINT'].header.keys()#for ordering
        for i,key in enumerate(hdulist['SUBINT'].header):
            self.subintheader[key] = hdulist['SUBINT'].header[key]

        DATA = hdulist['SUBINT'].data['DATA']
        if np.ndim(DATA)==5:
            DATA = DATA[:,0,:,:,:] #remove the nsblk column
        #Definitions in Base/Formats/PSRFITS/ProfileColumn.C
        DAT_FREQ = hdulist['SUBINT'].data['DAT_FREQ']
        DAT_WTS = hdulist['SUBINT'].data['DAT_WTS']
        if not weight:
            DAT_WTS = np.ones(np.shape(DAT_WTS))
        DAT_SCL = hdulist['SUBINT'].data['DAT_SCL']
        DAT_OFFS = hdulist['SUBINT'].data['DAT_OFFS']# + 0.5 #testing
        self.DAT_SCL = DAT_SCL #testing

        
        self.data = np.zeros((nsubint,npol,nchan,nbin))
        data = np.zeros((nsubint,npol,nchan,nbin))
        
        I = range(nsubint)
        J = range(npol)
        K = range(nchan)
        


        if np.size(DAT_WTS) == 1:
            DAT_WTS[0] = 1.0
        else:
            DAT_WTS /= np.max(DAT_WTS) #close???

        self.freq = DAT_FREQ
        self.weights = DAT_WTS

        if nsubint == 1 and npol == 1 and nchan == 1:
            self.data = (DAT_SCL*DATA+DAT_OFFS)#*DAT_WTS
        elif nsubint == 1 and npol == 1:
            for k in K:
                self.data[0,0,k,:] = (DAT_SCL[0,k]*DATA[0,0,k,:]+DAT_OFFS[0,k])#*DAT_WTS[0,k] #dat WTS[0]?
        elif nsubint == 1 and nchan == 1:               
            for j in J:
                self.data[0,j,0,:] = (DAT_SCL[0,j]*DATA[0,j,0,:]+DAT_OFFS[0,j])#*DAT_WTS[0]
        elif npol == 1 and nchan == 1:
            for i in I:
                self.data[i,0,0,:] = (DAT_SCL[i,0]*DATA[i,0,0,:]+DAT_OFFS[i,0])#*DAT_WTS[0]
        else: #if nsubint == 1 or npol == 1 or nchan == 1 this works, or all three are not 1, might want to split this up
            for i in I:
                for j in J:
                    jnchan = j*nchan
                    for k in K:
                        self.data[i,j,k,:] = (DAT_SCL[i,jnchan+k]*DATA[i,j,k,:]+DAT_OFFS[i,jnchan+k])#*DAT_WTS[i,k]

        bw = self.getBandwidth()
        #if bw < 0:
        #    print "foo"
        #    tempdata = np.copy(self.data)
        #    MAX = K[-1]
        #    for k in K:
        #        self.data[:,:,k,:] = tempdata[:,:,MAX-k,:]
            
        # All time-tagging info
        self.durations = self.getSubintinfo('TSUBINT')
        self.subint_starts = np.array(fmap(Decimal,self.getSubintinfo('OFFS_SUB')),dtype=np.dtype(Decimal))-self.getTbin(numwrap=Decimal)*Decimal(nbin/2.0)#+self.getMJD(full=False,numwrap=Decimal) #converts center-of-bin times to start-of-bin times, in seconds, does not include the integer MJD part
        self.channel_delays = np.zeros(nchan,dtype=np.dtype(Decimal)) #used to keep track of frequency-dependent channel delays, in bin units. Should be in Decimal?
            
        if prepare and not self.isCalibrator():
            self.pscrunch()
            self.dedisperse()

        self.calculateAverageProfile()

        
        if center_pulse and not self.isCalibrator(): #calibrator is not a pulse
            self.center()

        if baseline_removal:
            self.removeBaseline()

        hdulist.close()
        return


    def save(self,filename):
        """Save the file to a new FITS file"""

        primaryhdu = pyfits.PrimaryHDU(header=self.header) #need to make alterations to header
        hdulist = pyfits.HDUList(primaryhdu)

        if self.history is not None:
            cols = []
            for name in self.history.namelist:
                fmt,unit,array = self.history.dictionary[name]
                col = pyfits.Column(name=name,format=fmt,unit=unit,array=array)
                cols.append(col)
            historyhdr = pyfits.Header()
            for key in self.history.headerlist:
                historyhdr[key] = self.history.header[key]
            historyhdu = pyfits.BinTableHDU.from_columns(cols,name='HISTORY',header=historyhdr)
            hdulist.append(historyhdu)
            # Need to add in PyPulse changes into a HISTORY 
        #else: #else start a HISTORY table
            
                    
        if self.params is not None:
            cols = [pyfits.Column(name='PSRPARAM',format='128A',array=self.params.filename)]
            paramhdr = pyfits.Header()
            for key in self.paramheaderlist:
                paramhdr[key] = self.paramheader[key]
            paramhdu = pyfits.BinTableHDU.from_columns(cols,name='PSRPARAM')
            hdulist.append(paramhdu)
            # Need to include mode for PSREPHEM

        if len(self.tables) > 0:
            for table in self.tables:
                hdulist.append(table)

        cols = []
        for name in self.subintinfolist:
            fmt,unit,array = self.subintinfo[name]
            col = pyfits.Column(name=name,format=fmt,unit=unit,array=array)
            cols.append(col)
            # finish writing out SUBINT!

        cols.append(pyfits.Column(name='DAT_FREQ',format='%iE'%np.shape(self.freq)[1],unit='MHz',array=self.freq)) #correct size? check units?
        cols.append(pyfits.Column(name='DAT_WTS',format='%iE'%np.shape(self.weights)[1],array=self.weights)) #call getWeights()

        nsubint = self.getNsubint()
        npol = self.getNpol()
        nchan = self.getNchan()
        DAT_OFFS = np.zeros((nsubint,npol*nchan))
        DAT_SCL = np.zeros((nsubint,npol*nchan))
        DATA = self.getData(squeeze=False)
        # Following Base/Formats/PSRFITS/unload_DigitiserCounts.C
        for i in range(nsubint):
            for j in range(npol):
                jnchan = j*nchan
                for k in range(nchan):
                    MIN = np.min(DATA[i,j,k,:])
                    MAX = np.max(DATA[i,j,k,:])
                    RANGE = MAX - MIN
                    if MAX == 0 and MIN == 0:
                        DAT_SCL[i,jnchan+k] += 1.0
                    else:
                        if RANGE == 0:
                            DAT_SCL[i,jnchan+k] = MAX / 65535.0 #65534?
                        else:
                            DAT_SCL[i,jnchan+k] = RANGE / 65535.0
                        DAT_OFFS[i,jnchan+k] = MIN - (-32768.0 * DAT_SCL[i,jnchan+k])
                
                    DATA[i,j,k,:] = np.floor((DATA[i,j,k,:] - DAT_OFFS[i,jnchan+k])/DAT_SCL[i,jnchan+k] + 0.5) #why +0.5?
                    print np.min(DATA[i,j,k,:]),np.max(DATA[i,j,k,:])
                # -32768 to 32766?

                    #jnchan = j*nchan
                    #for k in K:
                    #    self.data[i,j,k,:] = (DAT_SCL[i,jnchan+k]*DATA[i,j,k,:]+DAT_OFFS[i,jnchan+k])#*DAT_WTS[i,k]



        #print self.DAT_SCL[0]
        #print 
        #print 1.0/DAT_SCL[0]
        #DAT_SCL = 1.0/DAT_SCL#???
        cols.append(pyfits.Column(name='DAT_OFFS',format='%iE'%np.size(DAT_OFFS[0]),array=DAT_OFFS))
        cols.append(pyfits.Column(name='DAT_SCL',format='%iE'%np.size(DAT_SCL[0]),array=DAT_SCL))
        cols.append(pyfits.Column(name='DATA',format='%iI'%np.size(DATA[0]),array=DATA))
        
        subinthdr = pyfits.Header()
        for key in self.subintheaderlist:
            subinthdr[key] = self.subintheader[key]
        subinthdu = pyfits.BinTableHDU.from_columns(cols,name='SUBINT',header=subinthdr)
        hdulist.append(subinthdu)



        hdulist.writeto(filename,clobber=True)#clobber=True?



    def unload(self,filename):
        return self.save(filename)

    def gc(self):
        """Manually clear the data cube for python garbage collection"""
        if self.verbose:
            t0=time.time()
        self.data = None
        self.data_orig = None
        if self.verbose:
            t1=time.time()
            print("Unload time: %0.2f s" % (t1-t0))
        g.collect()


    def shape(self,squeeze=True):
        """Return the current shape of the data array"""
        return np.shape(self.getData(squeeze=squeeze))
    def reset(self,prepare=True):
        """Replace the arch with the original clone"""
        if self.lowmem:
            self.load(self.filename,prepare=prepare) 
        else:
            self.data = np.copy(self.data_orig)
        self.durations = self.getSubintinfo('TSUBINT')
        #if prepare:
        #    self.scrunch()


    def scrunch(self,arg='Dp'):
        """average the data cube along different axes"""
        if 'T' in arg:
            self.data[0,:,:,:] = np.mean(self.data,axis=0) 
            self.data = self.data[0:1,:,:,:] #resize
            self.weights[0,:] = np.mean(self.weights,axis=0) #should be sum?
            self.weights = self.weights[0:1,:] #resize
            self.durations = np.array([self.getDuration()])
        if 'p' in arg:
            self.pscrunch() #throw this the other way
        if 'F' in arg:
            self.data[:,:,0,:] = np.mean(self.data,axis=2)
            self.data = self.data[:,:,0:1,:]
            self.weights[:,0] = np.mean(self.weights,axis=1) #problem?
            self.weigths = self.weights[:,0:1]
        if 'D' in arg:
            self.dedisperse()
        if 'B' in arg:
            self.data[:,:,:,0] = np.mean(self.data,axis=3)
            self.data = self.data[:,:,:,0:1]
        return self

    def tscrunch(self,nsubint=None,factor=None):
        """average the data cube along the time dimension"""
        if nsubint == 1 or (factor is None and nsubint is None):
            return self.scrunch('T')
        if factor == 1:
            return self
        if factor is None and nsubint is not None:
            factor = self.getNsubint()//nsubint
            if self.getNsubint()%nsubint != 0:
                factor += 1
    
        nsub = self.getNsubint()
        retval = np.zeros((len(np.r_[0:nsub:factor]),self.getNpol(),self.getNchan(),self.getNbin()))
        counts = np.zeros_like(retval)
        newdurations = np.zeros(np.shape(retval)[0])
        wretval = np.zeros((len(np.r_[0:nsub:factor]),self.getNchan()))
        wcounts = np.zeros_like(retval)
        for i in range(factor):
            # Data array
            arr = self.data[i:nsub:factor,:,:,:] 
            count = np.ones_like(arr)
            length = np.shape(arr)[0]
            retval[:length,:,:,:] += arr
            counts[:length,:,:,:] += count
            newdurations[:length] += self.durations[i:nsub:factor]
            # Weights array
            arr = self.weights[i:nsub:factor,:]
            count = np.ones_like(arr)
            wretval[:length,:,:,:] += arr
            wcounts[:length,:,:,:] += count
        retval = retval/counts
        #wretval = wretval/wcounts #is this correct?
        self.data = retval
        self.durations = newdurations
        self.weights = wretval
        return self

    def pscrunch(self):
        """
        average the data cube along the polarization dimension
        Coherence data is (see More/General/Integration_get_Stokes.C):
        A = PP
        B = QQ
        C = Re[PQ]
        D = Im[PQ]
        Therefore, if linear (FD_POLN == LIN), Stokes are:
        I = A+B
        Q = A-B
        U = 2C
        V = 2D
        If circular, Stokes are:
        I = A+B
        Q = 2C
        U = 2D
        V = A-B

        Note: What about npol==2, FD_HAND, other states, etc.?
        Should this modify npol? How to avoid double pscrunching?
        """
        if self.shape(squeeze=False)[1] == 1:
            return self
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
        """average the data cube along the frequency dimension"""
        if nchan == 1 or (factor is None and nchan is None):
            return self.scrunch('F')
        if factor == 1:
            return self
        if factor is None and nchan is not None:
            factor = self.getNchan()//nchan
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
        #self.freq = 
        return self

    def bscrunch(self,nbins=None,factor=None):
        """average the data cube along the phase dimension"""
        if nbins == 1 or (factor is None and nbins is None):
            return self.scrunch('B')
        if factor == 1:
            return self
        if factor is None and nchan is not None:
            factor = self.getNchan()//nchan
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
            self.channel_delays[i] += Decimal(str(time_delays[i])) #FIX THIS
            for j in J:
                for k in K:
                    self.data[j,k,i,:] = u.shiftit(self.data[j,k,i,:],sign*delay)
        self.calculateAverageProfile() #re-calculate the average profile
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
        if np.shape(self.average_profile)[0] != 1: #polarization add
            if self.subintheader['POL_TYPE'] == "AABBCRCI": #Coherence
                self.average_profile = self.average_profile[0,:] + self.average_profile[1,:] 
            elif self.subintheader['POL_TYPE'] == "IQUV": #Stokes
                self.average_profile = self.average_profile[0,:]
        else:
            self.average_profile = self.average_profile[0,:]
        self.calculateOffpulseWindow()

    def calculateOffpulseWindow(self):
        self.spavg = SP.SinglePulse(self.average_profile,windowsize=int(self.getNbin()//8))
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
        self.channel_delays += Decimal(str(diff*self.getTbin())) #FIX THIS
        self.calculateOffpulseWindow()
        return self

    def removeBaseline(self):
        """Removes the baseline of the pulses given an offpulse window"""
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
    remove_baseline = removeBaseline

    def calibrate(self,psrcal,fluxcal=None):
        """Calibrates using another archive"""
        if not psrcal.isCalibrator():
            raise ValueError("Require calibration archive")
        # Check if cals are appropriate?

        # Check if cal has the correct dimensions, if not perform interpolation
        
        # Time-average calibrators
        if fluxcal is not None:
            fluxcal.tscrunch()
            fdata = fluxcal.getData()
        psrcal.tscrunch()
        pdata = psrcal.getData()

        # Find cal levels, or have them predetermined?
        npol = self.getNpol()
        nchan = self.getNchan()
        nbin = self.getNbin()
        
        # Check header info CAL_DCYC, CAL_NPHS, etc, to determine on-diode
        lowinds = np.arange(0,nbin/2)
        highinds = np.arange(nbin/2,nbin)

        # Calculate calibrations
        psrcaldata = np.zeros((npol,nchan,2))
        for i in range(npol):
            for j in range(nchan):
                psrcaldata[i,j,0] = np.mean(pdata[i,j,lowinds])
                psrcaldata[i,j,1] = np.mean(pdata[i,j,highinds])
        if fluxcal is not None:
            fluxcaldata = np.zeros((npol,nchan,2))
            for i in range(npol):
                for j in range(nchan):
                    fluxcaldata[i,j,0] = np.mean(fdata[i,j,lowinds])
                    fluxcaldata[i,j,1] = np.mean(fdata[i,j,highinds])

            
        # Apply calibrations
        pass



    def getData(self,squeeze=True,setnan=None,weight=True):
        """Returns the data array, fully squeezed"""
        if weight:
            data = np.zeros_like(self.data)
            I,J,K,L = np.shape(self.data)
            I = range(I)
            J = range(J)
            K = range(K)
            for i in I:
                for j in J:
                    for k in K:                      
                        data[i,j,k,:] = self.data[i,j,k,:]*self.weights[i,k]
        else:
            data = self.data

        if squeeze:
            data = data.squeeze()

        if setnan is not None:
            data = np.where(data==setnan,np.nan,data)
        
        return np.copy(data) #removes pointer to data
    def getWeights(self,squeeze=True):
        weights = self.weights
        if squeeze:
            weights = weights.squeeze()
        return np.copy(weights)
    def setWeights(self,t,f,val):
        self.weights[t,f] = val



    def saveData(self,filename=None,ext='npy',ascii=False):
        """Save the data array to a different format"""
        if filename is None:
            filename = self.filename
            filename = ".".join(filename.split(".")[:-1])+"."+ext
        if self.verbose:
            print("Saving: %s" % filename)
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
        """ Write out a .npy file"""
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
            return self.freq[0]#self.getSubintinfo('DAT_FREQ')[0]  ### This block is a temporary replacement

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
        """Get pulse(t,f). If f==None, get pulse(t)"""
        if f is None:
            if self.shape(squeeze=False)[2] == 1:
                return self.getData()[t,:]
            return np.mean(self.getData()[t,:,:],axis=0)
        return self.getData()[t,f,:]



    # Assumes it is calibrated
    # Better to replace with SinglePulse's fitPulse
    def getPeakFlux(self,t,f=None):
        """Return the maximum value of the pulses, not typically used"""
        pulse = self.getPulse(t,f)
        return np.max(pulse)
    def getIntegratedFlux(self,t,f=None):
        """Return the integrated value of the pulses, not typically used"""
        pulse = self.getPulse(t,f)
        return np.trapz(pulse)


    def getSinglePulses(self,func=None,windowsize=None,**kwargs): 
        """Efficiently wraps self.data with SP.SinglePulse"""
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
        """Fit all of the pulses with a given template"""
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
                        print("i,%i"%(i,I[-1]))
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
            if window is None:
                return wrapfunc(np.mean(data,axis=2))
            else:
                return wrapfunc(np.mean(data[:,:,window],axis=2))



    def plot(self,ax=None,show=True):
        """Basic plotter of data"""
        data = self.getData()
        if len(np.shape(data))==1:
            if ax is None:
                plt.plot(data,'k')
                plt.xlim(0,len(data))
            else:
                ax.plot(data,'k')
                ax.set_xlim(0,len(data))
            if show:
                plt.show()
        else:
            print("Invalid dimensions")
    def imshow(self,ax=None,cbar=False,mask=None,show=True,**kwargs):
        """Basic imshow of data"""
        data = self.getData(setnan=0.0)
        if len(np.shape(data))==2:
            if mask is not None:
                u.imshow(ma.masked_array(data),ax=ax,mask=mask,**kwargs)
            else:
                u.imshow(data,ax=ax,**kwargs) 
            if cbar:
                plt.colorbar()
            if show:
                plt.show()
        else:
            print("Invalid dimensions")
        return ax


    def pavplot(self,ax=None,mode="GTpd",show=True,wcfreq=True):#,ax=None,mask=None,show=True):
        """Produces a pav-like plot for comparison"""
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
            print("Invalid dimensions")
        return ax


    def joyDivision(self,border=0.1,labels=False,album=True,**kwargs):
        """Calls waterfall() in the style of the Joy Division album cover"""
        return self.waterfall(border=border,labels=labels,album=album,**kwargs)
    def waterfall(self,offset=None,border=0,labels=True,album=False,bins=None,show=True):
        """
        Joy Division plot of data, works like imshow
        Can be slow for many calls of plot!
        """
        data = self.getData(squeeze=True)
        if len(np.shape(data))==2:
            if offset==None:
                offset = np.max(np.average(data,axis=0))#*0.5# * 2.10 #?

            fig = plt.figure(figsize=(6,6))           
            if album:
                bgcolor = 'black'
                ax=fig.add_subplot(111,axisbg=bgcolor)
                color='w'
            else: 
                bgcolor = 'white'
                ax=fig.add_subplot(111)
                color='k'

            if bins is None:
                bins = np.arange(self.getNbin())

            
            XMIN = 0
            #XMAX = len(data[0])-1
            XMAX = len(bins)-1
            YMIN = 0-offset
            YMAX = (1+len(data))*offset
            XLOW = XMIN-(XMAX-XMIN)*border
            XHIGH = (XMAX-XMIN)*border+XMAX
            YLOW = YMIN-(YMAX-YMIN)*border
            YHIGH = (YMAX-YMIN)*border+YMAX


            if album:
                x = np.arange(len(data[0]))
                lower_limit = np.ones(len(data[0]))*YLOW
                z = 0
                for i in range(len(data)-1,-1,-1):
                    z += 1 
                    y = data[i][bins]+offset*i
                    #y = np.roll(y,100*i) # for testing

                    ax.plot(y,color,zorder=z)

                    ax.set_xlim(XLOW,XHIGH)
                    ax.set_ylim(YLOW,YHIGH)
                    ax.fill_between(x,y,where=y>=YLOW,color=bgcolor,zorder=z) #testing


            else:
                for i in range(len(data)):
                    ax.plot(data[i][bins]+offset*i,color)

            ax.set_xlim(XLOW,XHIGH)
            ax.set_ylim(YLOW,YHIGH)

            if not labels:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            if show:
                plt.show()
        else:
            print("Invalid dimensions")



            
    ### NOTE: THIS NEEDS TO BE CHECKED WITH THE NEW CHANGES ###
        
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

        rollval,template = u.center_max(u.normalize(template,simple=True),full=True) # enforces a good fit

        #If given an offpulse, use that, else calculate a pre-defined one in the template Archive
        if "opw" in kwargs.items():
            opw = (kwargs['opw'] + rollval)%len(kwargs['opw']) # "roll" the array with the template
        else:
            sptemp = SP.SinglePulse(template,windowsize=len(template)/8)
            kwargs['opw'] = sptemp.opw

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
            tauhatdec = np.reshape(np.array(fmap(Decimal,tauhat.flatten()),dtype=np.dtype(Decimal)),shape)
            tauhat = tauhatdec * Decimal(dt/86400.0) #day units
            checknan = lambda x: x.is_nan()
        else:
            tauhat *= (dt*1e6)
            checknan = lambda x: np.isnan(x)
        sigma_tau *= (dt*1e6)

        output = "FORMAT 1\n"

        t0 = 0.0
        start_time = self.getMJD(full=True,numwrap=Decimal)
        for i,T in enumerate(Taxis):
            tobs = self.durations[i]
            if MJD:
                t0 = start_time + self.subint_starts[i]/Decimal(86400)
                #t0 = self.subint_starts[i]
                #t0 = Decimal(integration.get_start_time().in_days())
            for j,F in enumerate(Faxis):
                if checknan(tauhat[i,j]):
                    continue
                toa = '{0:0.15f}'.format(tauhat[i,j]+t0+self.channel_delays[j]/Decimal(86400))
                output += "%s %f %s   %0.3f  %s   -fe %s -be %s -bw %f -tobs %f -tmplt %s -nbin %i -nch %i -snr %0.2f -flux %0.2f -fluxerr %0.2f\n"%(self.filename,F,toa,sigma_tau[i,j],telescope,frontend,backend,chanbw,tobs,tempname,nbin,nchan,snrs[i,j],bhat[i,j],sigma_b[i,j])
                

        FILE = open(filename,'w')
        FILE.write(output)
        FILE.close()
        return




    ### ==============================
    ### Simple get functions
    ### ==============================


    def getNsubint(self):
        """Returns number of subintegrations"""
        return self.shape(squeeze=False)[0]
    def getNpol(self):
        """Returns number of polarizations"""
        return self.shape(squeeze=False)[1]
    def getNchan(self): 
        """Returns number of channels"""
        return self.shape(squeeze=False)[2]
    def getNbin(self):
        """Returns number of phase bins"""
        return self.shape(squeeze=False)[3]
    def getPeriod(self):
        """Returns period of the pulsar"""
        if self.isCalibrator():
            return 1.0/self.header['CAL_FREQ']
        if self.params is None:
            return None
        return self.params.getPeriod()
    # Best replacement for without PSRCHIVE
    def getValue(self,value):
        """Looks for a key in one of the headers and returns"""
        if value in self.header.keys():
            return self.header[value]
        if value in self.subintinfo.keys():
            return self.subintinfo[value][-1]
        if self.params is None:
            return None
        return self.params.get(value) #will return None if non-existent
    def getSubintinfo(self,value):
        """Returns value from subintinfo"""
        if value in self.subintinfo.keys():
            return self.subintinfo[value][-1]
        return None
    def getName(self):
        """Returns pulsar name"""
        return self.header['SRC_NAME']
    def getMJD(self,full=False,numwrap=float):
        """Returns MJD of observation"""
        if full:
            return numwrap(self.header['STT_IMJD'])+(numwrap(self.header['STT_SMJD'])+numwrap(self.header['STT_OFFS']))/numwrap(86400)
        return numwrap(self.header['STT_IMJD'])+numwrap(self.header['STT_OFFS'])
    def getTbin(self,numwrap=float):
        """Returns the time per bin"""
        return numwrap(self.getPeriod()) / numwrap(self.getNbin())
    def getDM(self):
        """Returns the header DM"""
        if self.params is None:
            return None
        return self.params.getDM()
    def getCoords(self,parse=True): #use get/set coorinates? Use astropy?
        """Returns the coordinate info in the header"""
        if parse:
            RA = tuple(map(float,self.header['RA'].split(":")))
            dec = tuple(map(float,self.header['DEC'].split(":")))
        else:
            RA = self.header['RA']
            dec = self.header['DEC']
        return RA,dec
    def getPulsarCoords(self,parse=True):
        """Returns the parsed coordinate info in the header"""
        return self.getCoords(parse=parse)
    def getTelescopeCoords(self):
        """Returns the telescope coordinates"""
        return self.header['ANT_X'],self.header['ANT_Y'],self.header['ANT_Z']
    def getBandwidth(self,header=False):
        """Returns the observation bandwidth"""
        if header:
            return self.header['OBSBW']
        else:
            return self.subintheader['CHAN_BW']*self.subintheader['NCHAN']
    def getDuration(self):
        """Returns the observation duratrion"""
        #return np.sum(self.subintinfo['TSUBINT']) #This is constant.
        return np.sum(self.getSubintinfo('TSUBINT')) #This is constant.
    def getDurations(self):
        """Returns the subintegration durations"""
        return self.durations
    def getCenterFrequency(self,weighted=False):
        """Returns the center frequency"""
        if weighted:
            return np.sum(self.freq*self.weights)/np.sum(self.weights)
        if "HISTORY" in self.keys:
            return self.history.getLatest('CTR_FREQ')
        else:
            return self.header['OBSFREQ'] #perhaps do an unweighted version from DAT_FREQ?
    def getTelescope(self):
        """Returns the telescope name"""
        return self.header['TELESCOP']
    def getFrontend(self):
        """Returns the frontend name"""
        return self.header['FRONTEND']
    def getBackend(self):
        """Returns the backend name"""
        return self.header['BACKEND']
    def getSN(self):
        """Returns the average pulse S/N"""
        return self.spavg.getSN()

    def isCalibrator(self):
        if self.header['OBS_MODE'] == CAL:
            return True
        return False



# Takes hdulist['HISTORY']
class History:
    def __init__(self,history):
        """Intializer"""
        self.header = dict()
        self.headerlist = history.header.keys()
        for key in self.headerlist:
            self.header[key] = history.header[key]
        self.dictionary = dict()
        self.namelist = list()
        for col in history.columns:
            self.namelist.append(col.name)
            self.dictionary[col.name] = (col.format,col.unit,list(col.array)) #make a np.array?
    def getValue(self,field,num=None):
        """Returns a dictionary array value for a given numeric entry"""
        if num is None:
            return self.dictionary[field][-1]
        else:
            return self.dictionary[field][-1][num]
    def getLatest(self,field):
        """Returns the latest key value"""
        return self.getValue(field,-1)
