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
import os
import sys
import gc as g
import decimal as d
Decimal = d.Decimal
import time
import warnings
from importlib import import_module
import inspect
import tempfile
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import astropy.coordinates as coordinates
import astropy.units as units
try:
    import astropy.io.fits as pyfits
except:
    import pyfits
import pypulse.calibrator as calib
Calibrator = calib.Calibrator
import pypulse.dynamicspectrum as DS
import pypulse.par as par
Par = par.Par
import pypulse.singlepulse as SP
import pypulse.utils as u

if sys.version_info.major == 2:
    fmap = map
elif sys.version_info.major == 3:
    fmap = lambda x, *args: list(map(x, *args))
    xrange = range

PSR = "PSR"
CAL = "CAL"
FON = "FON"
FOF = "FOF"
PCM = "PCM"
SEARCH = "SEARCH"

class Archive(object):
    def __init__(self, filename, prepare=True, lowmem=False, verbose=True,
                 weight=True, center_pulse=True, baseline_removal=True,
                 wcfreq=True, thread=False, cuda=False):
        ## Parse filename here
        self.pypulse_history = []
        self.record(inspect.currentframe())
        self.filename = str(filename) #fix unicode issue
        self.prepare = prepare
        self.lowmem = lowmem
        self.verbose = verbose
        self.center_pulse = center_pulse
        self.baseline_removal = baseline_removal
        self.wcfreq = wcfreq
        self.thread = thread
        self.cuda = cuda

        if verbose:
            print("Loading: %s" % self.filename)
            t0 = time.time()

        self.load(self.filename, prepare=prepare, center_pulse=center_pulse,
                  baseline_removal=baseline_removal, weight=weight,
                  wcfreq=wcfreq)
        if not self.lowmem:
            self.data_orig = np.copy(self.data)
            self.weights_orig = np.copy(self.weights)
        if verbose:
            t1 = time.time()
            print("Load time: %0.2f s" % (t1-t0))

        self.template = None

        #self.reset(False) #put prepare into here?, copying to arch is done here

        #if prepare:
        #    self.pscrunch()

        #if verbose and prepare:
        #    t2=time.time()
        #    print("Prep time: %0.2f s" % (t2-t1))

    def __repr__(self):
        return "Archive(%r, prepare=%r, lowmem=%r, verbose=%r)" % \
                (self.filename, self.prepare, self.lowmem, self.verbose)
    
    def __str__(self):
        return self.filename

    def load(self, filename, prepare=True, center_pulse=True,
             baseline_removal=True, weight=True, wcfreq=False):
        """
        Loads a PSRFITS file and processes
        http://www.atnf.csiro.au/people/pulsar/index.html?n=PsrfitsDocumentation.Txt
        """
        if filename is None: #Needed?
            filename = self.filename
        try:
            if self.lowmem:
                hdulist = pyfits.open(filename, ignore_missing_end=True, memmap=True)
            else:
                hdulist = pyfits.open(filename, ignore_missing_end=True)
        except IOError:
            print("Filename not found")
            raise SystemExit
        self.header = hdulist[0].header
        self.keys = fmap(lambda x: x.name, hdulist)
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
            nbin, nchan, npol, nsblk = fmap(int, hdulist['SUBINT'].columns[-1].dim[1:-1].split(","))

        if 'PSRPARAM' in self.keys:
            tablenames.remove('PSRPARAM')
            self.paramheaderlist = hdulist['PSRPARAM'].header.keys()
            self.paramheader = dict()
            for key in self.paramheaderlist:
                self.paramheader[key] = hdulist['PSRPARAM'].header[key]
            self.params = Par(fmap(lambda x: x[0], hdulist['PSRPARAM'].data), numwrap=float)
        elif 'PSREPHEM' in self.keys:
            tablenames.remove('PSREPHEM')
            paramkeys = fmap(lambda x: x.name, hdulist['PSREPHEM'].columns)
            paramvals = hdulist['PSREPHEM'].data[0]
            paramstrs = fmap(lambda x, y: "%s %s"%(x, y), paramkeys, paramvals)
            #self.paramheaderlist = hdulist['PSREPHEM'].header.keys()
            #self.paramheaderdict = dict()
            #for key in self.paramheaderlist:
            #    self.paramheaderdict[key] = hdulist['PSREPHEM'].header[key]
            self.params = Par(paramstrs, numwrap=float)
        else:
            self.params = None

        if 'POLYCO' in self.keys:
            tablenames.remove('POLYCO')
            self.polyco = Polyco(hdulist['POLYCO'], MJD=self.getMJD(full=True))
        else:
            self.polyco = None

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
            raise ValueError("This is a fluxcal file: currently not implemented")

        self.subintinfo = dict()
        self.subintinfolist = fmap(lambda x: x.name, hdulist['SUBINT'].columns[:-5])
        for i, column in enumerate(hdulist['SUBINT'].columns[:-5]):
            self.subintinfo[column.name] = (column.format,
                                            column.unit,
                                            hdulist['SUBINT'].data[column.name])
        self.subintheader = dict()
        self.subintheaderlist = hdulist['SUBINT'].header.keys()#for ordering
        for i, key in enumerate(hdulist['SUBINT'].header):
            self.subintheader[key] = hdulist['SUBINT'].header[key]

        DATA = hdulist['SUBINT'].data['DATA']
        if np.ndim(DATA) == 5:
            DATA = DATA[:, 0, :, :, :] #remove the nsblk column
        DATA = np.ascontiguousarray(DATA)

        #Definitions in Base/Formats/PSRFITS/ProfileColumn.C
        DAT_FREQ = hdulist['SUBINT'].data['DAT_FREQ']
        DAT_WTS = np.ascontiguousarray(hdulist['SUBINT'].data['DAT_WTS'])
        if not weight:
            DAT_WTS = np.ones(np.shape(DAT_WTS))
        DAT_SCL = np.ascontiguousarray(hdulist['SUBINT'].data['DAT_SCL'])
        DAT_OFFS = np.ascontiguousarray(hdulist['SUBINT'].data['DAT_OFFS'])# + 0.5 #testing
        # Enforce appropriate shapes in odd cases
        SHAPE = np.shape(DATA)
        if np.ndim(DAT_SCL) == 1:
            DAT_SCL = DAT_SCL.reshape(SHAPE[0], SHAPE[2])#SHAPE[1], SHAPE[2])
        if np.ndim(DAT_WTS) == 1:
            DAT_WTS = DAT_WTS.reshape(SHAPE[0], SHAPE[2])
        if np.ndim(DAT_OFFS) == 1:
            DAT_OFFS = DAT_OFFS.reshape(SHAPE[0], SHAPE[2])

        #self.DAT_SCL = DAT_SCL #testing

        #print DAT_WTS, np.max(DAT_WTS), np.min(DAT_WTS)
        if np.size(DAT_WTS) == 1:
            #DAT_WTS[0] = 1.0
            #DAT_WTS[0, 0] = 1.0
            self.weights = np.ones((1, 1))
        else:
            #DAT_WTS /= np.max(DAT_WTS) #close???
            #DAT_WTS *= 50
            self.weights = DAT_WTS
            self.weight_sum = np.sum(DAT_WTS)

        if self.lowmem: #Replace the data arrays with memmaps to reduce memory load
            SHAPE = np.shape(DATA)
            tfDATA = tempfile.NamedTemporaryFile()
            fp = np.memmap(tfDATA.name, dtype=np.int16, mode='w+', shape=SHAPE)
            fp[:] = DATA[:]
            del fp
            DATA = np.memmap(tfDATA.name, dtype=np.int16, mode='r', shape=SHAPE)

            '''
            SHAPE = np.shape(DAT_WTS)
            tfDAT_WTS = tempfile.NamedTemporaryFile()
            fp = np.memmap(tfDAT_WTS.name, dtype=np.float32, mode='w+', shape=SHAPE)
            fp[:] = DAT_WTS[:]
            del fp
            DAT_WTS = np.memmap(tfDAT_WTS.name, dtype=np.float32, mode='r', shape=SHAPE)

            SHAPE = np.shape(DAT_SCL)
            tfDAT_SCL = tempfile.NamedTemporaryFile()
            fp = np.memmap(tfDAT_SCL.name, dtype=np.float32, mode='w+', shape=SHAPE)
            fp[:] = DAT_SCL[:]
            del fp
            DAT_SCL = np.memmap(tfDAT_SCL.name, dtype=np.float32, mode='r', shape=SHAPE)

            SHAPE = np.shape(DAT_OFFS)
            tfDAT_OFFS = tempfile.NamedTemporaryFile()
            fp = np.memmap(tfDAT_OFFS.name, dtype=np.float32, mode='w+', shape=SHAPE)
            fp[:] = DAT_OFFS[:]
            del fp
            DAT_OFFS = np.memmap(tfDAT_OFFS.name, dtype=np.float32, mode='r', shape=SHAPE)
            '''

            tf = tempfile.NamedTemporaryFile()
            self._data = np.memmap(tf.name,
                                   dtype=np.float32,
                                   mode='w+',
                                   shape=(nsubint, npol, nchan, nbin))

        else:
            self._data = np.zeros((nsubint, npol, nchan, nbin))

        #self.data = np.zeros((nsubint, npol, nchan, nbin))
        #data = np.zeros((nsubint, npol, nchan, nbin))

        I = range(nsubint)
        J = range(npol)
        K = range(nchan)

        self.freq = DAT_FREQ

        if nsubint == 1 and npol == 1 and nchan == 1:
            self._data = (DAT_SCL*DATA+DAT_OFFS)#*DAT_WTS
        elif nsubint == 1 and npol == 1:
            for k in K:
                self._data[0, 0, k, :] = (DAT_SCL[0, k]*DATA[0, 0, k, :]+DAT_OFFS[0, k])#*DAT_WTS[0, k] #dat WTS[0]?
        elif nsubint == 1 and nchan == 1:              
            for j in J:
                self._data[0, j, 0, :] = (DAT_SCL[0, j]*DATA[0, j, 0, :]+DAT_OFFS[0, j])#*DAT_WTS[0]
        elif npol == 1 and nchan == 1:
            for i in I:
                self._data[i, 0, 0, :] = (DAT_SCL[i, 0]*DATA[i, 0, 0, :]+DAT_OFFS[i, 0])#*DAT_WTS[0]
        else: #if nsubint == 1 or npol == 1 or nchan == 1 this works,  or all three are not 1, might want to split this up
            t0 = time.time()
            cudasuccess = False
            if self.cuda:
                try:
                    gpuarray = import_module('pycuda.gpuarray')
                    compiler = import_module('pycuda.compiler')
                    driver = import_module('pycuda.driver')
                    autoinit = import_module('pycuda.autoinit')
                    cudasuccess = True
                except ImportError:
                    print("PyCUDA not imported")
                    #                __global__ void combine(float *retval, float *DAT_SCL, float *DATA, float *DAT_OFFS, int nbin)
                mod = compiler.SourceModule("""
                __global__ void combine(int16_t *retval, float *DAT_SCL, int16_t *DATA, float *DAT_OFFS, int nbin, int size)
                {
                    //uint idx = threadIdx.x + threadIdx.y*4;
                    uint xidx = blockDim.x * blockIdx.x + threadIdx.x;
                    uint yidx = blockDim.y * blockIdx.y + threadIdx.y;
                    uint zidx = blockDim.z * blockIdx.z + threadIdx.z;
                    uint idx = xidx + 5*yidx;
                    if (idx < size)
                        retval[idx] = 1;//DATA[idx];

                    //int subidx = idx/nbin;
                    //retval[idx] = ((DATA[idx] & 0x00FF) <<8) | ((DATA[idx]>>8) & 0x00ff); //swap-endian
                    //retval[idx] = (double)(DAT_SCL[subidx]*DATA[idx]+DAT_OFFS[subidx]);
                }
                """)
                combine = mod.get_function("combine")

                maxX = autoinit.device.get_attributes()[driver.device_attribute.MAX_BLOCK_DIM_X]
                maxY = autoinit.device.get_attributes()[driver.device_attribute.MAX_BLOCK_DIM_Y]
                maxZ = autoinit.device.get_attributes()[driver.device_attribute.MAX_BLOCK_DIM_Z]

                #combine(driver.Out(self.data),driver.In(np.ascontiguousarray(DAT_SCL)),driver.In(np.ascontiguousarray(DATA)),driver.In(np.ascontiguousarray(DAT_OFFS)),nbin,block=(4,4,1))
                data = np.zeros(np.shape(self.data), dtype='>i2')

                if nsubint <= maxX:
                    X = int(nsubint)
                    gridX = 1
                else:
                    pass
                # Assume maxZ >= 4 for now
                Z = int(npol)
                gridZ = 1
                if nchan <= maxY:
                    Y = int(nchan)
                    gridY = 1
                else:
                    pass

                print(np.shape(data), np.size(data))
                retval = np.zeros(np.shape(self.data), dtype='>i2')
                print(X, Y, Z)
                combine(driver.Out(retval), driver.In(DAT_SCL), driver.In(data), driver.In(DAT_OFFS), np.int32(nbin), np.int32(np.size(data)), block=(X, 1, 1), grid=(1, Z, Y))

                raise SystemExit

                #combine(driver.Out(DATA),driver.In(DAT_SCL),driver.In(DATA),driver.In(DAT_OFFS),nbin,block=(4,4,1))

                combine(driver.Out(retval), driver.In(DAT_SCL), driver.In(data), driver.In(DAT_OFFS), nbin, block=(4, 4, 4))
                #combine(data_gpu,driver.In(DAT_SCL),data_gpu,driver.In(DAT_OFF),nbin,block=(4,4,1))

                #driver.memcpy_dtoh(retval, data_gpu)
                #combine(driver.Out(data),driver.In(DAT_SCL),driver.In(DATA),driver.In(DAT_OFFS),nbin,block=(4,4,1))
                #print "Break1"
                #print retval,np.all(retval==256)
                #print np.shape(retval)
                #print len(np.where(retval==256)[0])
                #for i in retval:
                #    print i,
                #print "break"
                #print data.#dtype,DATA.dtype
                raise SystemExit

            if self.thread and not cudasuccess:
                def loop_func(i):
                    for j in J:
                        jnchan = j*nchan
                        for k in K:
                            self._data[i, j, k, :] = (DAT_SCL[i, jnchan+k]*DATA[i, j, k, :]+DAT_OFFS[i, jnchan+k])#*DAT_WTS[i, k]
                u.parmap(loop_func, I)
            elif not cudasuccess:
                nsub, npol, nchan, nbin = DATA.shape
                scale = DAT_SCL.reshape(nsub, npol, nchan)
                offset = DAT_OFFS.reshape(nsub, npol, nchan)
                weights = DAT_WTS.reshape(nsub, 1, nchan, 1)
                self._data = (scale*DATA.transpose((3, 0, 1, 2)) + offset).transpose((1, 2, 3, 0))
            t1 = time.time()
        self.data = self._data

        bw = self.getBandwidth()

        # All time-tagging info
        self.durations = self.getSubintinfo('TSUBINT')
        self.subint_starts = np.array(fmap(Decimal, self.getSubintinfo('OFFS_SUB')), dtype=np.dtype(Decimal))#+self.getTbin(numwrap=Decimal)*Decimal(nbin/2.0)#+self.getMJD(full=False, numwrap=Decimal) #converts center-of-bin times to start-of-bin times, in seconds, does not include the integer MJD part. This means that a template sitting in the center will have zero extra time

        # used to keep track of frequency-dependent channel delays,
        # in time units:
        self.channel_delays = np.zeros(nchan, dtype=np.dtype(Decimal))

        if prepare and not self.isCalibrator():
            self.pscrunch()
            self.dedisperse(wcfreq=wcfreq)

        self.calculateAverageProfile()

        if center_pulse and not self.isCalibrator() and prepare:
            #calibrator is not a pulse,
            # prepare must be run so that dedisperse is run?
            self.center()

        if baseline_removal and not self.isCalibrator():
            self.removeBaseline()

        hdulist.close()
        return

    def save(self, filename):
        """Save the file to a new FITS file"""

        primaryhdu = pyfits.PrimaryHDU(header=self.header) #need to make alterations to header
        hdulist = pyfits.HDUList(primaryhdu)

        if self.history is not None:
            cols = []
            for name in self.history.namelist:
                fmt, unit, array = self.history.dictionary[name]
                #print name, fmt, unit, array
                col = pyfits.Column(name=name, format=fmt, unit=unit, array=array)
                cols.append(col)
            historyhdr = pyfits.Header()
            for key in self.history.headerlist:
                historyhdr[key] = self.history.header[key]
            historyhdu = pyfits.BinTableHDU.from_columns(cols, name='HISTORY', header=historyhdr)
            hdulist.append(historyhdu)
            # Need to add in PyPulse changes into a HISTORY
        #else: #else start a HISTORY table

        if self.params is not None:
            #PARAM and not PSRPARAM?:
            cols = [pyfits.Column(name='PSRPARAM', format='128A', array=self.params.filename)]
            paramhdr = pyfits.Header()
            for key in self.paramheaderlist:
                paramhdr[key] = self.paramheader[key]
            paramhdu = pyfits.BinTableHDU.from_columns(cols, name='PSRPARAM')
            hdulist.append(paramhdu)
            # Need to include mode for PSREPHEM

        if self.polyco is not None:
            cols = []
            for name in self.polyco.namelist:
                fmt, unit, array = self.polyco.dictionary[name]
                #print name, fmt, unit, array
                col = pyfits.Column(name=name, format=fmt, unit=unit, array=array)
                cols.append(col)
            polycohdr = pyfits.Header()
            for key in self.polyco.headerlist:
                polycohdr[key] = self.polyco.header[key]
            polycohdu = pyfits.BinTableHDU.from_columns(cols, name='POLYCO', header=polycohdr)
            hdulist.append(polycohdu)

        if len(self.tables) > 0:
            for table in self.tables:
                hdulist.append(table)

        cols = []
        for name in self.subintinfolist:
            fmt, unit, array = self.subintinfo[name]
            col = pyfits.Column(name=name, format=fmt, unit=unit, array=array)
            cols.append(col)
            # finish writing out SUBINT!

        cols.append(pyfits.Column(name='DAT_FREQ', format='%iE'%np.shape(self.freq)[1], unit='MHz', array=self.freq)) #correct size? check units?
        cols.append(pyfits.Column(name='DAT_WTS', format='%iE'%np.shape(self.weights)[1], array=self.weights)) #call getWeights()

        nsubint, npol, nchan, nbin = self.shape(squeeze=False)
        
        DAT_OFFS = np.zeros((nsubint, npol*nchan), dtype=np.float32)
        DAT_SCL = np.zeros((nsubint, npol*nchan), dtype=np.float32)
        DATA = self.getData(squeeze=False, weight=False)
        saveDATA = np.zeros(self.shape(squeeze=False), dtype=np.int16)
        # Following Base/Formats/PSRFITS/unload_DigitiserCounts.C
        for i in xrange(nsubint):
            for j in xrange(npol):
                jnchan = j*nchan
                for k in xrange(nchan):
                    MIN = np.min(DATA[i, j, k, :])
                    MAX = np.max(DATA[i, j, k, :])
                    RANGE = MAX - MIN
                    if MAX == 0 and MIN == 0:
                        DAT_SCL[i, jnchan+k] = 1.0
                    else:
                        DAT_OFFS[i, jnchan+k] = 0.5*(MIN+MAX)
                        DAT_SCL[i, jnchan+k] = (MAX-MIN)/32766.0 #this is slightly off the original value? Results in slight change of data

                    saveDATA[i, j, k, :] = np.floor((DATA[i, j, k, :] - DAT_OFFS[i, jnchan+k])/DAT_SCL[i, jnchan+k] + 0.5) #why +0.5?

        cols.append(pyfits.Column(name='DAT_OFFS', format='%iE'%np.size(DAT_OFFS[0]), array=DAT_OFFS))
        cols.append(pyfits.Column(name='DAT_SCL', format='%iE'%np.size(DAT_SCL[0]), array=DAT_SCL))
        cols.append(pyfits.Column(name='DATA', format='%iI'%np.size(saveDATA[0]), array=saveDATA, unit='Jy', dim='(%s,%s,%s)'%(nbin, nchan, npol))) #replace the unit here

        subinthdr = pyfits.Header()
        for key in self.subintheaderlist:
            subinthdr[key] = self.subintheader[key]
        subinthdu = pyfits.BinTableHDU.from_columns(cols, name='SUBINT', header=subinthdr)
        hdulist.append(subinthdu)

        hdulist.writeto(filename, clobber=True)#clobber=True?

    def unload(self, filename):
        return self.save(filename)

    def gc(self):
        """Manually clear the data cube for python garbage collection"""
        """ This is broken with the weights change """
        if self.verbose:
            t0 = time.time()
        self._data = None
        self.data = None
        self.data_orig = None
        self.weights = None
        self.weights_orig = None
        if self.verbose:
            t1 = time.time()
            print("Unload time: %0.2f s" % (t1-t0))
        g.collect()

    def shape(self, squeeze=True):
        """Return the current shape of the data array"""
        return np.shape(self.getData(squeeze=squeeze))

    def reset(self, prepare=True):
        """Replace the arch with the original clone"""
        self.record(inspect.currentframe()) #temporary, actually change the history!
        if self.lowmem:
            self.load(self.filename, prepare=prepare)
        else:
            self.data = np.copy(self.data_orig)
            self.weights = np.copy(self.weights_orig)
        self.durations = self.getSubintinfo('TSUBINT')
        #if prepare:
        #    self.scrunch()

    def scrunch(self, arg='Dp', **kwargs):
        """average the data cube along different axes"""
        self.record(inspect.currentframe())
        if 'T' in arg:
            self.data[0, :, :, :] = np.mean(self.data, axis=0)
            self.data = self.data[0:1, :, :, :] #resize
            self.weights[0, :] = np.mean(self.weights, axis=0) #should be sum?
            self.weights = self.weights[0:1, :] #resize
            self.durations = np.array([self.getDuration()])
        if 'p' in arg:
            self.pscrunch() #throw this the other way
        if 'F' in arg:
            self.data[:, :, 0, :] = np.mean(self.data, axis=2)
            self.data = self.data[:, :, 0:1, :]
            self.weights[:, 0] = np.mean(self.weights, axis=1) #problem?
            self.weights = self.weights[:, 0:1]
        if 'D' in arg:
            if "wcfreq" in kwargs:
                self.dedisperse(wcfreq=kwargs['wcfreq'])
            else:
                self.dedisperse(wcfreq=self.wcfreq)
        if 'B' in arg:
            self.data[:, :, :, 0] = np.mean(self.data, axis=3)
            self.data = self.data[:, :, :, 0:1]
        return self

    def tscrunch(self, nsubint=None, factor=None):
        """average the data cube along the time dimension"""
        #if nsubint == 1 or (factor is None and nsubint is None):
        #    return self.scrunch('T')
        if factor is None and nsubint is None:
            nsubint = 1
        if factor == 1:
            return self
        if factor is None and nsubint is not None:
            factor = self.getNsubint()//nsubint
            if self.getNsubint()%nsubint != 0:
                factor += 1
        self.record(inspect.currentframe())

        nsubint, npol, nchan, nbin = self.shape(squeeze=False)

        retval = np.zeros((len(np.r_[0:nsubint:factor]), npol, nchan, nbin))
        newdurations = np.zeros(np.shape(retval)[0])

        weightsum = np.sum(self.weights)
        weightretval = np.zeros((len(np.r_[0:nsubint:factor]), nchan))

        newnsubint = nsubint//factor
        for i in xrange(newnsubint):
            weightretval[i, :] = np.sum(self.weights[i*factor:(i+1)*factor, :], axis=0)

        for i in xrange(newnsubint):
            newdurations[i] += np.sum(self.durations[i*factor:(i+1)*factor])
            for j in xrange(npol):
                for k in xrange(nchan):
                    for l in xrange(nbin):
                        retval[i, j, k, l] = np.sum(self.data[i*factor:(i+1)*factor, j, k, l] * self.weights[i*factor:(i+1)*factor, k])

        self.weights = weightretval
        self.data = retval/weightsum
        self.durations = newdurations

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
        self.record(inspect.currentframe())
        if self.subintheader['POL_TYPE'] == "AABBCRCI": #Coherence:
            A = self.data[:, 0, :, :]
            B = self.data[:, 1, :, :]
            self._data[:, 0, :, :] = A+B
        elif self.subintheader['POL_TYPE'] == "IQUV": #Stokes
            I = self.data[:, 0, :, :] #No need to average the other dimensions?
            self.data[:, 0, :, :] = I
        self.data = self.data[:, 0:1, :, :] #keeps the shape
        return self

    def fscrunch(self, nchan=None, factor=None):
        """average the data cube along the frequency dimension"""
        #if nchan == 1 or (factor is None and nchan is None):
        #    return self.scrunch('F')
        if factor is None and nchan is None:
            nchan = 1
        if factor == 1:
            return self
        if factor is None and nchan is not None:
            factor = self.getNchan()//nchan
            if self.getNchan()%nchan != 0:
                factor += 1
        self.record(inspect.currentframe())

        nsubint, npol, nchan, nbin = self.shape(squeeze=False)
        
        retval = np.zeros((nsubint, npol, len(np.r_[0:nchan:factor]), nbin))

        weightsum = np.sum(self.weights)
        weightretval = np.zeros((nsubint, len(np.r_[0:nchan:factor])))


        newnchan = nchan//factor

        # Force the frequency axis to be 1D
        if np.ndim(self.freq) == 1:
            freq = self.freq
        else:
            freq = self.freq[0]
        newfreq = np.zeros(newnchan) #this will only be 1D

        for k in xrange(newnchan):
            weightretval[:, k] = np.sum(self.weights[:, k*factor:(k+1)*factor], axis=1)
            newfreq[k] = np.sum(freq[k*factor:(k+1)*factor])/float(factor) #unweighted!

        for i in xrange(nsubint):
            for j in xrange(npol):
                for k in xrange(newnchan):
                    for l in xrange(nbin):
                        retval[i, j, k, l] = \
                        np.sum(self.data[i, j, k*factor:(k+1)*factor, l] * self.weights[i, k*factor:(k+1)*factor])


        self.weights = weightretval
        self.data = retval/weightsum
        self.freq = newfreq
        return self

    def bscrunch(self, nbins=None, factor=None):
        """average the data cube along the phase dimension"""
        if nbins == 1 or (factor is None and nbins is None):
            return self.scrunch('B')
        if factor == 1:
            return self
        if factor is None and nbins is not None:
            factor = self.getNbin()//nbins
            if self.getNbin()%nbins != 0:
                factor += 1
        else:
            self.record(inspect.currentframe())
            nsubint, npol, nchan, nbin = self.shape(squeeze=False)
            
            retval = np.zeros((nsubint, npol, nchan, len(np.r_[0:nbin:factor])))
            counts = np.zeros_like(retval)
            for i in xrange(factor):
                arr = self.data[:, :, :, i:nbin:factor]
                count = np.ones_like(arr)
                length = np.shape(arr)[3]
                retval[:, :, :, :length] += arr
                counts[:, :, :, :length] += count
            retval = retval/counts
            self.data = retval
        return self

    def dedisperse(self, DM=None, barycentric=True, reverse=False, wcfreq=False):
        """
        De-disperse the pulses
        if DM is given, use this value to compute the time_delays
        """
        nchan = self.getNchan()
        if nchan == 1: #do not dedisperse?
            return self
        self.record(inspect.currentframe())
        Faxis = self.getAxis('F', wcfreq=wcfreq)

        Faxis = self.freq #potentially two-dimensional thing?

        nsubint, npol, nchan, nbin = self.shape(squeeze=False)
        if DM is None:
            DM = self.getDM()
        cfreq = self.getCenterFrequency(weighted=wcfreq)

        K = 4.149e3
        K = 1.0/2.41e-4 #constant used to be more consistent with PSRCHIVE
        Kconst = 1.0/2.41e-4 #constant used to be more consistent with PSRCHIVE
        time_delays = K*DM*(cfreq**(-2) - np.power(Faxis, -2)) #freq in MHz, delays in seconds

        #time_delays = Decimal(K)*Decimal(DM)*(Decimal(str(cfreq))**(-2) - np.array(fmap(lambda x: Decimal(str(x))**-2,Faxis)))

        #print time_delays

        dt = self.getTbin(numwrap=float)
        '''
        bin_delays = np.array(fmap(lambda x: Decimal(str(x)),time_delays)) / dt
        #print "foo",bin_delays,nbin,dt,self.getPeriod(),self.getNbin()
        bin_delays = bin_delays % Decimal(nbin)
        '''
        if reverse:
            sign = 1
        else:
            sign = -1
        #time_delays *= (-1*sign)

        #freq in MHz, delays in seconds
        time_delay = Kconst*DM*(cfreq**(-2) - Faxis[:, None, :]**(-2))
        bin_delay = time_delay / dt
        bin_delay = bin_delay % nbin
        self.data = u.shiftit(self.data, sign*bin_delay)
        self.calculateAverageProfile() #re-calculate the average profile
        return self

    def dededisperse(self, DM=None, barycentric=True, wcfreq=False):
        """
        Remove the dedispersion of the pulses
        Note: Errors might propagate?
        """
        self.dedisperse(DM=DM, barycentric=barycentric, reverse=True, wcfreq=wcfreq)
        return self

    def calculateAverageProfile(self):
        """
        Averages the data along the time/polarization/frequency axes
        """
        self.average_profile = np.mean(np.mean(self.getData(squeeze=False), axis=2), axis=0) #This may not be the appropriate weighting
        if np.shape(self.average_profile)[0] != 1: #polarization add
            if self.subintheader['POL_TYPE'] == "AABBCRCI": #Coherence
                self.average_profile = (self.average_profile[0, :] +
                                        self.average_profile[1, :])
            elif self.subintheader['POL_TYPE'] == "IQUV": #Stokes
                self.average_profile = self.average_profile[0, :]
        else:
            self.average_profile = self.average_profile[0, :]
        self.calculateOffpulseWindow()
        # Remove baseline. A bit organizationally strange
        self.spavg.remove_baseline()
        self.average_profile = self.spavg.data
        return self.average_profile

    def calculateOffpulseWindow(self):
        """
        Automatically calculates the off-pulse window
        for the average SP
        """
        self.spavg = SP.SinglePulse(self.average_profile, windowsize=int(self.getNbin()//8))
        self.opw = self.spavg.opw
        return self.opw

    def calculateTemplate(self, mode='vonmises', sigma=None, lam=None,
                          window_length=11, polyorder=5, **kwargs):
        """
        Calculate a template shape
        """
        if mode == "vonmises":
            template = self.spavg.vonmises_smoothing(**kwargs)
        elif mode == "gaussian":
            template = self.spavg.gaussian_smoothing(**kwargs)
        elif mode == "spline":
            template = self.spavg.spline_smoothing(sigma=sigma, lam=lam, **kwargs)
        elif mode == "savgol":
            template = self.spavg.savgol_smoothing(window_length=window_length, polyorder=polyorder)
        else:
            template = None
            self.template = None
        if template is not None:
            self.template = SP.SinglePulse(template, windowsize=int(self.getNbin()//8))
            self.opw = self.template.opw
        return self.template

    def superprep(self):
        """
        This will do pscrunching, centering, and dedispersing all at once to avoid multiple loops?
        Does not work so far.
        """
        nsubint, npol, nchan, nbin = self.shape(squeeze=False)

        center_bin = int(nbin*phase_offset)
        maxind = np.argmax(self.average_profile)
        diff = center_bin - maxind

        Faxis = self.getAxis('F', wcfreq=wcfreq)
        DM = self.getDM()
        cfreq = self.getCenterFrequency()
        time_delays = 4.149e3*DM*(cfreq**(-2) - np.power(Faxis, -2)) #DM in MHz, delays in seconds
        bin_delays = (time_delays / self.getPeriod())*nbin
        bin_delays = bin_delays % nbin
        if reverse:
            sign = 1
        else:
            sign = -1


        for i in xrange(nsubint):
            for j in xrange(npol):
                for k in xrange(nchan):
                    temp = self.data[i, j, k, :]
                    temp = u.shiftit(temp, sign*delay)
                    self.data[i, j, k, :] = np.roll(temp, diff)
        self.average_profile -= np.mean(self.average_profile[self.spavg.opw])
        return self


        self.average_profile = np.roll(self.average_profile, diff)
        self.calculateOffpulseWindow()
        return self

    def center(self, phase_offset=0.5):
        """
        Center the peak of the pulse in the middle of the data arrays.
        """
        self.record(inspect.currentframe())
        nsubint, npol, nchan, nbin = self.shape(squeeze=False)

        if phase_offset >= 1 or phase_offset <= 0:
            phase_offset = 0.5
        center_bin = int(nbin*phase_offset)
        maxind = np.argmax(self.average_profile)
        diff = center_bin - maxind

        self.data = np.roll(self.data, diff, axis=-1)
        self.average_profile = np.roll(self.average_profile, diff)
        #print "diff", diff, diff*self.getTbin()
        self.channel_delays += Decimal(str(-1*diff*self.getTbin())) #this is unnecessary? FIX THIS
        self.calculateOffpulseWindow()
        return self

    def removeBaseline(self):
        """Removes the baseline of the pulses given an offpulse window"""
        self.record(inspect.currentframe())
        nsubint, npol, nchan, nbin = self.shape(squeeze=False)

        baseline = np.mean(self.data[..., self.spavg.opw], axis=-1)
        self.data -= baseline[..., np.newaxis]
        self.average_profile -= np.mean(self.average_profile[self.spavg.opw])
        return self
    remove_baseline = removeBaseline

    def getLevels(self, differences=False):
        """ Returns calibration levels if this is a calibrator"""
        if not self.isCalibrator():
            print("Not a calibration file")
            return
        # Ensure data are scrunched in time,
        # or ignore this and simply calculate the weighted time average?
        self.tscrunch()

        #Pre-define
        data = self.getData()
        npol = self.getNpol()
        nchan = self.getNchan()
        nbin = self.getNbin()

        # Check header info CAL_DCYC, CAL_NPHS, etc, to determine on-diode
        # or take an absolute value?
        first = np.mean(data[0, :, :nbin/2])
        second = np.mean(data[0, :, nbin/2:])
        if first > second:
            highinds = np.arange(0, nbin/2)
            lowinds = np.arange(nbin/2, nbin)
        else:
            lowinds = np.arange(0, nbin/2)
            highinds = np.arange(nbin/2, nbin)

        # Calculate calibrations
        freqs = self.getAxis('F')
        if differences:
            caldata = np.zeros((npol, nchan))
            calerrs = np.zeros((npol, nchan))
            for i in xrange(npol):
                for j in xrange(nchan):
                    caldata[i, j] = np.mean(data[i, j, highinds]) - np.mean(data[i, j, lowinds])
                    calerrs[i, j] = np.sqrt(np.std(data[i, j, highinds])**2 /
                                            len(highinds) + np.std(data[i, j, lowinds])**2 / len(lowinds))
        else:
            caldatalow = np.zeros((npol, nchan))
            caldatahigh = np.zeros((npol, nchan))
            calerrslow = np.zeros((npol, nchan))
            calerrshigh = np.zeros((npol, nchan))
            for i in xrange(npol):
                for j in xrange(nchan):
                    caldatalow[i, j] = np.mean(data[i, j, lowinds])
                    caldatahigh[i, j] = np.mean(data[i, j, highinds])
                    calerrslow[i, j] = np.std(data[i, j, lowinds]) / np.sqrt(len(lowinds))
                    calerrshigh[i, j] = np.std(data[i, j, highinds]) / np.sqrt(len(highinds))

        if differences:
            return freqs, caldata, calerrs
        return freqs, caldatalow, caldatahigh, calerrslow, calerrshigh

    def calibrate(self, psrcalar, fluxcalonar=None, fluxcaloffar=None):
        """Calibrates using another archive"""
        self.record(inspect.currentframe())
        if (not (isinstance(psrcalar, Calibrator) or
                 (isinstance(psrcalar, Archive) and psrcalar.isCalibrator()))):
            raise ValueError("Require calibration archive")
        # Check if cals are appropriate?

        if isinstance(psrcalar, Calibrator):
            cal = psrcalar
        else:
            # Calculate calibration levels
            psrcalfreqs, psrcaldata, psrcalerrs = psrcalar.getLevels(differences=True)

            # Check if cal has the correct dimensions,  if not perform interpolation
            freqs = self.getAxis('F')
            if len(freqs) != len(psrcalfreqs):
                pass

            cal = Calibrator(psrcalfreqs, psrcaldata, psrcalerrs)
        #if fluxcalon is not None:
        #    fluxcaloncal = Calibrator(fluxcalonfreqs, fluxcalondata, fluxcalonerrs)
        #    fluxcaloffcal = Calibrator(fluxcalofffreqs, fluxcaloffdata, fluxcalofferrs)

        if fluxcalonar is not None and fluxcaloffar is not None:
            fluxcaldata = np.zeros((npol, nchan))
            for i in xrange(npol):
                for j in xrange(nchan):
                    fluxcaldata[i, j] = (np.mean(fdata[i, j, highinds]) -
                                         np.mean(fdata[i, j, lowinds]))

        cal.applyCalibration(self)
        return cal
        # Apply calibrations

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        self.weighted_data = value*self.weights[:,None,:,None]

    def getData(self, squeeze=True, setnan=None, weight=True):
        """Returns the data array,  fully squeezed"""
        if weight:
            data = self.weighted_data
        else:
            data = self._data

        if squeeze:
            data = data.squeeze()

        if setnan is not None:
            data = np.where(data == setnan, np.nan, data)
        
        #return np.copy(data) #removes pointer to data
        return data
    
    def setData(self, newdata):
        """Sets the data,  very dangerous!"""
        self.record(inspect.currentframe())
        if np.shape(newdata) == np.shape(self.data):
            self.data = np.copy(newdata)

    def getWeights(self, squeeze=True):
        """ Return copy of weights array """
        weights = self.weights
        if squeeze:
            weights = weights.squeeze()
        return np.copy(weights)

    def setWeights(self, val, t=None, f=None):
        """
        Set weights to a certain value
        Can be used for RFI routines
        """
        self.record(inspect.currentframe())
        if t is None and f is None:
            self.weights[:, :] = val
        elif t is None:
            self.weights[:, f] = val
        elif f is None:
            self.weights[t, f] = val
        else:
            self.weights[t, f] = val

    def saveData(self, filename=None, ext='npy', ascii=False, outdir=None):
        """Save the data array to a different format"""
        if filename is None:
            filename = self.filename
            filename = ".".join(filename.split(".")[:-1])+"."+ext

        if outdir is not None:
            filename = outdir + "/" + filename.split("/")[-1]

        if self.verbose:
            print("Saving: %s" % filename)
        if ascii:
            nsubint, npol, nchan, nbin = self.shape(squeeze=False)
            output = ""
            if nsubint == 1 and npol == 1 and chan == 1:
                np.savetxt(filename, self.getData())
                return
            elif ((nsubint == 1 and npol == 1) or
                  (nsubint == 1 and nchan == 1) or
                  (npol == 1 and nchan == 1)):
                np.savetxt(filename, self.getData())
                return
            for i in xrange(nsubint):
                for j in xrange(npol):
                    for k in xrange(nchan):
                        for l in xrange(nbin):
                            output += "%i %i %i %i %.18e\n" % (i, j, k, l, self.data[i, j, k, l])
            with open(filename, 'w') as FILE:
                FILE.write(output)
        else:
            np.save(filename, self.getData())
        return

    def outputPulses(self, filename):
        """ Write out a .npy file"""
        np.save(filename, self.getData())
        return

    def getAxis(self, flag=None, edges=False, wcfreq=False, datfreq=True):
        """
        Get F/T axes for plotting
        If edges: do not return centers for each. Better for imshow plotting because of extents.
        """
        if flag == 'T':
            durations = self.durations
            csum = np.cumsum(durations)
            edgearr = np.concatenate(([0], csum))
            if edges:
                return edgearr
            else: #centeredg
                return csum-np.diff(edgearr)/2.0
        elif flag == 'F':
            if np.ndim(self.freq) == 1:
                return self.freq
            if self.getNchan() == len(self.freq[0]):
                return self.freq[0] #return self.getSubintinfo('DATFREQ')[0]  ### This block is a temporary replacement

            nchan = self.getNchan()
            fc = self.getCenterFrequency(weighted=wcfreq)
            bw = self.getBandwidth()
            df = np.abs(bw)/nchan

            if edges:
                arr = np.array((np.arange(nchan+1) - (nchan+1)/2.0 + 0.5)*df + fc)
            else:
                #note: unweighted frequencies!
                arr = np.array((np.arange(nchan) - nchan/2.0 + 0.5)*df + fc)


            if bw < 0.0:
                return arr[::-1] #???
            return arr

        else: #do both?
            pass

    def getFrequencies(self, **kwargs):
        """Convenience function for getAxis"""
        return getAxis('F', **kwargs)

    def getTimes(self, **kwargs):
        """Convenience function for getAxis"""
        return getAxis('T', **kwargs)

    #Assumes the shape of data is (t, f, b) (i.e. polarization scrunched)
    def getPulse(self, t, f=None):
        """Get pulse(t, f). If f==None,  get pulse(t)"""
        if f is None:
            if self.shape(squeeze=False)[2] == 1:
                return self.getData()[t, :]
            return np.mean(self.getData()[t, :, :], axis=0)
        return self.getData()[t, f, :]

    # Assumes it is calibrated
    # Better to replace with SinglePulse's fitPulse
    def getPeakFlux(self, t, f=None):
        """Return the maximum value of the pulses, not typically used"""
        pulse = self.getPulse(t, f)
        return np.max(pulse)

    def getIntegratedFlux(self, t, f=None):
        """Return the integrated value of the pulses, not typically used"""
        pulse = self.getPulse(t, f)
        return np.trapz(pulse)

    def getSinglePulses(self, func=None, windowsize=None, **kwargs):
        """Efficiently wraps self.data with SP.SinglePulse"""
        if func is None:
            func = lambda x: x
        newshape = self.shape()[:-1]
        data = self.getData() #properly weighted
        period = self.getPeriod()
        if newshape == ():
            return SP.SinglePulse(func(data), period=period, windowsize=windowsize, **kwargs)
        retval = np.empty(newshape, dtype=np.object)
        for ind in np.ndindex(newshape):
            pulse = func(data[ind])
            retval[ind] = SP.SinglePulse(pulse, period=period, windowsize=windowsize, **kwargs)
        return retval

    #Given a list of numbers corresponding to the arguments returned
    def fitPulses(self, template, nums, flatten=False, func=None, windowsize=None, **kwargs):
        """Fit all of the pulses with a given template"""
        if len(template) != self.getNbin():
            raise IndexError("Improper template size")
        nums = np.array(nums)
        if windowsize is not None:
            sptemp = SP.SinglePulse(template, windowsize=windowsize)
            opw = sptemp.opw
            kwargs["opw"] = opw #apply this windowing to alll single pulses
        sps = self.getSinglePulses(func=func, **kwargs)

        if np.shape(sps) == (): #single pulse
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
    def getDynamicSpectrum(self, window=None, template=None, mpw=None,
                           align=None, windowsize=None, weight=True,
                           verbose=False, snr=False, maketemplate=False):
        """
        Return the dynamic spectrum
        window: return the dynamic spectrum using only a certain phase bins
        Should use a numpy array for this
        When thrown into imshow, tranpose puts frequency on the y-axis, time on the x
        """
        fullshape = self.shape(squeeze=False)
        #requires polarization scrunch for now:
        if fullshape[0] != 1 and fullshape[1] == 1 and fullshape[2] != 1:
            bw = self.getBandwidth()
            data = self.getData()
            shape = self.shape()

            if bw < 0:
                wrapfunc = lambda x: np.transpose(x) #do not flipud?
            else:
                wrapfunc = lambda x: np.transpose(x)

            if template is None and maketemplate:
                if self.template is None:
                    template = self.calculateTemplate()
                else:
                    template = self.template

            Fedges = self.getAxis('F', edges=True)
            Tedges = self.getAxis('T', edges=True)

            if template is not None:
                if isinstance(template, SP.SinglePulse):
                    sptemp = template
                elif isinstance(template, Archive): #use windowsize set to the default?
                    sptemp = SP.SinglePulse(u.center_max(u.normalize(template.getData(), simple=True)),
                                            windowsize=template.getNbin()//8)
                elif isinstance(template, np.ndarray):
                    sptemp = SP.SinglePulse(u.center_max(u.normalize(template, simple=True)),
                                            windowsize=len(template)//8)
                elif isinstance(template, str):
                    artemp = Archive(template, verbose=False)
                    sptemp = SP.SinglePulse(u.center_max(u.normalize(artemp.getData(), simple=True)),
                                            windowsize=template.getNbin()//8)

                if mpw is not None: #best way to handle this now?
                    sptemp.mpw = mpw
                elif windowsize is not None: #redo the windowsize
                    if windowsize >= len(sptemp.data):
                        raise IndexError("Improper window size")
                    sptemp = SP.SinglePulse(sptemp.data, windowsize=windowsize)

                gs = np.zeros((fullshape[0], fullshape[2]))
                offs = np.zeros((fullshape[0], fullshape[2]))
                sig_gs = np.zeros((fullshape[0], fullshape[2]))
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
                        sp = SP.SinglePulse(data[i], opw=sptemp.opw, align=align)
                        baseline = sp.getOffpulseNoise(mean=True) #get mean value of offpulse
                        spfit = sp.fitPulse(sptemp.data)
                        if spfit is not None:
                            gs[i] = spfit[ind] #bhat
                            offs[i] = baseline
                            sig_gs[i] = spfit[4]
                    gs, offs, sig_gs = wrapfunc(gs), wrapfunc(baseline), wrapfunc(sig_gs)
                else:
                    for i in I:
                        if verbose:
                            print("%i,%i"%(i, I[-1]))
                        for j in J:
                            sp = SP.SinglePulse(data[i, j], opw=sptemp.opw, align=align)
                            baseline = sp.getOffpulseNoise(mean=True) #get mean value of offpulse
                            spfit = sp.fitPulse(sptemp.data)
                            #if spfit is None:
                            #    print i,j, np.shape(data)
                            #    plt.plot(data[i,j])
                            #    plt.show()
                            #    raise SystemExit
                            if spfit is not None:
                                gs[i, j] = spfit[ind] #bhat
                                offs[i, j] = baseline
                                sig_gs[i, j] = spfit[4]
                    gs, offs, sig_gs = wrapfunc(gs), wrapfunc(offs), wrapfunc(sig_gs)
                #return wrapfunc(gs), wrapfunc(offs), wrapfunc(sig_gs)
                return DS.DynamicSpectrum(gs, offs, sig_gs, F=Fedges, T=Tedges)

            #kind of hard wired
            if window is None:
                return DS.DynamicSpectrum(wrapfunc(np.mean(data, axis=2)), F=Fedges, T=Tedges)
            else:
                return DS.DynamicSpectrum(wrapfunc(np.mean(data[:, :, window], axis=2)), F=Fedges, T=Tedges)

    def plot(self, ax=None, show=True):
        """Basic plotter of data"""
        data = self.getData()
        if len(np.shape(data)) == 1:
            if ax is None:
                plt.plot(data, 'k')
                plt.xlim(0, len(data))
            else:
                ax.plot(data, 'k')
                ax.set_xlim(0, len(data))
            if show:
                plt.show()
        else:
            print("Invalid dimensions")
    def imshow(self, ax=None, cbar=False, mask=None, show=True,
               filename=None, setnan=0.0, **kwargs):
        """Basic imshow of data"""
        data = self.getData(setnan=setnan)
        if len(np.shape(data)) == 2:
            if mask is not None:
                u.imshow(ma.masked_array(data, mask=mask), ax=ax, **kwargs)
            else:
                u.imshow(data, ax=ax, **kwargs)
            if cbar:
                plt.colorbar()
            if filename is not None:
                plt.savefig(filename)
            if show:
                plt.show()
            else:
                plt.close()
        else:
            raise IndexError("Invalid dimensions for plotting")
        return ax

    def pavplot(self, ax=None, mode="GTpd", show=True, wcfreq=True):
        """Produces a pav-like plot for comparison"""
        data = self.getData(setnan=0.0)
        if len(np.shape(data)) == 2:
            shape = self.shape(squeeze=False)
            if ax is None:
                fig = plt.figure()
                ax = fig.add_subplot(111)
            cmap = plt.cm.afmhot
            cmap.set_bad(color='k', alpha=1.0)
            if shape[0] == 1 and shape[1] == 1: #fix this to match mode
                Fedges = self.getAxis('F', edges=True) #is this true?
                u.imshow(self.getData(), ax=ax, extent=[0, 1, Fedges[0], Fedges[-1]], cmap=cmap)
                ax.set_xlabel("Pulse Phase")
                ax.set_ylabel("Frequency (MHz)")
                ax.set_title("%s %s\nFreq %0.3f MHz BW: %0.3f Length %0.3f S/N %0.3f"%(self.getName(), self.filename, self.getCenterFrequency(weighted=wcfreq), self.getBandwidth(), self.getDuration(), self.getSN()))#get the basename?
                ax2 = ax.twinx()
                ax2.set_ylim(0, self.getNchan())
                ax2.set_ylabel("Index")
                if show:
                    plt.show()
            if shape[2] == 1 and shape[1] == 1: #fix this to match mode
                Tedges = self.getAxis('T', edges=True) #is this true?
                u.imshow(self.getData(), ax=ax, extent=[0, 1, Tedges[0], Tedges[-1]], cmap=cmap)
                ax.set_xlabel("Pulse Phase")
                ax.set_ylabel("Time") #units
                #ax.set_title("%s %s\nFreq %0.3f MHz BW: %0.3f Length %0.3f S/N %0.3f"%(self.getName(),self.filename,self.getCenterFrequency(weighted=wcfreq),self.getBandwidth(),self.getDuration(),self.getSN()))#get the basename?
                ax2 = ax.twinx()
                ax2.set_ylim(0, self.getNchan())
                ax2.set_ylabel("Index")
                if show:
                    plt.show()
        else:
            print("Invalid dimensions")
        return ax

    def joyDivision(self, border=0.1, labels=False, album=True, **kwargs):
        """Calls waterfall() in the style of the Joy Division album cover"""
        return self.waterfall(border=border, labels=labels, album=album, **kwargs)

    def waterfall(self, offset=None, border=0, labels=True, album=False, bins=None, show=True):
        """
        Joy Division plot of data, works like imshow
        Can be slow for many calls of plot!
        """
        data = self.getData(squeeze=True)
        if len(np.shape(data)) == 2:
            if offset is None:
                offset = np.max(np.average(data, axis=0))#*0.5# * 2.10 #?

            fig = plt.figure(figsize=(6, 6))
            if album:
                bgcolor = 'black'
                ax = fig.add_subplot(111, facecolor=bgcolor) #axisbg
                color = 'w'
            else:
                bgcolor = 'w'#hite'
                ax = fig.add_subplot(111)
                color = 'k'

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
                for i in range(len(data)-1, -1, -1):
                    z += 1
                    y = data[i][bins]+offset*i
                    #y = np.roll(y, 100*i) # for testing

                    ax.plot(y, color, zorder=z)

                    ax.set_xlim(XLOW, XHIGH)
                    ax.set_ylim(YLOW, YHIGH)
                    ax.fill_between(x, y, where=(y >= YLOW), color=bgcolor, zorder=z) #testing

            else:
                for i in range(len(data)):
                    ax.plot(data[i][bins]+offset*i, color)

            ax.set_xlim(XLOW, XHIGH)
            ax.set_ylim(YLOW, YHIGH)

            if not labels:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            if show:
                plt.show()
        else:
            print("Invalid dimensions")

    ### NOTE: THIS NEEDS TO BE CHECKED WITH THE NEW CHANGES ###

    def time(self, template, filename, MJD=False, simple=False, wcfreq=False,
             flags="", appendto=False, overridecfreq=0.0, **kwargs):
        """
        Times the pulses and outputs in the tempo2_IPTA format similar to pat.
        MJD: if True, return TOAs in MJD units, else in time units corresponding to a bin number
        """

        if isinstance(template, Archive):
            artemp = template
            tempname = artemp.filename
            template = artemp.getData()
        elif isinstance(template, str):
            tempname = template
            artemp = Archive(tempname)
            template = artemp.getData()
        elif isinstance(template, (list, np.ndarray)):
            tempname = "None"
        else:
            return

        #template = u.shiftit(template, -4.8)
        rollval, template = u.center_max(u.normalize(template, simple=True),
                                         full=True) # enforces a good fit
        #print artemp.channel_delays[0]
        #print "roll", type(rollval)

        #If given an offpulse, use that, else calculate a pre-defined one in the template Archive
        if "opw" in kwargs.items():
            opw = (kwargs['opw'] + rollval)%len(kwargs['opw']) # "roll" the array with the template
        else:
            sptemp = SP.SinglePulse(template, windowsize=len(template)//8)
            kwargs['opw'] = sptemp.opw

        tauhat, bhat, sigma_tau, sigma_b, snrs = \
        self.fitPulses(template, [1, 2, 3, 4, 5], **kwargs) #tauhat is a relative shift
        Taxis = self.getAxis('T')
        Faxis = self.getAxis('F', wcfreq=wcfreq)
        if overridecfreq != 0.0:
            fc = self.getCenterFrequency(weighted=wcfreq)
            Faxis = Faxis - fc + overridecfreq

        #Reshape if necessary
        tauhat = tauhat.reshape(len(Taxis), len(Faxis))
        bhat = bhat.reshape(len(Taxis), len(Faxis))
        sigma_tau = sigma_tau.reshape(len(Taxis), len(Faxis))
        sigma_b = sigma_b.reshape(len(Taxis), len(Faxis))
        snrs = snrs.reshape(len(Taxis), len(Faxis))

        telescope = self.getTelescope() #lowercase? This may cause tempo2 errors
        frontend = self.getFrontend()
        backend = self.getBackend()
        bw = np.abs(self.getBandwidth())
        nchan = self.getNchan()
        chanbw = bw / nchan
        nbin = self.getNbin()

        dt = self.getTbin()

        #plt.plot(template*np.max(self.getData()), 'k')
        #plt.plot(self.getData(), 'r')
        #plt.show()

        if MJD:
            tauhatdec = np.reshape(np.array(fmap(Decimal, tauhat.flatten()),
                                            dtype=np.dtype(Decimal)),
                                   np.shape(tauhat))
            #print "tauhatdec",tauhatdec
            #+self.getTbin(numwrap=Decimal)*Decimal(nbin/2.0)
            #tauhat = tauhatdec * Decimal(dt)/Decimal(86400) #day units


            #tauhatdec += (-1*tauhatdec) #template tests implies tauhat is unnecessary?
            #tauhatdec = np.array(fmap(lambda x: x-int(x)+rollval,tauhatdec)) #why the heck
            #print tauhatdec,np.dtype(tauhatdec)

            # .item() allows for numpy.int to be cast in Python 3
            tauhatdec = np.array(fmap(lambda x: x+Decimal(rollval.item()),
                                      tauhatdec))

            #print tauhatdec
            tauhat = tauhatdec * Decimal(dt)/Decimal(86400) #day units
            #tauhat = (Decimal(nbin/2.0)-tauhatdec) * Decimal(dt)/Decimal(86400) #day units
            tauhat -= (artemp.channel_delays[0]*
                       self.getTbin(numwrap=Decimal)/
                       artemp.getTbin(numwrap=Decimal))/Decimal(86400)
            #print "tauhat",tauhat
            checknan = lambda x: x.is_nan()
        else:
            tauhat *= (dt*1e6)
            checknan = lambda x: np.isnan(x)
        sigma_tau *= (dt*1e6)

        output = ""

        t0 = 0.0
        start_time = self.getMJD(full=True, numwrap=Decimal)
        for i, T in enumerate(Taxis):
            tobs = self.durations[i]
            if MJD:
                t0 = start_time + self.subint_starts[i]/Decimal(86400)
                #print "start_time",start_time
                #print "subint_starts",self.subint_starts[i]
                #t0 = self.subint_starts[i]
                #t0 = Decimal(integration.get_start_time().in_days())
            for j, F in enumerate(Faxis):
                if checknan(tauhat[i, j]):
                    continue
                #Testing

                #if self.channel_delays[j] < 0:
                #    plt.plot(template*np.max(self.getData()[j]),'k')
                #    plt.plot(self.getData()[j],'r')
                #    plt.show()

                if self.channel_delays[j] <= Decimal(0):
                    self.channel_delays[j] += Decimal(self.getPeriod())
                #if self.channel_delays[j] < Decimal(0.0001):
                #    print(self.channel_delays[j],F)

                #print "foo",tauhat,self.channel_delays[j],self.subint_starts[i]/Decimal(86400)#self.getTbin(),self.getTbin()*2048
                toa = '{0:0.15f}'.format(Decimal(tauhat[i, j])+
                                         Decimal(t0)+self.channel_delays[j]/Decimal(86400))

                if isinstance(flags, (tuple, list, np.ndarray)):
                    flags = " ".join(flags)
                elif not isinstance(flags, str):
                    raise ValueError("Flags must be in string, tuple, list, or np.ndarray format")

                output += "%s %f %s   %0.3f  %s   -fe %s -be %s -bw %f -tobs %f -tmplt %s -nbin %i -nch %i -chan %i -subint %i -snr %0.2f -flux %0.2f -fluxerr %0.2f %s\n"%(self.filename, F, toa, sigma_tau[i, j], telescope, frontend, backend, chanbw, tobs, tempname, nbin, nchan, j, i, snrs[i, j], bhat[i, j], sigma_b[i, j], flags)

        if filename is None:
            if not appendto: #make the user decide whether or not to print this every time in a loop or not
                output = "FORMAT 1\n" + output
            print(output)
        else:
            if appendto and os.path.isfile(filename):
                with open(filename, 'a') as FILE:
                    FILE.write(output)
            else:
                output = "FORMAT 1\n" + output
                with open(filename, 'w') as FILE:
                    FILE.write(output)
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

    def getPeriod(self, header=False):
        """Returns period of the pulsar"""
        if self.isCalibrator():
            return 1.0/self.header['CAL_FREQ']
        #if self.params is None:FOO
        #    return None
        if header or self.polyco is None:
            return self.params.getPeriod()
        else:
            P0 = self.polyco.calculatePeriod()
            return P0
        
            #print P0,self.params.getPeriod()
            if np.abs(P0) < 1e-5: #Problem with large DT POLYCO values?
                return self.params.getPeriod()
            else:
                ratio = (P0-self.params.getPeriod())/self.params.getPeriod()
                if ratio < 0.5 or ratio > 2:
                    return self.params.getPeriod()
            return P0
            #return self.polyco.calculatePeriod()

    # Best replacement for without PSRCHIVE
    def getValue(self, value):
        """Looks for a key in one of the headers and returns"""
        if value in self.header.keys():
            return self.header[value]
        if value in self.subintinfo.keys():
            return self.subintinfo[value][-1]
        if self.params is None:
            return None
        return self.params.get(value) #will return None if non-existent

    def getSubintinfo(self, value):
        """Returns value from subintinfo"""
        if value in self.subintinfo.keys():
            return self.subintinfo[value][-1]
        return None

    def getName(self):
        """Returns pulsar name"""
        return self.header['SRC_NAME']

    def getMJD(self, full=False, numwrap=float):
        """Returns MJD of observation"""
        if full:
            return (numwrap(self.header['STT_IMJD']) +
                    (numwrap(self.header['STT_SMJD'])+numwrap(self.header['STT_OFFS']))/numwrap(86400))
        return numwrap(self.header['STT_IMJD'])+numwrap(self.header['STT_OFFS'])

    def getTbin(self, numwrap=float):
        """Returns the time per bin"""
        return numwrap(self.getPeriod()) / numwrap(self.getNbin())

    def getDM(self, numwrap=float):
        """Returns the data header DM"""
        if 'DM' in self.subintheader:
            return numwrap(self.subintheader['DM'])
        elif 'DM' in self.header:
            return numwrap(self.header['DM'])
        elif 'CHAN_DM' in self.header:
            return numwrap(self.header['CHAN_DM'])
        if self.params is None:
            return
        return self.params.getDM()

    def getRM(self, numwrap=float):
        """Returns the data header RM"""
        if 'RM' in self.subintheader.keys():
            return numwrap(self.subintheader['RM'])

    def getCoords(self, string=False, parse=False):
        """Returns the coordinate info in the header"""
        if string:
            RA = self.header['RA']
            dec = self.header['DEC']
            return RA, dec
        elif parse:
            RA = tuple(map(float, self.header['RA'].split(":")))
            dec = tuple(map(float, self.header['DEC'].split(":")))
            return RA, dec
        return coordinates.SkyCoord("%s %s"%(self.header['RA'], self.header['DEC']),
                                    unit=(units.hourangle, units.degree))
    getPulsarCoords = getCoords

    def getTelescopeCoords(self):
        """Returns the telescope coordinates"""
        return self.header['ANT_X'], self.header['ANT_Y'], self.header['ANT_Z']

    def getBandwidth(self, header=False):
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

    def getCenterFrequency(self, weighted=False):
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
        if (self.header['OBS_MODE'] == CAL or
                self.header['OBS_MODE'] == FON or
                self.header['OBS_MODE'] == FOF):
            return True
        return False

    #def varargtest(self,*args):
    #    self.record(inspect.currentframe())

    def record(self, frame):
        args, varargs, keywords, values = inspect.getargvalues(frame)
        funcname = frame.f_code.co_name
        string = "%s("%funcname
        for arg in args[1:]:
            if isinstance(values[arg], str):
                string += "%s=\"%s\","%(arg, values[arg])
            else:
                string += "%s=%s,"%(arg, values[arg])
        if varargs is not None:  # Var args typically not implemented in PyPulse
            argdict = values[varargs]
            string += "%s,"%(str(argdict)[1:-1].replace(", ", ","))
        if keywords is not None:
            kwargdict = values[keywords]
            for kwarg in kwargdict:
                string += "%s=%s,"%(kwarg, kwargdict[kwarg])
        if string[-1] == "(":
            string += ")"
        else:
            string = string[:-1] + ")"
        self.pypulse_history.append(string)

    def print_pypulse_history(self):
        for elem in self.pypulse_history:
            print(elem)


# Takes hdulist['HISTORY']
class History(object):
    def __init__(self, history):
        """Intializer"""
        self.header = dict()
        self.headerlist = history.header.keys()
        for key in self.headerlist:
            self.header[key] = history.header[key]
        self.dictionary = dict()
        self.namelist = list()
        for col in history.columns:
            self.namelist.append(col.name)
            self.dictionary[col.name] = (col.format, col.unit, list(col.array)) #make a np.array?
    def getValue(self, field, num=None):
        """Returns a dictionary array value for a given numeric entry"""
        if num is None:
            return self.dictionary[field][-1]
        else:
            try:
                return self.dictionary[field][-1][num]
            except IndexError:
                print("Entry out of range")
                return None
    def getLatest(self, field):
        """Returns the latest key value"""
        return self.getValue(field, -1)
    def printEntry(self, i):
        """Prints the i-th history entry"""
        for name in self.namelist:
            value = self.getValue(name, i)
            if value is None:
                return
            print(name, self.getValue(name, i))


# Takes hdulist['POLYCO']
# Similar to History class
class Polyco(object):
    def __init__(self, polyco, MJD=None):
        """Initializer"""
        self.MJD = MJD
        self.header = dict()
        self.headerlist = polyco.header.keys()
        for key in self.headerlist:
            self.header[key] = polyco.header[key]
        self.dictionary = dict()
        self.namelist = list()
        for col in polyco.columns:
            self.namelist.append(col.name)
            self.dictionary[col.name] = (col.format, col.unit, list(col.array)) #make a np.array?
    def getValue(self, field, num=None):
        """Returns a dictionary array value for a given numeric entry"""
        if num is None:
            return self.dictionary[field][-1]
        else:
            return self.dictionary[field][-1][num]
    def getLatest(self, field):
        """Returns the latest key value"""
        return self.getValue(field, -1)
    def calculate(self, MJD=None):
        if self.MJD is None and MJD is None:
            pass
        elif MJD is None:
            MJD = self.MJD

        #NSITE = self.getValue('NSITE', num=0)
        REF_FREQ = self.getValue('REF_FREQ', num=0)
        #PRED_PHS = self.getValue('PRED_PHS', num=0)
        REF_MJD = self.getValue('REF_MJD', num=0)
        REF_PHS = self.getValue('REF_PHS', num=0)
        REF_F0 = self.getValue('REF_F0', num=0)
        COEFF = self.getValue('COEFF', num=0)
        #print "POLYCO",REF_FREQ,REF_MJD,REF_PHS,REF_F0,COEFF
        #http://tempo.sourceforge.net/ref_man_sections/tz-polyco.txt
        DT = (MJD-REF_MJD)*1440.0
        #print "DT",DT,MJD,REF_MJD
        PHASE = REF_PHS + DT*60*REF_F0
        #print "PHASE",PHASE
        FREQ = 0.0
        for i, c in enumerate(COEFF):
            PHASE += c*np.power(DT, i)
            if i == 0:
                continue
            FREQ += c*i*np.power(DT, i-1)
        FREQ = REF_F0 + FREQ/60.0
        return PHASE, FREQ
    def calculatePeriod(self, MJD=None):
        PHASE, FREQ = self.calculate(MJD=MJD)
        return 1.0/FREQ
