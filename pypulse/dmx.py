'''
Michael Lam 2019
Loads a dmxparse DMX file (tempo output)
'''
import sys
import numpy as np
import matplotlib.pyplot as plt

if sys.version_info.major == 2:
    fmap = map
elif sys.version_info.major == 3:
    fmap = lambda x, *args: list(map(x, *args))


class DM(object):
    def __init__(self, epoch, value=None, err=None, R1=None, R2=None,
                 F1=None, F2=None, dmxbin=None):
        if (value is not None and err is not None and R1 is not None and
                R2 is not None and F1 is not None and F2 is not None and dmxbin is not None):
            self.epoch = epoch
            self.value = value
            self.err = err
            self.R1 = R1
            self.R2 = R2
            self.F1 = F1
            self.F2 = F2
            self.dmxbin = dmxbin
        else: #parse all arguments
            self.dmstring = epoch #stores string
            splitstring = self.dmstring.strip().split()
            self.epoch = float(splitstring[0])
            self.value = float(splitstring[1])
            self.err = float(splitstring[2])
            self.R1 = float(splitstring[3])
            self.R2 = float(splitstring[4])
            self.F1 = float(splitstring[5])
            self.F2 = float(splitstring[6])
            self.dmxbin = splitstring[7]

    def __str__(self):
        pass

    def getMJD(self):
        return self.getEpoch()

    def getEpoch(self):
        return self.epoch

    def getDM(self):
        return self.value

    def getValue(self):
        return self.value

    def getErr(self):
        return self.getError()

    def getError(self):
        return self.err

    def getR1(self):
        return self.R1

    def getR2(self):
        return self.R2

    def getF1(self):
        return self.F1

    def getF2(self):
        return self.F2

    def getBin(self):
        return self.getDMXbin()

    def getDMXbin(self):
        return self.dmxbin

    # Use these with extreme caution!
    def setMJD(self, epoch):
        self.setEpoch(epoch)

    def setEpoch(self, epoch):
        self.epoch = epoch

    def setDM(self, value):
        self.setValue(value)

    def setValue(self, value):
        self.value = value

    def setErr(self, err):
        self.setError(err)

    def setError(self, err):
        self.err = err

    def setR1(self, R1):
        self.R1 = R1

    def setR2(self, R2):
        self.R2 = R2

    def setF1(self, F1):
        self.F1 = F1

    def setF2(self, F2):
        self.F2 = F2

    def setBin(self, dmxbin):
        self.setDMXbin(dmxbin)

    def setDMXbin(self, dmxbin):
        self.dmxbin = dmxbin


class DMX(object):
    def __init__(self, filename):
        self.filename = filename

        with open(filename, 'r') as FILE:
            lines = FILE.readlines()

        self.DMs = []
        self.comment_dict = dict()
        
        for i, line in enumerate(lines):
            if line[0] == "#":
                self.comment_dict[i] = line
                continue
            dm = DM(line)
            self.DMs.append(dm)

    def __repr__(self):
        return "DMX(%s)"%self.filename


    def plot(self, filename=None, show=True):
        """ Simple plotter """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.errorbar(self.getMJDs(), self.getDMs(), yerr=self.getErrs(), fmt='k.')

        ax.set_xlabel("MJD")
        ax.set_ylabel("DMX (pc cm^-3)")
        if filename is not None:
            plt.savefig(filename)
        if show:
            plt.show()
    

    def save(self, filename=None):
        """ Save DMX file """
        pass

    def getMJDs(self):
        """ Return MJDs of all DMs """
        return self.getEpochs()

    def getEpochs(self):
        """ Return MJDs of all DMs """
        return self.getter(lambda x: x.getEpoch())

    def getDMs(self):
        """ Return values of all DMs """
        return self.getter(lambda x: x.getValue())

    def getValues(self):
        """ Return values of all DMs """
        return self.getter(lambda x: x.getValue())

    def getErrs(self):
        """ Return errors of all DMs """
        return self.getErrors()

    def getErrors(self):
        """ Return errors of all DMs """
        return self.getter(lambda x: x.getError())

    def getR1s(self):
        """ Return R1s of all DMs """
        return self.getter(lambda x: x.getR1())

    def getR2s(self):
        """ Return R2 of all DMs """
        return self.getter(lambda x: x.getR2())

    def getF1s(self):
        """ Return F1s of all DMs """
        return self.getter(lambda x: x.getF1())

    def getF2s(self):
        """ Return F2 of all DMs """
        return self.getter(lambda x: x.getF2())

    def getBin(self):
        """ Return DMX bin of all DMs """
        return self.getDMXbin()

    def getDMXbin(self):
        """ Return DMX bin of all DMs """
        return self.getter(lambda x: x.getDMXbin())

    def getter(self, func):
        """ Generic getter. Not written as get() because it requires a function """
        return np.array(fmap(func, self.DMs))

    def getDMseries(self):
        """ get the time series, mirroring Par() """
        ts = self.getMJDs()
        dmxs = self.getValues()
        errs = self.getErrs()
        return ts, dmxs, errs

    def getTspan(self, years=False):
        """ Return total timespan of data """
        mjds = self.getMJDs()
        if years:
            return np.ptp(mjds)/365.25
        return np.ptp(mjds)
