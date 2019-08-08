'''
Michel Lam 2015
Loads a tim file
'''
import sys
import decimal
import numpy as np
#import re
if sys.version_info.major == 2:
    fmap = map
elif sys.version_info.major == 3:
    fmap = lambda x, *args: list(map(x, *args))

#numre = re.compile('(\d+[.]\d+D[+]\d+)|(-?\d+[.]\d+)')
#flagre = re.compile('-[a-zA-Z]')

#www.atnf.csiro.au/research/pulsar/tempo2/index.php?n=Documentation.ObservationFiles
COMMANDS = ["EFAC", "EQUAD", "T2EFAC", "T2EQUAD", "GLOBAL_EFAC", "EMAX", "EMIN", "EFLOOR", "END", "FMAX", "FMIN", "INCLUDE", "INFO", "MODE", "NOSKIP", "PHASE", "SIGMA", "SKIP", "TIME", "TRACK"]


'''
if all five primary arguments are given, run as normal
else filename is a string and must be parsed

kwargs are flags
'''

DECIMAL = decimal.Decimal

class TOA(object):
    def __init__(self, filename, freq=None, MJD=None, err=None,
                 siteID=None, numwrap=float, **kwargs):
        self.flags = []
        #behave using all arguments regularly
        if freq is not None and MJD is not None and err is not None and siteID is not None:
            self.filename = filename
            self.freq = float(freq) #numwrap?
            self.MJD = numwrap(MJD)
            self.err = float(err) #numwrap?
            self.siteID = siteID
            for flag, value in kwargs.items():
                setattr(self, flag, value)
                self.flags.append(flag)
        else: #parse all arguments
            self.toastring = filename #stores toa string
            splitstring = self.toastring.strip().split()
            self.filename = splitstring[0]
            self.freq = float(splitstring[1])
            self.MJD = numwrap(splitstring[2])
            self.err = float(splitstring[3])
            self.siteID = splitstring[4]
            for i in range(5, len(splitstring), 2):
                flag = splitstring[i][1:]
                setattr(self, flag, splitstring[i+1])
                self.flags.append(flag)

    #def __repr__(self):
    #    return
    def __str__(self):
        if isinstance(self.MJD, DECIMAL):
            retval = "%s %0.6f %s % 7.3f %+4s  "%(self.filename, self.freq, self.MJD, self.err, str(self.siteID))
        else:
            retval = "%s %0.6f %0.15f % 7.3f %+4s  "%(self.filename, self.freq, self.MJD, self.err, str(self.siteID))
        for flag in self.flags:
            retval += "-%s %s "%(flag, getattr(self, flag))
        retval = retval[:-1]
        return retval

    def getFilename(self):
        return self.filename

    def getFreq(self):
        return self.getFrequency()

    def getFrequency(self):
        return self.freq

    def getMJD(self):
        return self.MJD

    def getErr(self):
        return self.getError()

    def getError(self):
        return self.err

    def getSiteID(self):
        return self.siteID

    def get(self, flag):
        value = None
        try:
            value = getattr(self, flag)
        except AttributeError:
            return None
        return value

    # Use these with extreme caution!
    def comment(self, cut=None):
        self.filename = "C "+self.filename
        if cut is not None:
            self.add("cut", cut)

    def setFilename(self, filename):
        self.filename = filename

    def setFreq(self, freq):
        self.setFrequency(freq)

    def setFrequency(self, freq):
        self.freq = freq

    def setMJD(self, MJD):
        self.MJD = MJD

    def setErr(self, err):
        self.setError(err)

    def setError(self, err):
        self.err = err

    def setSiteID(self, siteID):
        self.siteID = siteID

    def set(self, flag, value):
        if hasattr(self, flag):
            setattr(self, flag, value)
        else:
            raise AttributeError("TOA does not contain flag: %s"%flag)

    def add(self, flag, value):
        if hasattr(self, flag):
            raise AttributeError("Flag already exists: %s"%flag)
        else:
            self.flags.append(flag)
            setattr(self, flag, value)


class Tim(object):
    def __init__(self, filename, numwrap=float, usedecimal=False):
        self.usedecimal = usedecimal
        if self.usedecimal:
            self.numwrap = DECIMAL
        else:
            self.numwrap = numwrap
        self.load(filename)

    def load(self, filename):
        self.filename = filename

        if isinstance(filename, (list, np.ndarray)):
            lines = filename
        elif isinstance(filename, (str, np.str)):
            with open(filename, 'r') as FILE:#this assumes the file exists
                lines = FILE.readlines()
        else:
            return None

        self.comment_dict = dict() #store these for saving later
        self.command_dict = dict() 
        self.numlines = len(lines)

        self.toas = list()
        for i, line in enumerate(lines):
            line = line.strip() #for non-IPTA-compliant TOAs
            if len(line) == 0 or line[:2] == "C " or line[:2] == "CC" or line[0] == "#": #CC might be bad, but otherwise there are too many variants
                self.comment_dict[i] = line
                continue
            stripline = line.strip()
            count = stripline.count(" ")
            splitline = stripline.split()
            if len(splitline) == 0 or splitline[0] in COMMANDS or count < 4: #is a command
                self.command_dict[i] = tuple(splitline) #primitive handling
            else:
                toa = TOA(line, numwrap=self.numwrap)
                self.toas.append(toa)

    def __repr__(self):
        numwrapstr = repr(self.numwrap).split("'")[1]
        return "Tim(%r, numwrap=%s, usedecimal=%r)" % (self.filename, numwrapstr, self.usedecimal)

    def comment(self, func, cut=None):
        """ Apply boolean function to comment TOAs """
        for index, toa in enumerate(self.toas):
            if func(toa):
                self.comment_line(index, cut=cut)

    def comment_line(self, index, cut=None):
        """ Comment out a single TOA, syntactic sugar """
        self.toas[index].comment(cut=cut)
                
    def any(self, func):
        """ Apply boolean function and see if any TOA meets said condition """
        for toa in self.toas:
            if func(toa):
                return True
        return False

    def all(self, func):
        """ Apply boolean function and see if all TOAs meet said condition """
        for toa in self.toas:
            if not func(toa):
                return False
        return True

    def save(self, filename=None):
        """ Save tim file """
        output = ""

        ntoa = 0
        for i in range(self.numlines):
            if i in self.comment_dict.keys():
                output += (self.comment_dict[i]+"\n")
            elif i in self.command_dict.keys():
                output += (" ".join(self.command_dict[i])+"\n")
            else:
                output += (str(self.toas[ntoa])+"\n")
                ntoa += 1
        if filename is None:
            filename = self.filename
        with open(filename, 'w') as FILE:
            FILE.write(output)

    def getFilenames(self):
        """ Return filenames of all TOAs """
        return np.array(fmap(lambda x: x.getFilename(), self.toas))
            
    def getFreqs(self):
        """ Return frequencies of all TOAs """
        return np.array(fmap(lambda x: x.getFreq(), self.toas))
    getFrequencies = getFreqs

    def getMJDs(self):
        """ Return MJDs of all TOAs """
        return np.array(fmap(lambda x: x.getMJD(), self.toas))

    def getErrs(self):
        """ Return uncertainties of all TOAs """
        return np.array(fmap(lambda x: x.getErr(), self.toas))
    getErrors = getErrs

    def get(self, value, numwrap=None):
        """ Return value of flag """
        if numwrap is None:
            retval = np.array(fmap(lambda x: x.get(value), self.toas))
        else:
            retval = np.array(fmap(lambda x: numwrap(x.get(value)), self.toas))
        return retval

    def getTspan(self, years=False):
        """ Return total timespan of data """
        mjds = self.getMJDs()
        if years:
            return np.ptp(mjds)/self.numwrap("365.25")
        return np.ptp(mjds)

    def set(self, flag, value):
        """ Set value of flag for all TOAs """
        for toa in self.toas:
            toa.set(flag, value)
        return
