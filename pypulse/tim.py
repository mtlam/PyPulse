'''
Michel Lam 2015
Loads a tim file
'''

import decimal as d
import numpy as np
import re
import sys
if sys.version_info.major == 2:
    fmap = map    
elif sys.version_info.major == 3:
    fmap = lambda x,*args: list(map(x,*args))

numre = re.compile('(\d+[.]\d+D[+]\d+)|(-?\d+[.]\d+)')
#flagre = re.compile('-[a-zA-Z]')

#www.atnf.csiro.au/research/pulsar/tempo2/index.php?n=Documentation.ObservationFiles
COMMANDS = ["EFAC","EQUAD","T2EFAC","T2EQUAD","GLOBAL_EFAC","EMAX","EMIN","EFLOOR","END","FMAX","FMIN","INCLUDE","INFO","MODE","NOSKIP","PHASE","SIGMA","SKIP","TIME","TRACK"]


'''
if all five primary arguments are given, run as normal
else filename is a string and must be parsed

kwargs are flags
'''
class TOA:
    def __init__(self,filename,freq=None,MJD=None,err=None,siteID=None,numwrap=d.Decimal,**kwargs):
        self.flags = []
        if freq is not None and MJD is not None and err is not None and siteID is not None: #behave using all arguments regularly
            self.filename = filename
            self.freq = float(freq) #numwrap?
            self.MJD = numwrap(MJD)
            self.err = float(err) #numwrap?
            self.siteID = siteID
            for flag,value in kwargs.items():
                setattr(self,flag,value)
                self.flags.append(flag)
        else: #parse all arguments
            self.toastring = filename #stores toa string
            splitstring = self.toastring.strip().split()
            self.filename = splitstring[0]
            self.freq = float(splitstring[1])
            self.MJD = numwrap(splitstring[2])
            self.err = float(splitstring[3])
            self.siteID = splitstring[4]
            for i in range(5,len(splitstring),2):
                flag = splitstring[i][1:]
                setattr(self,flag,splitstring[i+1])
                self.flags.append(flag)

    #def __repr__(self):
    #    return 
    def __str__(self):
        if isinstance(self.MJD,d.Decimal):
            retval = "%s %0.6f %s % 7.3f %+4s  "%(self.filename,self.freq,self.MJD,self.err,self.siteID)
        else:
            retval = "%s %0.6f %0.15f % 7.3f %+4s  "%(self.filename,self.freq,self.MJD,self.err,self.siteID)
        for i,flag in enumerate(self.flags):
            retval += "-%s %s "%(flag,getattr(self,flag))
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
    def getError(self):
        return self.err
    def getSiteID(self):
        return self.siteID
    def get(self,flag):
        value = None
        try:
            value = getattr(self,flag)
        except AttributeError:
            return None
        return value

    # Use these with extreme caution!
    def comment(self):
        self.filename = "C "+self.filename
    def setFilename(self,filename):
        self.filename = filename
    def setFreq(self,freq):
        self.setFrequency(freq)
    def setFrequency(self,freq):
        self.freq = freq
    def setMJD(self,MJD):
        self.MJD = MJD
    def setError(self,err):
        self.err = err
    def setSiteID(self,siteID):
        self.siteID = siteID
    def set(self,flag,value):
        if hasattr(self,flag):
            setattr(self,flag,value)
        else:
            raise AttributeError("TOA does not contain flag: %s"%flag)

    def add(self,flag,value):
        if hasattr(self,flag):
            raise AttributeError("Flag already exists: %s"%flag)
        else:
            self.flags.append(flag)
            setattr(self,flag,value)



class Tim:
    def __init__(self,filename,numwrap=d.Decimal):
        self.numwrap = numwrap
        self.load(filename)

    def load(self,filename):
        self.filename = filename

        if type(filename) == list or type(filename) == np.ndarray:
            lines = filename
        elif type(filename) == str or type(filename) == np.str or isinstance(filename,np.str_):
            with open(filename,'r') as FILE:#this assumes the file exists
                lines = FILE.readlines()
        else:
            return None

        self.comment_dict = dict() #store these for saving later
        self.command_dict = dict() 
        self.numlines = len(lines)

        self.toas = list()
        for i,line in enumerate(lines):
            if line[:2] == "C ":
                self.comment_dict[i] = line
                continue
            stripline = line.strip()
            count = stripline.count(" ")
            if count < 4: #is a command
                self.command_dict[i] = tuple(stripline.split()) #primitive handling
            else:
                toa = TOA(line,numwrap=self.numwrap)
                self.toas.append(toa)




    def save(self,filename):
        output = ""

        ntoa = 0
        for i in range(self.numlines):
            if i in self.comment_dict.keys():
                output += self.comment_dict[i]
            elif i in self.command_dict.keys():
                output += (" ".join(self.command_dict[i])+"\n")
            else:
                output += (str(self.toas[ntoa])+"\n")
                ntoa += 1
        with open(filename,'w') as FILE:
            FILE.write(output)



    def getTspan(self,years=False):
        mjds = fmap(lambda x: x.getMJD(),self.toas)
        if years:
            return np.ptp(mjds)/self.numwrap("365.25")
        return np.ptp(mjds)
