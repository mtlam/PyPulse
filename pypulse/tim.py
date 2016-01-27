'''
Michel Lam 2015
Loads a tim file
'''

import decimal as d
import numpy as np
import re

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
        if freq is not None and MJD is not None and err is not None and siteID is not None: #behave using all arguments regularly
            self.filename = filename
            self.freq = float(freq) #numwrap?
            self.MJD = numwrap(MJD)
            self.err = float(err) #numwrap?
            self.siteID = siteID
            for key,value in kwargs.items():
                setattr(self,key,value)
        else: #parse all arguments
            self.toastring = filename #stores toa string
            splitstring = self.toastring.strip().split()
            self.filename = splitstring[0]
            self.freq = float(splitstring[1])
            self.MJD = numwrap(splitstring[2])
            self.err = float(splitstring[3])
            self.siteID = splitstring[4]
            for i in range(5,len(splitstring),2):
                setattr(self,splitstring[i][1:],splitstring[i+1])
    #def __repr__(self):
    #    return 
    def __str__(self):
        return self.filename #?
            

    def getFilename(self):
        return self.filename
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
    


class Tim:
    def __init__(self,filename,numwrap=d.Decimal):
        self.filename = filename

        if type(filename) == list or type(filename) == np.ndarray:
            lines = filename
        elif type(filename) == str or type(filename) == np.str or isinstance(filename,np.str_):
            FILE = open(filename,'r') #this assumes the file exists
            lines = FILE.readlines()
        else:
            return None

        self.commands = [] #currently only stores these, but does not apply anything to any order
        self.toas = list()
        for line in lines:
            if line[:2] == "C ":
                continue
            stripline = line.strip()
            count = stripline.count(" ")
            if count < 4: #is a command
                self.commands.append(tuple(stripline.split())) #primitive handling
            else:
                toa = TOA(line)
                self.toas.append(toa)


