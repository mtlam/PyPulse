'''
Michel Lam 2015
Loads a parameter file

Look at fit flags?

Now takes either a list of strings or a filename



if 3 or 5, maybe no fit flag
Noise model parameters take a flag, should I store which flag that is? (e.g. store the "-f" or "-fe"?)
'''

import decimal as d
import numpy as np
import re

numre = re.compile('(\d+[.]\d+D[+]\d+)|(-?\d+[.]\d+)')
flagre = re.compile('-[a-zA-Z]')


#numwrap could be float
class Par:
    def __init__(self,filename,numwrap=d.Decimal):
        self.filename = filename
        if type(filename) == list or type(filename) == np.ndarray:
            lines = filename
        elif type(filename) == str:
            FILE = open(filename,'r')
            lines = FILE.readlines()
        else:
            return None

        self.numwrap = numwrap
        self.paramlist = list()
        self.parameters = dict()
        self.errors = dict()
        for line in lines:
            splitline = line.strip().split()
            if len(splitline) == 0:
                continue
            self.paramlist.append(splitline[0])

            if flagre.match(splitline[1][:2]) and len(splitline)>=4:
                key = splitline[2]
                value = numwrap(splitline[3])
                self.parameters[(splitline[0],key)] = value
                if len(splitline) == 6:
                    error = numwrap(splitline[-1])
                    self.errors[(splitline[0],key)] = error
            else:
                if numre.match(splitline[1]):
                    value = numwrap(splitline[1].replace('D','e'))
                else:
                    value = splitline[1]

                if len(splitline) == 4:
                    if numre.match(splitline[-1]):
                        error = numwrap(splitline[-1].replace('D','e'))
                    else:
                        error = splitline[-1]
                    self.errors[splitline[0]] = error
                self.parameters[splitline[0]] = value
        if type(filename) == str:                
            FILE.close()


    def __repr__(self):
        return "Par(%r)"%self.filename #numwrap?
    def __str__(self):
        return self.filename



    def save(self,filename):
        pass




    def get(self,tag,flag=None,error=False):
        if flag:
            tag = (tag,flag)
        if error:
            if tag in self.errors:
                return self.errors[tag]
            return None
        if tag in self.parameters:
            return self.parameters[tag]
        return None
    def getPeriod(self):
        if 'P0' in self.parameters:
            return self.parameters['P0']
        if 'F0' in self.parameters:
            return self.numwrap(1)/self.parameters['F0']
    def getPeriodDot(self):
        if 'P1' in self.parameters:
            return self.parameters['P1']
        if 'F1' in self.parameters:
            return self.numwrap(-1.0)*self.parameters['F1'] / (self.parameters['F0']**2)
        return None
    def getDM(self):
        return self.get('DM')
    def getDMX(self):
        keys = self.parameters.keys()
        Ncomponents = 0
        for key in keys:
            if key[0:4] == 'DMX_':
                Ncomponents += 1
        if Ncomponents == 0:
            return None
        #DM = self.getDM()
        ts = np.zeros(Ncomponents)
        dmxs = np.zeros(Ncomponents)
        errs = np.zeros(Ncomponents)
        for i in range(Ncomponents):
            ts[i] = self.get('DMXEP_%04i'%(i+1))
            dmxs[i] = self.get('DMX_%04i'%(i+1))
            errs[i] = self.get('DMX_%04i'%(i+1),error=True) #check to make sure this exists?
        return ts,dmxs,errs
    def getDMseries(self):
        ts,dmxs,errs = self.getDMX()
        DM = self.getDM()
        return ts,dmxs + float(DM), errs #float from possible decimal!
    def getName(self):
        return self.parameters["PSR"]

