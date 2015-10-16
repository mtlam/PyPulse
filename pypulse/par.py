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

c = 2.9979e8
PC_TO_M = 3.086e16
MAS_TO_RAD = np.pi/(180*60*60*1000)
YR_TO_S = 3.154e7

#numwrap could be float
class Par:
    def __init__(self,filename,numwrap=d.Decimal):
        self.filename = filename
        if type(filename) == list or type(filename) == np.ndarray:
            lines = filename
        elif type(filename) == str or type(filename) == np.str or isinstance(filename,np.str_):
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
            if len(splitline) <= 1:
                continue
            if splitline[0][0] == "#":
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
                elif splitline[1].isdigit():
                    value = int(splitline[1])
                elif splitline[1][1:].isdigit() and (splitline[1][0] == "+" or splitline[1][0] == "-"):
                    value = int(splitline[1])
                else:
                    value = splitline[1]

                if len(splitline) == 4:
                    if numre.match(splitline[-1]):
                        error = numwrap(splitline[-1].replace('D','e'))
                    elif splitline[1].isdigit():
                        error = int(splitline[-1])
                    elif splitline[1][1:].isdigit() and (splitline[1][0] == "+" or splitline[1][0] == "-"):
                        error = int(splitline[-1])
                    else:
                        error = splitline[-1]
                    self.errors[splitline[0]] = error
                self.parameters[splitline[0]] = value
        if type(filename) == str:                
            FILE.close()


    def __repr__(self):
        return "Par(%r)"%self.filename #numwrap?
    def __str__(self):
        if type(self.filename) == list or type(self.filename) == np.ndarray:
            return "\n".join(self.filename)
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
            F0 = self.parameters['F0']
        elif 'F' in self.parameters:
            F0 = self.parameters['F']
        elif 'IF0' in self.parameters:
            F0 = (self.parameters['IF0'] + self.parameters['FF0'])/self.numwrap(1000.0)
        return self.numwrap(1)/F0
    def getPeriodDot(self,shklovskii=False):
        if 'P1' in self.parameters:
            Pdot = self.parameters['P1']
        elif 'F1' in self.parameters:
            Pdot = self.numwrap(-1.0)*self.parameters['F1'] / (self.parameters['F0']**2)
        else:
            return None
        keys = self.parameters.keys()
        if shklovskii: #Correct for the shklovskii effect
            PM = self.getPM()
            if PM is None or "PX" not in keys:
                return Pdot

            P = self.getPeriod() #s
            PX = self.get("PX") #mas
            PXerr = self.get("PX",error=True)
            if PX <= 0 or (PXerr is not None and PXerr>=np.abs(PX)):
                return Pdot
        
            PM = PM * self.numwrap(MAS_TO_RAD/YR_TO_S) #mas/yr -> rad/s
            D = self.numwrap(1000*PC_TO_M)/PX #kpc -> m
            Pdot_pm = P*PM**2 *D/self.numwrap(c) #s/s
            return Pdot-Pdot_pm
        else:
            return Pdot
            
    def getPM(self):
        keys = self.parameters.keys()
        PM = None
        if "PMRA" in keys and "PMDEC" in keys:
            PM = np.sqrt(self.get("PMRA")**2 + self.get("PMDEC")**2) #mas/yr
        elif "PMRA" in keys:
            PM = abs(self.get("PMRA"))
        elif "PMDEC" in keys:
            PM = abs(self.get("PMDEC"))
        elif "PMLAMBDA" in keys and "PMBETA" in keys:
            PM = np.sqrt(self.get("PMLAMBDA")**2 + self.get("PMBETA")**2) #mas/yr
        elif "PMLAMBDA" in keys:
            PM = abs(self.get("PMLAMBDA"))
        elif "PMBETA" in keys:
            PM = abs(self.get("PMBETA"))
        return PM


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
        if "PSR" in self.parameters:
            return self.parameters["PSR"]
        elif "PSRJ" in self.parameters:
            return self.parameters["PSRJ"]
        return None
