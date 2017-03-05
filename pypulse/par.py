'''
Michel Lam 2015
Loads a parameter file
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


class Parameter:
    def __init__(self,name,value=None,fit=None,error=None,flag=None,flagvalue=None,numwrap=d.Decimal):
        if name[0] == "#":
            return None #?
        self.numwrap = numwrap
        # Initialize all values just in case
        self.name = name
        self.value = value
        self.fit = fit
        self.error = error
        self.flag = flag
        self.flagvalue = flagvalue
        if value is None: #parse all arguments
            self.parstring = name
            splitstring = self.parstring.strip().split()
            if len(splitstring) == 0:
                return None #?
            self.name = splitstring[0]
                

            if flagre.match(splitstring[1][:2]) and len(splitstring)>=4: #flag present
                self.flag = splitstring[1]
                self.flagvalue = splitstring[2]
                self.value = numwrap(splitstring[3])
                if len(splitstring) >= 5:                                    
                    self.error = numwrap(splitstring[-1])
                    if len(splitstring) == 6:
                        self.fit = int(splitstring[4])
            else: #no flag present
                if numre.match(splitstring[1]):
                    self.value = numwrap(splitstring[1].replace('D','e'))
                elif splitstring[1].isdigit():
                    self.value = int(splitstring[1])
                elif splitstring[1][1:].isdigit() and (splitstring[1][0] == "+" or splitstring[1][0] == "-"):
                    self.value = int(splitstring[1])
                else:
                    self.value = splitstring[1]

                if len(splitstring) == 3 or len(splitstring) == 4:
                    if numre.match(splitstring[-1]):
                        self.error = numwrap(splitstring[-1].replace('D','e'))
                    elif splitstring[1].isdigit():
                        self.error = int(splitstring[-1])
                    elif splitstring[1][1:].isdigit() and (splitstring[1][0] == "+" or splitstring[1][0] == "-"):
                        self.error = int(splitstring[-1])
                    else:
                        self.error = splitstring[-1]
                    if len(splitstring) == 3:
                        self.fit = int(splitstring[2])

    def getName(self):
        return self.name
    def getValue(self):
        return self.value
    def getFit(self):
        return self.fit
    def getError(self):
        return self.error
    def getFlag(self):
        return self.flag
    def getFlagValue(self):
        return self.flagvalue





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
        self.paramlist = list() #each unique parameter
        self.paramnames = list() #the names of each parameter
        for line in lines:
            p = Parameter(line)
            self.paramlist.append(p)
            self.paramnames.append(p.getName())
        self.paramnames = np.array(self.paramnames,dtype=np.str)
        if type(filename) == str:                
            FILE.close()


    def __repr__(self):
        return "Par(%r)"%self.filename #numwrap?
    def __str__(self):
        if type(self.filename) == list or type(self.filename) == np.ndarray:
            return "\n".join(self.filename)
        return self.filename



    def save(self,filename):
        # Crude saving attempt
        output = ""
        for param in self.paramlist:
            output += param.parstring
        with open(filename,'w') as FILE:
            FILE.write(output)



    def getInd(self,tag):
        return np.where(self.paramnames==tag)[0]
    def get(self,tag,flag=None,error=False):
        if tag in self.paramnames:
            ind = self.getInd(tag)
            if error:
                return self.paramlist[ind].getError()
            if flag:
                return self.paramlist[ind].getFlagValue()
            return self.paramlist[ind].getValue()
        return None
    def getPeriod(self):
        if 'P0' in self.paramnames:
            return self.get('P0')
        if 'F0' in self.paramnames:
            F0 = self.get('F0')
        elif 'F' in self.paramnames:
            F0 = self.get('F')
        elif 'IF0' in self.paramnames:
            F0 = (self.get('IF0') + self.get('FF0'))/self.numwrap(1000.0)
        return self.numwrap(1.0)/F0
    def getPeriodDot(self,shklovskii=False):
        if 'P1' in self.paramnames:
            Pdot = self.get('P1')
        elif 'F1' in self.paramnames:
            Pdot = self.numwrap(-1.0)*self.get('F1') / (self.get('F0')**2)
        else:
            return None

        if shklovskii: #Correct for the shklovskii effect
            PM = self.getPM()
            if PM is None or "PX" not in self.paramnames:
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

    def getFrequency(self):
        return self.numwrap(1.0)/self.getPeriod()
    def getFrequencyDot(self,shklovskii=False):
        return self.numwrap(-1.0) * self.getPeriodDot(shklovskii=shklovskii) / self.getPeriod()**2
        
            
    def getPM(self):
        keys = self.paramnames
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
    def getDMX(self,full_output=False):
        keys = self.paramnames
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
        if full_output:
            R1s = np.zeros(Ncomponents)
            R2s = np.zeros(Ncomponents)
            F1s = np.zeros(Ncomponents)
            F2s = np.zeros(Ncomponents)
        for i in range(Ncomponents):
            ts[i] = self.get('DMXEP_%04i'%(i+1))
            if np.isnan(ts[i]):
                ts[i] = 0.5*(self.get('DMXR1_%04i'%(i+1))+self.get('DMXR2_%04i'%(i+1)))
            dmxs[i] = self.get('DMX_%04i'%(i+1))
            errs[i] = self.get('DMX_%04i'%(i+1),error=True) #check to make sure this exists?
            if full_output:
                R1s[i] = self.get('DMXR1_%04i'%(i+1))
                R2s[i] = self.get('DMXR2_%04i'%(i+1))
                F1s[i] = self.get('DMXF1_%04i'%(i+1))
                F2s[i] = self.get('DMXF2_%04i'%(i+1))
                if np.isnan(ts[i]):
                    ts[i] = (R1s[i]+R2s[i])/2.0
        if full_output:
            return ts,dmxs,errs,R1s,R2s,F1s,F2s
        return ts,dmxs,errs
    def getDMseries(self):
        ts,dmxs,errs = self.getDMX()
        DM = self.getDM()
        return ts,dmxs + float(DM), errs #float from possible decimal!

    def getFD(self):
        coeffs = []
        for param in self.paramnames:
            if "FD" in param:
                coeffs.append(self.get(param))
        if len(coeffs) == 0:
            return None
        return np.array(coeffs)
    def getFDfunc(self):
        """
        Returns a function that provides the timing delays as a function of observing frequency
        """
        FD = self.getFD()
        if FD is None:
            return None
        FD = FD[::-1]
        FD = np.concatenate((FD,[0]))
        f = lambda nu: 1e6*np.polyval(FD,np.log(nu)) #nu in GHz, returns values in microseconds
        return f

    def getName(self):
        if "PSR" in self.paramnames:
            return self.get("PSR")
        elif "PSRJ" in self.paramnames:
            return self.get("PSRJ")
        return None

    def getTspan(self,years=False):
        start = self.get("START")
        finish = self.get("FINISH")
        if years:
            return (finish-start)/365.25
        return finish-start
