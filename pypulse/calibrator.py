'''




'''

import numpy as np

from matplotlib.pyplot import *
import os
import sys
if sys.version_info.major == 2:
    fmap = map    
elif sys.version_info.major == 3:
    fmap = lambda x,*args: list(map(x,*args))
    xrange = range



class Calibrator:
    def __init__(self,freqs,S,Serr=None,pol_type='Coherence',fd_poln='LIN',verbose=True):
        self.pol_type = pol_type
        self.freqs = freqs
        self.S = S
        if Serr is None:
            Serr = np.zeros(4,dtype=np.float32) 
        self.Serr = Serr
        self.verbose = verbose

        if self.pol_type == 'Coherence' or self.pol_type == 'AABBCRCI':
            A,B,C,D = S
            Aerr,Berr,Cerr,Derr = Serr
            if fd_poln == 'LIN':
                S0 = A+B #I
                S1 = A-B #Q
                S2 = 2*C #U
                S3 = 2*D #V
                S0err = np.sqrt(Aerr**2 + Berr**2)
                S1err = S0err
                S2err = 2*Cerr
                S3err = 2*Derr
        elif self.pol_type == 'Stokes' or self.pol_type == 'IQUV':
            S0,S1,S2,S3 = S
            S0err,S1err,S2err,S3err = Serr
        else:
            raise SystemExit
        self.I = S0
        self.Q = S1
        self.U = S2
        self.V = S3
        self.Ierr = S0err
        self.Qerr = S1err
        self.Uerr = S2err
        self.Verr = S3err




    def pacv(self):
        '''
        Emulates pacv <archive>
        See More/Polarimetry/SingleAxisSolver.C
        '''
        dG = 2*self.Q/self.I
        dpsi = np.arctan2(self.V,self.U)
        subplot(311)
        plot(dpsi,'k.')
        subplot(312)
        plot(100*dG,'k.')
        subplot(313)
        U_0 = self.U/np.cos(dpsi)
        #plot(self.I)
        #plot(np.sqrt(self.U**2+self.V**2)/U_0,'k.')
        show()

    def pacv_csu(self):
        '''
        Emulates pacv -n csu <archive>
        '''

        errorbar(self.freqs,self.I,yerr=self.Ierr,fmt='k.')
        errorbar(self.freqs,self.Q,yerr=self.Qerr,fmt='r.')
        errorbar(self.freqs,self.U,yerr=self.Uerr,fmt='g.')
        errorbar(self.freqs,self.V,yerr=self.Verr,fmt='b.')
        xlabel('Frequency')
        ylabel('Calibrator Stokes')
        show()
        

    def apply_calibration(self,ar):
        M_PAs = []
        PAR_ANG = ar.getSubintinfo('PAR_ANG')
        if PAR_ANG is None:
            POS_ANG = ar.getSubintinfo('POS_ANG')
            if POS_ANG is None:
                raise IndexError("No parallactic/position angle information")
            elif self.verbose:
                print("Calibrating using position angles")
            PAR_ANG = POS_ANG
        elif self.verbose:
            print("Calibrating using parallactic angles")
        PAR_ANG *= np.pi/180 #check if degrees!


        dG = 2*self.Q/self.I
        dpsi = np.arctan2(self.V,self.U)


        print(ar.shape())
        data = ar.getData() #perform checks here

        # Remove baselines?


        POL_TYPE = ar.subintheader['POL_TYPE']


        I = xrange(ar.getNsubint())
        J = xrange(ar.getNchan())
        K = xrange(ar.getNbin())
        calibrated_data = np.zeros_like(data)
        for i in I:
            print i
            M_PA = self.build_mueller_matrix_PA(PAR_ANG[i])
            for j in J:
                M_differential = self.build_mueller_matrix_differential((dG[j],dpsi[j])) # must match exactly
                M = np.dot(M_differential,M_PA)
                Minv = np.linalg.inv(M)
                for k in K:
                    S = self.convert_polarization(data[i,:,j,k],POL_TYPE,"IQUV")
                    calibrated_data[i,:,j,k] = self.convert_polarization(np.dot(Minv,S),"IQUV",POL_TYPE)
                # reset baseline
                #if i == 5 and j == 50:
                #    plot(calibrated_data[i,0,j,:])
                #    show()
                    #imshow(calibrated_data[i,1,:,:])
                    #show()
                    #imshow(calibrated_data[i,2,:,:])
                    #show()
                    #imshow(calibrated_data[i,3,:,:])
                    #show()
                #    raise SystemExit
                #calibrated_data[i,:,j,:] -= np.mean(calibrated_data[i,:,j,ar.opw])
            #if i == 10:
            #    imshow(calibrated_data[i,0,:,:])
            #    show()
            #    raise SystemExit

                
        print np.mean(calibrated_data[5,:,25,:])
        ar.setData(calibrated_data)



    def build_mueller_matrix_PA(self,PA):
        if PA is None:
            M_PA = np.identity(4)
        else:
            cos = np.cos(2*PA)
            sin = np.sin(2*PA)
            M_PA = [[1,0,0,0],
                    [0,cos,sin,0],
                    [0,-sin,cos,0],
                    [0,0,0,1]]
        return M_PA

    def build_mueller_matrix_differential(self,differential):
        if differential is None:
            M_differential = np.identity(4)
        else:
            dG,dpsi = differential
            cos = np.cos(dpsi)
            sin = np.sin(dpsi)
            M_differential = [[1,dG/2.0,0,0],
                              [dG/2.0,1,0,0],
                              [0,0,cos,-sin],
                              [0,0,sin,cos]]
        return M_differential




    def convert_polarization(self,S,intype,outtype,linear=True):
        if intype == outtype:
            return S
        elif intype == "AABBCRCI" and outtype == "IQUV": # Coherence -> Stokes
            A,B,C,D = S
            if linear:
                I = A+B
                Q = A-B
                U = 2*C
                V = 2*D
            else:
                I = A+B
                Q = 2*C
                U = 2*D
                V = A-B
            outS = [I,Q,U,V]
        elif intype == "IQUV" and outtype == "AABBCRCI": # Stokes -> Coherence
            I,Q,U,V = S
            if linear:
                pass
                A = (I+Q)/2.0
                B = (I-Q)/2.0
                C = U/2.0
                D = V/2.0
            else:
                A = (I+V)/2.0
                B = (I-V)/2.0
                C = Q/2.0
                D = U/2.0
            outS = [A,B,C,D]
        if type(S) == np.ndarray:
            return np.array(outS)
        elif type(S) == list:
            return outS
            


    def build_mueller_matrix(self,PA=None,feed=None,CC=None,differential=None):
        """
        Following Lorimer & Kramer methodology
        PA = parallactic angle (scalar)
        feed = 
        CC = cross coupling (4-vector of )
        differential = differential gain and phase (2-vector)
        """
        if PA is None:
            M_PA = np.identity(4)
        else:
            cos = np.cos(2*PA)
            sin = np.sin(2*PA)
            M_PA = [[1,0,0,0],
                    [0,cos,sin,0],
                    [0,-sin,cos,0],
                    [0,0,0,1]]
        if feed is None:
            M_feed = np.identity(4)
        else:
            cos = np.cos(2*feed)
            sin = np.sin(2*feed)
            M_feed = [[1,0,0,0],
                      [0,cosarg,0,sin],
                      [0,0,1,0],
                      [0,-sin,0,cos]]
        if CC is None:
            M_CC = np.identity(4)
        else:
            M_CC = np.identity(4) #NOT YET IMPLEMENTED
        if differential is None:
            M_differential = np.identity(4)
        else:
            dG,dpsi = differential
            cos = np.cos(dpsi)
            sin = np.sin(dpsi)
            M_differential = [[1,dG/2.0,0,0],
                              [dG/2.0,1,0,0],
                              [0,0,cos,-sin],
                              [0,0,sin,cos]]
        return np.dot(np.dot(np.dot(M_differential,M_CC),M_feed),M_PA)
        


    def calculate_PA(self,lat,dec,HA):
        """
        Helper function
        lat = latitude
        dec = declination of source
        HA = hour angle
        """
        return np.arctan2(np.sin(HA)*np.cos(lat),np.sin(lat)*np.cos(dec) - np.cos(lat)*np.sin(phi)*np.cos(HA))







    def calculate_calibrator_flux(source,freqs,filename=None):
        """
        Process the fluxcal.cfg file. 
        All frequencies must be in MHz!    
        """
        if filename is None:
            filename = os.path.join(os.path.dirname(__file__),"config","fluxcal.cfg")
        with open(filename,'r') as FILE:
            lines = FILE.readlines()

        currentvals = []
        found = False
        for i,line in enumerate(lines):
            if line[0] == "\n" or line[0] == "#" or line[0] == " ":
                continue

            splitline = line.strip().split()
            if len(currentvals) == 0:
                currentvals = splitline
            if splitline[0] == "aka" and source in splitline[1]:
                found = True
                break
            elif source in splitline[0]:
                found = True
                currentvals = splitline
                break
            elif splitline[0] != "aka":
                currentvals = splitline
        if not found:
            raise ValueError("Flux calibration source not found")

        fluxes = np.zeros(len(freqs))
        if currentvals[0][0] == "&": #Format 2, Flux in Jy for a frequency in GHz is: log10(S) = a_0 + a_1*log10(f) + a_2*(log10(f))^2 + ...
            logfreqs = np.log10(freqs/1000.0)
            coeffs = map(lambda x: float(x),currentvals[3:])
            for i,coeff in enumerate(coeffs):
                fluxes += coeff*np.power(logfreqs,i)
        else:
            freq = float(currentvals[3]) #MHz
            flux = float(currentvals[4]) #Jy
            index = float(currentvals[5])

            fluxes = flux*np.power((np.array(freqs)/freq),-1*index)
        return fluxes
