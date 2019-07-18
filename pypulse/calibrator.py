'''

'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import astropy.coordinates as coordinates
import astropy.units as units

if sys.version_info.major == 2:
    fmap = map
elif sys.version_info.major == 3:
    fmap = lambda x, *args: list(map(x, *args))
    xrange = range

ON = "ON"
OFF = "OFF"

class Calibrator(object):
    def __init__(self, freqs, S, Serr=None, pol_type='Coherence', fd_poln='LIN', verbose=True):
        self.pol_type = pol_type
        self.freqs = np.array(freqs)
        self.S = np.array(S)
        if Serr is None:
            Serr = np.zeros(4, dtype=np.float32)
        self.Serr = np.array(Serr)
        self.verbose = verbose

        if self.pol_type == 'Coherence' or self.pol_type == 'AABBCRCI':
            A, B, C, D = S
            Aerr, Berr, Cerr, Derr = Serr
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
            S0, S1, S2, S3 = S
            S0err, S1err, S2err, S3err = Serr
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
        dpsi = np.arctan2(self.V, self.U)
        plt.subplot(311)
        plt.plot(dpsi, 'k.')
        plt.subplot(312)
        plt.plot(100*dG, 'k.')
        plt.subplot(313)
        U_0 = self.U/np.cos(dpsi)
        #plot(self.I)
        #plot(np.sqrt(self.U**2+self.V**2)/U_0, 'k.')
        plt.show()

    def pacv_csu(self):
        '''
        Emulates pacv -n csu <archive>
        '''

        plt.errorbar(self.freqs, self.I, yerr=self.Ierr, fmt='k.')
        plt.errorbar(self.freqs, self.Q, yerr=self.Qerr, fmt='r.')
        plt.errorbar(self.freqs, self.U, yerr=self.Uerr, fmt='g.')
        plt.errorbar(self.freqs, self.V, yerr=self.Verr, fmt='b.')
        plt.xlabel('Frequency')
        plt.ylabel('Calibrator Stokes')
        plt.show()

    def applyFluxcal(self, fluxcalonar, fluxcaloffar=None):
        if fluxcaloffar is None: #The fluxcalon file contains both ON and OFF observations
            pass
        else:
            fluxcalonfreqs, fluxcalondatalow, fluxcalondatahigh, fluxcalonerrslow, fluxcalonerrshigh = fluxcalonar.getLevels()
            fluxcalofffreqs, fluxcaloffdatalow, fluxcaloffdatahigh, fluxcalofferrslow, fluxcalofferrshigh = fluxcaloffar.getLevels()

        source = fluxcalonar.getName()
        if source != fluxcaloffar.getName():
            raise ValueError("Mismatch in fluxcal names")

        config = CalibratorConfig()
        calflux = config.calculateCalibratorFlux(source, self.freqs)

        # Flux calibrator amplitudes, the difference between on and off source without the noise diode operational.
        S_cal = fluxcalondatalow - fluxcaloffdatalow
        #print np.shape(calflux), np.shape(S_cal)

        #self.I = self.I

    def applyCalibration(self, ar):
        M_PAs = []
        PAR_ANG = ar.getSubintinfo('PAR_ANG')
        if PAR_ANG is None:
            POS_ANG = ar.getSubintinfo('POS_ANG')
            if POS_ANG is None:
                print("No parallactic/position angle information")
            elif self.verbose:
                print("Calibrating using position angles")
            PAR_ANG = POS_ANG
        elif self.verbose:
            print("Calibrating using parallactic angles")
        if PAR_ANG is not None:
            PAR_ANG *= np.pi/180 #check if degrees!

        dG = 2*self.Q/self.I
        dpsi = np.arctan2(self.V, self.U)

        data = ar.getData(squeeze=False) #perform checks here
        # Remove baselines?

        POL_TYPE = ar.subintheader['POL_TYPE']

        I = xrange(ar.getNsubint())
        J = xrange(ar.getNchan())
        K = xrange(ar.getNbin())
        calibrated_data = np.zeros_like(data)
        for i in I:
            print(i)
            if PAR_ANG is not None:
                M_PA = self.buildMuellerMatrixPA(PAR_ANG[i])
            for j in J:
                M_differential = self.buildMuellerMatrixDifferential((dG[j], dpsi[j])) # must match exactly
                if PAR_ANG is not None:
                    M = np.dot(M_differential, M_PA)
                else:
                    M = M_differential
                Minv = np.linalg.inv(M)
                for k in K:
                    S = self.convertPolarization(data[i, :, j, k], POL_TYPE, "IQUV")
                    #print np.shape(Minv), np.shape(S), np.shape(calibrated_data[i, :, j, k])
                    calibrated_data[i, :, j, k] = self.convertPolarization(np.dot(Minv, S), "IQUV", POL_TYPE)
                # reset baseline
                #if i == 5 and j == 50:
                #    plot(calibrated_data[i, 0, j, :])
                #    show()
                    #imshow(calibrated_data[i, 1, :, :])
                    #show()
                    #imshow(calibrated_data[i, 2, :, :])
                    #show()
                    #imshow(calibrated_data[i, 3, :, :])
                    #show()
                #    raise SystemExit
                #calibrated_data[i, :, j, :] -= np.mean(calibrated_data[i, :, j, ar.opw])
            #if i == 10:
            #    imshow(calibrated_data[i, 0, :, :])
            #    show()
            #    raise SystemExit

        #print np.mean(calibrated_data[5, :, 25, :])
        ar.setData(calibrated_data)

    def buildMuellerMatrixPA(self, PA):
        if PA is None:
            M_PA = np.identity(4)
        else:
            cos = np.cos(2*PA)
            sin = np.sin(2*PA)
            M_PA = [[1, 0, 0, 0],
                    [0, cos, sin, 0],
                    [0, -sin, cos, 0],
                    [0, 0, 0, 1]]
        return M_PA

    def buildMuellerMatrixDifferential(self, differential):
        if differential is None:
            M_differential = np.identity(4)
        else:
            dG, dpsi = differential
            cos = np.cos(dpsi)
            sin = np.sin(dpsi)
            M_differential = [[1, dG/2.0, 0, 0],
                              [dG/2.0, 1, 0, 0],
                              [0, 0, cos, -sin],
                              [0, 0, sin, cos]]
        return M_differential

    def convertPolarization(self, S, intype, outtype, linear=True):
        if intype == outtype:
            return S
        elif intype == "AABBCRCI" and outtype == "IQUV": # Coherence -> Stokes
            A, B, C, D = S
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
            outS = [I, Q, U, V]
        elif intype == "IQUV" and outtype == "AABBCRCI": # Stokes -> Coherence
            I, Q, U, V = S
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
            outS = [A, B, C, D]
        if isinstance(S, np.ndarray):
            return np.array(outS)
        elif isinstance(S, list):
            return outS

    def buildMuellerMatrix(self, PA=None, feed=None, CC=None, differential=None):
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
            M_PA = [[1, 0, 0, 0],
                    [0, cos, sin, 0],
                    [0, -sin, cos, 0],
                    [0, 0, 0, 1]]
        if feed is None:
            M_feed = np.identity(4)
        else:
            cos = np.cos(2*feed)
            sin = np.sin(2*feed)
            M_feed = [[1, 0, 0, 0],
                      [0, cos, 0, sin],
                      [0, 0, 1, 0],
                      [0, -sin, 0, cos]]
        if CC is None:
            M_CC = np.identity(4)
        else:
            M_CC = np.identity(4) #NOT YET IMPLEMENTED
        if differential is None:
            M_differential = np.identity(4)
        else:
            dG, dpsi = differential
            cos = np.cos(dpsi)
            sin = np.sin(dpsi)
            M_differential = [[1, dG/2.0, 0, 0],
                              [dG/2.0, 1, 0, 0],
                              [0, 0, cos, -sin],
                              [0, 0, sin, cos]]
        return np.dot(np.dot(np.dot(M_differential, M_CC), M_feed), M_PA)

    def calculatePA(self, lat, dec, HA):
        """
        Helper function
        lat = latitude
        dec = declination of source
        HA = hour angle
        """
        return np.arctan2(np.sin(HA)*np.cos(lat),
                          np.sin(lat)*np.cos(dec) - np.cos(lat)*np.sin(dec)*np.cos(HA))

### ==================================================
### Helper functions
### ==================================================

class CalibratorConfig:
    def __init__(self, filename=None):
        if filename is None:
            filename = os.path.join(os.path.dirname(__file__), "config", "fluxcal.cfg")
        self.filename = filename
        self.configlines = self.readConfigFile()

    def readConfigFile(self):
        """
        Process the fluxcal.cfg file.
        """
        with open(self.filename, 'r') as FILE:
            lines = FILE.readlines()
        retval = []
        for i, line in enumerate(lines):
            if line[0] == "\n" or line[0] == "#" or line[0] == " ":
                continue
            retval.append(line.strip())
        return retval

    def getConfigLine(self, source):
        currentvals = []
        found = False
        for i, line in enumerate(self.configlines):
            splitline = line.split()
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
        return currentvals

    def getCalibratorCoords(self, source):
        configline = self.getConfigLine(source)
        return coordinates.SkyCoord("%s %s"%(configline[1], configline[2]), unit=(units.hourangle, units.degree))

    def checkOnOff(self, source, coords, tolerance=1):
        """
        Check if the flux cal is on or off source
        Tolerance in arcminutes
        """
        configline = self.getConfigLine(source)
        sourcecoords = self.getCalibratorCoords(source)
        #print coords, sourcecoords, sourcecoords.separation(coords)
        if sourcecoords.separation(coords) <= tolerance*units.arcmin:
            return ON
        return OFF

    def calculateCalibratorFlux(self, source, freqs):
        """
        All frequencies must be in MHz!
        """
        configline = self.getConfigLine(source)

        freqs = np.array(freqs)
        fluxes = np.zeros(len(freqs))
        #Format 2, Flux in Jy for a frequency in GHz is: log10(S) = a_0 + a_1*log10(f) + a_2*(log10(f))^2 + ...
        if configline[0][0] == "&": 
            logfreqs = np.log10(freqs/1000.0)
            coeffs = fmap(float, configline[3:])
            for i, coeff in enumerate(coeffs):
                fluxes += coeff*np.power(logfreqs, i)
        else:
            freq = float(configline[3]) #MHz
            flux = float(configline[4]) #Jy
            index = float(configline[5])

            fluxes = flux*np.power((freqs/freq), -1*index)
        return fluxes
