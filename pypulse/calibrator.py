import numpy as np





class Calibrator:
    def __init__(self,S,pol_type='Coherence',fd_poln='LIN'):
        self.pol_type = pol_type

        if self.pol_type == 'Coherence' or self.pol_type == 'AABBCRCI':
            A,B,C,D = S
            if fd_poln == 'LIN':
                S0 = A+B #I
                S1 = A-B #Q
                S2 = 2*C #U
                S3 = 2*D #V
        elif self.pol_type == 'Stokes' or self.pol_type == 'IQUV':
            S0,S1,S2,S3 = S
        else:
            raise SystemExit
            
        pass



    def calculate_PA(self,lat,dec,HA):
        """
        Helper function
        lat = latitude
        dec = declination of source
        HA = hour angle
        """
        return np.arctan2(np.sin(HA)*np.cos(lat),np.sin(lat)*np.cos(dec) - np.cos(lat)*np.sin(phi)*np.cos(HA))


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
        


