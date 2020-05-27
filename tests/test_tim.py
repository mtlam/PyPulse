'''
Michael Lam 2020
Unit tests for tim.py
'''

import unittest
import decimal
import pypulse.tim as tim


class TestTOA(unittest.TestCase):
    """ TOA testing class """

    @classmethod
    def setUpClass(cls):
        cls.filename = "data/J1909-3744_NANOGrav_11yv1.tim"
        cls.tim = tim.Tim(cls.filename, usedecimal=True) #numwrap only applies to MJD
        cls.toas = cls.tim.toas

    def test_getFilename(self):
        """ Test getFilename() """
        self.assertEqual(self.toas[-1].getFilename(), "guppi_57386_J1909-3744_0008.11y.x.ff")

    
    def test_getFreq(self):
        """ Test getFreq() """
        self.assertEqual(self.toas[-1].getFreq(), 1157.031006)
        
    def test_getMJD(self):
        """ Test getMJD() """
        self.assertEqual(self.toas[-1].getMJD(), decimal.Decimal("57386.798000084492063"))

        
    def test_getErr(self):
        """ Test getErr() """
        self.assertEqual(self.toas[-1].getErr(), 0.381)


    def test_get(self):
        """ Test get() """
        self.assertEqual(self.toas[-1].get("f"), "Rcvr1_2_GUPPI")
        self.assertEqual(float(self.toas[-1].get("bw")), 12.5)


class TestTim(unittest.TestCase):
    """ Tim testing class """

    @classmethod
    def setUpClass(cls):
        cls.filename = "data/J1909-3744_NANOGrav_11yv1.tim"
        cls.tim = tim.Tim(cls.filename, usedecimal=True)

    def test_getFilenames(self):
        """ Test getFilenames() """
        values = self.tim.getFilenames()
        self.assertEqual(values[-1], "guppi_57386_J1909-3744_0008.11y.x.ff")

    def test_getFreqs(self):
        """ Test getFreqs() """
        values = self.tim.getFreqs()
        self.assertEqual(float(values[-1]), 1157.031006)

    def test_getMJDs(self):
        """ Test getMJDs() """
        values = self.tim.getMJDs()
        self.assertEqual(values[-1], decimal.Decimal("57386.798000084492063"))

    def test_getErrs(self):
        """ Test getErrs() """
        values = self.tim.getErrs()
        self.assertEqual(float(values[-1]), 0.381)

    def test_get(self):
        """ Test get() """
        f = self.tim.get("f")
        bw = self.tim.get("bw", numwrap=float) #forced to be a float
        self.assertEqual(f[-1], "Rcvr1_2_GUPPI")
        self.assertEqual(bw[-1], 12.5)

    def test_getTspan(self):
        """ Test getTspan() """
        self.assertAlmostEqual(float(self.tim.getTspan()), 4094.78146, places=4)
        self.assertAlmostEqual(float(self.tim.getTspan(years=True)), 11.2109,
                               places=4)
        
if __name__ == '__main__':
    unittest.main()
