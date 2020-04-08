'''
Michael Lam 2020
Unit tests for par.py
'''

import unittest
import pypulse.par as par

class TestParameter(unittest.TestCase):
    """ Paramaeter testing class """

    @classmethod
    def setUpClass(cls):
        cls.filename = "data/J1909-3744_NANOGrav_11yv1.gls.par"
        cls.par = par.Par(cls.filename)
        cls.param = cls.par.paramlist[5] # PX
        cls.ecorr = cls.par.paramlist[-5]

    def test_getName(self):
        """ Test getName() """
        self.assertEqual(self.param.getName(), "PX")

    def test_getValue(self):
        """ Test getValue() """
        self.assertEqual(self.param.getValue(), 0.9177)

    def test_getFit(self):
        """ Test getFit() """
        self.assertEqual(self.param.getFit(), 1)
        
    def test_getError(self):
        """ Test getError() """
        self.assertEqual(self.param.getError(), 0.0295)

    def test_getFlag(self):
        """ Test getFlag() """
        self.assertEqual(self.ecorr.getFlag(), "-f")

    def test_getFlagValue(self):
        """ Test getFlagValue() """
        self.assertEqual(self.ecorr.getFlagValue(), "Rcvr_800_GUPPI")

    def test_hasFlag(self):
        """ Test hasFlag() """
        self.assertTrue(self.ecorr.hasFlag())


class TestPar(unittest.TestCase):
    """ Par testing class """

    @classmethod
    def setUpClass(cls):
        cls.filename = "data/J1909-3744_NANOGrav_11yv1.gls.par"
        cls.par = par.Par(cls.filename)

    def test_get(self):
        """ Test get() """
        values = [0.00370, 0.07670, 0.00590, 0.06000]
        for i, val in enumerate(self.par.get("ECORR")):
            self.assertEqual(values[i], val)


    #def test_getParameterFlags(self):
    #    """ Test getFlags() """
    #    print(self.par.getParameterFlags("ECORR"))

    def test_getName(self):
        """ Test getName() """
        self.assertEqual(self.par.getName(), "J1909-3744")

    def test_getPX(self):
        """ Test getPX() and getPX(error=True) """
        self.assertEqual(self.par.getPX(), 0.9177)
        self.assertEqual(self.par.getPX(error=True), 0.0295)

    def test_getDM(self):
        """ Test getDM() """
        self.assertEqual(self.par.getDM(), 10.391362)


    def test_getTspan(self):
        """ Test getTspan() """
        self.assertAlmostEqual(self.par.getTspan(), 4094.783)
        self.assertAlmostEqual(self.par.getTspan(years=True), 11.2109,
                               places=4)
        
        
if __name__ == '__main__':
    unittest.main()
