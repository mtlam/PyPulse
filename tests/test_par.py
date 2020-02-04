'''
Michael Lam 2020
Unit tests for par.py
'''

import unittest
import pypulse.par as par

class TestPar(unittest.TestCase):
    """ Parameter testing class """

    @classmethod
    def setUpClass(cls):
        cls.filename = "data/J1909-3744_NANOGrav_11yv1.gls.par"
        cls.par = par.Par(cls.filename)

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


if __name__ == '__main__':
    unittest.main()
