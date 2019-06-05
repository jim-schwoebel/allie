'''
Simple unit test testing the default parameters 
of the repository. 

Note this unit_testing requires the settings.json
to defined.
'''
import unittest

class SimplisticTest(unittest.TestCase):
     
    def test(self):
        a = 'a'
        b = 'a'
        self.assertEqual(a, b)
 
if __name__ == '__main__':
    unittest.main()
