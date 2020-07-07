import unittest
import nlp

class TestNLPMethods(unittest.TestCase):

    def test_normalization(self):
        text = [['Июль', 'был', 'жарким', '.'],['Весна', '-', 'теплой', '.']]

        self.assertEqual(nlp.normalization(text), [['июль', 'жарким'],['весна', 'теплой']])


if __name__ == '__main__':
    unittest.main()