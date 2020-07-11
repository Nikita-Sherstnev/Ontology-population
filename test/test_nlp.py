import unittest
from main import nlp


class TestNLPMethods(unittest.TestCase):

    def test_normalization(self):
        text = [['Июль', 'был', 'жарким', '.'], ['Весна', '-', 'теплой', '.']]
        expected_text = [['июль', 'жарким'], ['весна', 'теплой']]
        self.assertEqual(nlp.normalization(text), expected_text)


if __name__ == '__main__':
    unittest.main()
