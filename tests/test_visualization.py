import unittest
from unittest.mock import patch
import matplotlib.pyplot as plt
from src.visualization import generate_word_cloud

class TestVisualization(unittest.TestCase):

    @patch('src.visualization.components.html')
    def test_generate_word_cloud(self, mock_html):
        keywords = [("Python", 1), ("programming", 0.5), ("language", 0.3)]
        plt.figure()
        try:
            generate_word_cloud(keywords)
            mock_html.assert_called_once()
            result = True
        except Exception as e:
            result = False
            print(f"Error generating word cloud: {e}")
        
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
