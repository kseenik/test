import unittest
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from model import preprocess_text, model, y_test, y_pred, y_attacked, y_attacked_pred


class TestSpamModel(unittest.TestCase):

    def test_preprocess_text(self):
        raw_text = "Check out the website https://example.com! This is the first text with number 123."
        expected_result = "check websit first text number"
        self.assertEqual(preprocess_text(raw_text), expected_result)

    
    def test_model_prediction(self):
        accuracy = accuracy_score(y_test, y_pred)
        self.assertGreater(accuracy, 0.8)  # Ожидаемая точность должна быть выше 80%

    
    def test_classification_report(self):
        report = classification_report(y_test, y_pred, output_dict=True)
        accuracy = report['accuracy']
        self.assertGreater(accuracy, 0.8)  # Проверка точности

    
    def test_confusion_matrix(self):
        conf_matrix = confusion_matrix(y_attacked, y_attacked_pred)
        self.assertEqual(conf_matrix.shape, (2, 2))  # Матрица ошибок должна быть 2x2

    
    def test_plot_confusion_matrix(self):
        conf_matrix = confusion_matrix(y_attacked, y_attacked_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
        plt.title('Confusion Matrix (Attacked Dataset)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()


if __name__ == "__main__":
    unittest.main()

