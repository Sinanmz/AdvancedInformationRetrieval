import os
import sys
current_script_path = os.path.abspath(__file__)
finetuner = os.path.dirname(current_script_path)
core_dir = os.path.dirname(finetuner)
Logic_dir = os.path.dirname(core_dir)
project_root = os.path.dirname(Logic_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from Logic.core.finetuner.BertFinetuner_mask import BERTFinetuner


# Instantiate the class
bert_finetuner = BERTFinetuner(project_root+'/data/IMDB_Crawled.json', top_n_genres=5)

# Load the dataset
bert_finetuner.load_dataset()

# Preprocess genre distribution
bert_finetuner.preprocess_genre_distribution()

# Split the dataset
bert_finetuner.split_dataset()

# Fine-tune BERT model
bert_finetuner.fine_tune_bert()

# Compute metrics
bert_finetuner.evaluate_model()

bert_finetuner.classification_report('train')
bert_finetuner.classification_report('val')
bert_finetuner.classification_report('test')

# Save the model (optional)
# bert_finetuner.save_model('Movie_Genre_Classifier')



# Outputs:

# Train Set Evaluation:
# Precision: 0.9997010463378176
# Recall: 0.9978657589997768
# F1: 0.9987818614285313
# Accuracy: 0.996624731512734
# -----------------------------------------
# Validation Set Evaluation:
# Precision: 0.6955069424571555
# Recall: 0.6417047695893849
# F1: 0.6637562698596289
# Accuracy: 0.4429447852760736
# -----------------------------------------
# Test Set Evaluation:
# Precision: 0.6745329025055461
# Recall: 0.6492931501308465
# F1: 0.659557649457462
# Accuracy: 0.45521472392638035
# -----------------------------------------
# Classification Report for Train Split:
#                precision    recall  f1-score   support

#       Action       1.00      1.00      1.00      1655
#        Drama       1.00      1.00      1.00      4109
#       Comedy       1.00      1.00      1.00      2094
#    Animation       1.00      1.00      1.00       669
#        Crime       1.00      1.00      1.00      1284

#    micro avg       1.00      1.00      1.00      9811
#    macro avg       1.00      1.00      1.00      9811
# weighted avg       1.00      1.00      1.00      9811
#  samples avg       1.00      1.00      1.00      9811

# Classification Report for Val Split:
#                precision    recall  f1-score   support

#       Action       0.70      0.73      0.71       220
#        Drama       0.77      0.84      0.80       507
#       Comedy       0.69      0.54      0.61       260
#    Animation       0.59      0.44      0.50        80
#        Crime       0.72      0.66      0.69       165

#    micro avg       0.73      0.71      0.72      1232
#    macro avg       0.70      0.64      0.66      1232
# weighted avg       0.72      0.71      0.71      1232
#  samples avg       0.75      0.74      0.71      1232

# Classification Report for Test Split:
#                precision    recall  f1-score   support

#       Action       0.62      0.66      0.64       191
#        Drama       0.80      0.85      0.82       520
#       Comedy       0.69      0.58      0.63       260
#    Animation       0.60      0.49      0.54        78
#        Crime       0.65      0.67      0.66       154

#    micro avg       0.72      0.71      0.72      1203
#    macro avg       0.67      0.65      0.66      1203
# weighted avg       0.72      0.71      0.71      1203
#  samples avg       0.75      0.75      0.72      1203