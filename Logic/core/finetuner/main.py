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