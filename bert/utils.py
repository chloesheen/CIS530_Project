import time
import datetime
import os

from config import PATH

def mkdir(dir):
    """Create a directory, ignoring exceptions

    # Arguments:
        dir: Path of directory to create
    """
    try:
        os.mkdir(dir)
    except:
        pass

def setup_dirs():
    """
    Create directories for this project.
    """
    mkdir(PATH + "/logs/")
    mkdir(PATH + "/models/")

def convert_example_to_features(example, model_type, tokenizer, max_len=512):
    """
    Takes in a text example and converts it into input IDs and input mask.
    """
    input_ids = tokenizer.encode(example,
                                 add_special_tokens=True,
                                 max_length=max_len,
                                 pad_to_max_length=True)
    input_mask = [int(input_id > 0) for input_id in input_ids]
    return input_ids, input_mask

def accuracy_score(preds, labels):
    """
    Calculates accuracy score for a single batch.
    """
    preds = preds.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
