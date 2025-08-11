import pandas as pd 

def load_benchmark_data(file_path):
    """
    Load benchmark data from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing benchmark data.

    Returns:
        pd.DataFrame: DataFrame containing the benchmark data.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def calculate_bleu_score(reference, hypothesis):
    """
    Calculate the BLEU score between a reference and a hypothesis.

    Args:
        reference (str): The reference text.
        hypothesis (str): The hypothesis text.

    Returns:
        float: The BLEU score.
    """
    from nltk.translate.bleu_score import sentence_bleu
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    return sentence_bleu([reference_tokens], hypothesis_tokens)


text1 = "The quick brown fox jumps over the lazy dog."
text2 = "The fast brown fox leaps over the lazy dog."
bleu_score = calculate_bleu_score(text1, text2)
print(f"BLEU score between the two texts: {bleu_score:.4f}")