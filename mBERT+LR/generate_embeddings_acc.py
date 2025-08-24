import pandas as pd
import numpy as np
import torch
# Use AutoModel for feature extraction
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
PRETRAINED_MODEL_NAME = 'bert-base-multilingual-cased'
INPUT_CSV_PATH = "dt1.csv"
BATCH_SIZE = 16

# --- OUTPUT FILES  ---
EMBEDDINGS_OUTPUT_PATH = 'embeddings_accuracy.npy'
LABELS_OUTPUT_PATH = 'labels_accuracy.csv'


def generate_cls_embeddings(texts, model, tokenizer, device):
    """
    Generates [CLS] token embeddings for a batch of texts.
    """
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the [CLS] token embedding (it's the first token in the sequence)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def main():
    """
    Main function to load data, generate embeddings for accuracy analysis, and save files.
    """
    # 1. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Load Pretrained Model and Tokenizer
    print(f"Loading tokenizer and model: {PRETRAINED_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    model = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME).to(device)
    model.eval()  # Set model to evaluation mode

    # 3. Load and Prepare Data (Focus on accuracy)
    print(f"Loading data from {INPUT_CSV_PATH}...")
    df = pd.read_csv(INPUT_CSV_PATH)

    # Drop rows where 'text' or 'accuracy' columns have missing values
    df = df.dropna(subset=['text', 'accuracy']).copy()

    # Clean the text and accuracy columns
    df['text'] = df['text'].astype(str).str.strip()
    df['accuracy'] = df['accuracy'].astype(
        str).str.strip().str.lower()

    # Filter out any rows where the accuracy label is 'na'
    df = df[df['accuracy'] != 'na'].reset_index(drop=True)

    texts = df['text'].tolist()
    labels = df[['accuracy']]

    print(f"Found {len(texts)} valid text samples for accuracy analysis.")

    # 4. Generate Embeddings in Batches
    all_embeddings = []
    print(f"Generating embeddings with batch size {BATCH_SIZE}...")

    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Generating Embeddings"):
        batch_texts = texts[i: i + BATCH_SIZE]
        batch_embeddings = generate_cls_embeddings(
            batch_texts, model, tokenizer, device)
        all_embeddings.append(batch_embeddings)

    final_embeddings = np.vstack(all_embeddings)

    print(f"Embeddings generated with shape: {final_embeddings.shape}")

    # 5. Save the Embeddings and Labels
    np.save(EMBEDDINGS_OUTPUT_PATH, final_embeddings)
    print(f"Embeddings saved to: {EMBEDDINGS_OUTPUT_PATH}")

    labels.to_csv(LABELS_OUTPUT_PATH, index=False)
    print(f"accuracy labels saved to: {LABELS_OUTPUT_PATH}")


if __name__ == '__main__':
    main()
