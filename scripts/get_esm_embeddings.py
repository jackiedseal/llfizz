import argparse
import pandas as pd
import torch
import esm
from tqdm import tqdm

MAX_SEQUENCE_LENGTH = 1023

def main(input_csv, output_csv):
    # Load sequences from CSV 
    df = pd.read_csv(input_csv)
    df['sequence'] = df['sequence'].str.strip()

    initial_count = len(df)

    df = df[df['sequence'].str.len() <= MAX_SEQUENCE_LENGTH].reset_index(drop=True)
    print(df['sequence'], df['sequence'].str.len().max())
    sequences = list(zip(df['entry'], df['sequence']))

    # Save how many sequences remain
    final_count = len(df)

    # Print how many were dropped
    print(f"Dropped {initial_count - final_count} sequences longer than {MAX_SEQUENCE_LENGTH} residues.")

    # Load ESM model
    # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D() # ESM-2
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S() # ESM-1b
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    # Process sequences in batches
    batch_size = 32
    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size), desc="Processing sequences", unit="batch"):
            batch_seqs = sequences[i:i+batch_size]
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_seqs)

            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]

            for j, (_, seq) in enumerate(batch_seqs):
                # Mean over non-special tokens
                embedding = token_representations[j, 1:len(seq)+1].mean(0)
                all_embeddings.append(embedding.cpu())

    # Convert to DataFrame and save
    embedding_array = torch.stack(all_embeddings)
    np_array = embedding_array.detach().cpu().numpy()
    output_df = pd.DataFrame({
        "entry": df['entry'],
        "label": df['label']
    })
    output_df = pd.concat([output_df, pd.DataFrame(np_array)], axis=1)

    output_path = output_csv
    output_df.to_csv(output_path, index=False)
    print(f"Embeddings saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract ESM embeddings from sequences in a CSV.")
    parser.add_argument("input_csv", type=str, help="Path to input CSV file with columns: entry, sequence, label")
    parser.add_argument("output_csv", type=str, help="Path to output CSV file to save embeddings")
    args = parser.parse_args()
    main(args.input_csv, args.output_csv)

