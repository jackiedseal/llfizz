import pandas as pd
import argparse

def csv_to_fasta(csv_file, fasta_file):
    """Convert a CSV file with a 'Sequence' column to a FASTA file, ignoring the header."""
    # Read CSV file, automatically handling headers
    df = pd.read_csv(csv_file)

    # Skip the first row (header is already handled, but just in case)
    sequences = df["Sequence"].tolist()  # Convert column to a list

    # Open FASTA file for writing
    with open(fasta_file, "w") as fasta:
        for index, sequence in enumerate(sequences, start=1):
            fasta.write(f">design_{index}\n{sequence}\n")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert a CSV file to a FASTA file, ignoring the header.")
    parser.add_argument("input_csv", help="Path to the input CSV file")
    parser.add_argument("output_fasta", help="Path to the output FASTA file")

    # Parse command-line arguments
    args = parser.parse_args()

    # Convert CSV to FASTA
    csv_to_fasta(args.input_csv, args.output_fasta)

if __name__ == "__main__":
    main()


