import random
import sys

from llfizz.constants import AMINOACIDS

# COX15 is 45AAs
# DED1 CTD is 72AAs
# DED1 NTD is 55AAs


def generate_protein_sequence(length):
    # return "M" + "".join(random.choices(AMINOACIDS, k=(length - 1)))
    return "".join(random.choices(AMINOACIDS, k=(length)))


def write_fasta_file(output_file, seq_length, num_sequences=100):
    with open(output_file, "w") as f:
        for i in range(num_sequences):
            sequence = generate_protein_sequence(seq_length)
            f.write(f">Sequence_{i+1}\n{sequence}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <sequence_length> <output_fasta_file>")
        sys.exit(1)

    seq_length = int(sys.argv[1])
    output_file = sys.argv[2]
    write_fasta_file(output_file, seq_length)
