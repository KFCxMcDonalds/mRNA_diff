import os
import subprocess
import tempfile

import torch
import numpy as np

from tqdm import tqdm

from Bio import SeqIO

def remove_redundancy(input_fasta, output_fasta, redundancy):
    # conda install -c bioconda cd-hit
    cd_hit_command = [
        "cd-hit",
        "-i", input_fasta,
        "-o", output_fasta,
        "-c", str(redundancy/100)
    ]
    subprocess.run(cd_hit_command)
    print(f"cd-hit redandancy removing completed. Saved to {output_fasta}")

def cleaner(input_fasta):
    with tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
        for record in SeqIO.parse(input_fasta, "fasta"):
            seq = str(record.seq).upper()
            # delete sequences with unknown nucleotides 'N'
            if 'N' not in seq:
                header = f">{record.description}\n"
                sequence = seq + "\n"
                temp_file.write(header)
                temp_file.write(sequence)
        
        temp_file_name = temp_file.name
    
    os.replace(temp_file_name, input_fasta)
    
    print("sequences with nucleotides other than ATCG has been deleted.")


def length_filter(min, max, input_fasta, type):
    output_fasta = input_fasta.split('.')[0] + "_" + str(min) + "to" + str(max) + ".fasta"

    with open(output_fasta, 'w') as output_handle:
        num_filtered_sequences = 0
        for record in SeqIO.parse(input_fasta, "fasta"):
            seq = str(record.seq)
            if len(seq) >= min:
                if type == '5utr':
                    if len(seq) > max:
                        seq = seq[-max:]
                elif type in ['3utr', 'cds']:
                    if len(seq) > max:
                        seq = seq[:max]
                num_filtered_sequences += 1
                header = f">{record.description}\n"
                sequence = seq + "\n"
                output_handle.write(header)
                output_handle.write(sequence)

    print(f"length filter completed. Saved to {output_fasta}\n{num_filtered_sequences} sequences left.")

def read_and_pad(sequences, max_length):
    # will be called by onehot_encoder
    new_seqs = []
    for seq in sequences:
        # padding to maximum length of dataset, use '*' as padding signal
        if len(seq) < max_length:
            seq += '*' * (max_length - len(seq))
        new_seqs.append(seq[:max_length])

    return new_seqs

def onehot_encoder(input_fasta):
    # padding
    sequences = []
    for record in SeqIO.parse(input_fasta, "fasta"):
        seq = str(record.seq)
        sequences.append(seq)

    max_length = max([len(s) for s in sequences])
    sequences = read_and_pad(sequences, max_length)
    print("end of padding")

    # store in temple file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
        for seq in sequences:
            temp_file.write((seq + '\n').encode())

    # one-hot encoder
    encoding = {
        "A": [1, 0, 0, 0, 0],
        "T": [0, 1, 0, 0, 0],
        "G": [0, 0, 1, 0, 0],
        "C": [0, 0, 0, 1, 0],
        "*": [0, 0, 0, 0, 1],  # padding symbol
    }
    onehot = np.zeros((len(sequences), max_length, 5), dtype=np.float32)
    print("onehot matrix created.")

    with open(temp_filename, 'r') as temp_file:
        for i, line in tqdm(enumerate(temp_file), total=len(sequences)):
            s = line.strip()
            for j, el in enumerate(s):
                onehot[i, j] = encoding[el]
    os.remove(temp_filename)

    ohe_tensor = torch.tensor(onehot, dtype=torch.float32)
    ohe_trans_tensor = torch.transpose(ohe_tensor, 1, 2)
    print(ohe_trans_tensor.shape)
    # save to file
    dir = os.path.dirname(input_fasta)
    filename = os.path.basename(input_fasta)
    output_pt = dir + "/ohe_" + filename.split('.')[0] + ".pt"
    print(output_pt)
    torch.save(ohe_trans_tensor, output_pt)

    print(f"onehot encoding completed. Saved to {output_pt}.\n{len(sequences)} sequences left.")

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.dirname(root) + "/data/"

#     # remove redundancy
#     sims = [95]
#     for sim in sims:
#         remove_redundancy(data_dir+"5utr_full.fasta", data_dir+"5utr_"+str(sim)+".fasta", redundancy=sim)
    
#     # sequences clean: unknown nucleotides except ATCG
#     cleaner(data_dir+ "5utr_95.fasta")

    # length filter
    minL = 64
    maxL = 256
    length_filter(minL, maxL, data_dir+"5utr_95.fasta", type="5utr")

    # onehot encoder
    onehot_encoder(f"{data_dir}5utr_95_{minL}to{maxL}.fasta")
    



