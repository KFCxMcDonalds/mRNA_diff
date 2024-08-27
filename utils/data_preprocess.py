import os
import subprocess

from Bio import SeqIO

def remove_redundancy(input_fasta, output_fasta, redundancy):
    # conda install -c bioconda cd-hit
    cd_hit_command = [
        "cd-hit",
        "-i", input_fasta,
        "-o", output_fasta,
        "-c", redundancy
    ]
    subprocess.run(cd_hit_command)
    print(f"cd-hit redandant completed, saves to {output_fasta}")

def length_filter(min, max, input_fasta, output_fasta):
    sequences = list(SeqIO.parse(input_fasta, "fasta"))
    filtered_sequences = [record for record in sequences if min <= len(record.seq) <= max]
    num_filtered_sequences = len(filtered_sequences)
    with open(output_fasta, 'w') as output_handle:
        for record in filtered_sequences:
            header = f">{record.description}\n"
            sequence = str(record.seq) + "\n"
            output_handle.write(header)
            output_handle.write(sequence)

    print(f"length filter completed, saves to {output_fasta}\n {num_filtered_sequences} sequences left.")

def read_and_pad(file_fasta, max_length=512):
    # save to origin file
    sequences = []
    for record in SeqIO.parse(file_fasta, "fasta"):
        seq = str(record.seq)
        # padding to maximum length of dataset, use '*' as padding signal
        if len(seq) < max_length:
            seq += '*' * (max_length - len(seq))
        sequences.append(seq[:max_length])
    return sequences

def onehot_encoder(input_fasta, output_fasta):
    encoding = {''}




if __name__ == "__main__":
    # remove redundancy
    sims = [80, 90]
    for sim in sims:

        root = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.dirname(root) + "/data/"

        remove_redundancy(data_dir+"5utr_full.fasta", data_dir+"5utr_"+str(sim)+".fasta", redundancy=str(sim/100))



