import torch


def tensor2rna(tensor):
    # (channel, length) -> seq
    mapping = {
        0: 'A',
        1: 'T',
        2: 'G',
        3: 'C',
        4: '*'
    }
    discrete_tensor = torch.argmax(tensor, dim=0)

    tensor_list = discrete_tensor.cpu().tolist()
    rna_sequence = ''.join(mapping[base] for base in tensor_list)

    return rna_sequence

def write2fasta(seqs, output_file):
    # seqs: list of strings
    with open(output_file, 'a') as f:
        for i, seq in enumerate(seqs):
            f.write(f">sequences_{i+1}\n{seq}\n")

