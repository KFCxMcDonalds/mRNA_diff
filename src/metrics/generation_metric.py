import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO

def gc_content_dist(nature_seqs, generated_seqs):

    def calculate_gc_content(seq):
        gc_count = seq.count('G') + seq.count('C')
        return gc_count / len(seq) if len(seq) > 0 else 0

    nature_gc_contents = [calculate_gc_content(seq) for seq in nature_seqs]
    generated_gc_contents = [calculate_gc_content(seq) for seq in generated_seqs]

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=[nature_gc_contents, generated_gc_contents], scale='width', inner="quartile")
    plt.title('GC Content Distribution Comparison')
    plt.xlabel('Sequence Type')
    plt.ylabel('GC Content')
    plt.xticks([0, 1], ['Natural Sequences', 'Generated Sequences'])
    plt.savefig('gc_content_distribution_2048.png')
    plt.close()
    
    print("GC content distribution plot has been saved as gc_content_distribution.png")

    return nature_gc_contents, generated_gc_contents


if __name__ == "__main__":
    nature_file = "/home/liwenwu/files/mRNA_diff/data/5utr_95_64to256_PRI.fasta"
    generated_file = "/home/liwenwu/files/mRNA_diff/generation/5utr/.fasta"

    nature_seqs = [str(record.seq) for record in SeqIO.parse(nature_file, "fasta")]
    generated_seqs = [str(record.seq) for record in SeqIO.parse(generated_file, "fasta")]

    print(f"Read {len(nature_seqs)} natural sequences")
    print(f"Read {len(generated_seqs)} generated sequences")

    # gc content distribution
    gc_content_dist(nature_seqs, generated_seqs)



