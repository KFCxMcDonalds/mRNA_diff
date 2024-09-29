from Bio import Entrez, SeqIO
from Bio.Seq import Seq
import time


def get_id_list(id_file):
    res = []
    with open(id_file, "r") as file:
        for line in file:
            res.append(line.strip())
    return res


def get_feature_seqs(record, feature, feature_type):
    seq = []
    if len(feature.location.parts) == 1:
        return feature.extract(record.seq)
    else:
        for p in feature.location.parts:
            if not p.parts[0].ref:
                seq.append(p.extract(record.seq))
            else:
                ref = p.ref
                start = p.start
                end = p.end
                while True:
                    try:
                        h = Entrez.efetch(db="nucleotide", id=ref, rettype="gb", retmode="text")
                        break
                    except Exception as e:
                        print(f"HTTP error:{str(e)}\n! retry")
                        time.sleep(5)
                r = list(SeqIO.parse(h, "genbank"))[0]
                h.close()
                for feature in record.features:
                    if feature.type == feature_type:
                        seq.append(r.seq[start: end])
                        break
    res = seq[0]
    for s in seq[1:]:
        res += s

    return res

def fetch_sequences(id_list, output_file, feature_type, batch_size=9998):
    total_ids = len(id_list)
    total = 0

    for start in range(0, total_ids, batch_size):
        end = min(start + batch_size, total_ids)
        batch_ids = id_list[start:end]
        ids = ",".join(batch_ids)

        while True:
            try:
                handle = Entrez.efetch(db="nucleotide", id=ids, rettype="gb", retmode="text")
                records = list(SeqIO.parse(handle, "genbank"))
                handle.close()
                break
            except Exception as e:
                print(f"HTTP error:{str(e)}\n! retry")
                time.sleep(5)

        # 自定义FASTA记录的描述部分
        for record in records:
            if feature_type not in [i.type for i in record.features]:
                print("did not find required feature!")
            for feature in record.features:
                if feature.type == feature_type:
                    topology = record.annotations.get("topology", "N/A")
                    source = record.annotations.get("source", "N/A")
                    data_file_division = record.annotations.get("data_file_division", "N/A")
                    feature_seq = get_feature_seqs(record, feature, feature_type)
                    seq_length = len(feature_seq)
                    description = f"{seq_length} {topology} {data_file_division} | {source}"

                    fasta_record = SeqIO.SeqRecord(feature_seq, id=record.id, description=description)

                    # output to file
                    with open(output_file, "a") as output_handle:
                        output_handle.write(f">{fasta_record.id} {fasta_record.description}\n{str(fasta_record.seq)}\n")
                        total += 1
                        if total % 10000 == 0:
                            print(f"now fetched: {total}, percentage: {total / total_ids * 100:.2f}%")
                    break
    print(f"now fetched: {total}, percentage: {total / total_ids * 100:.2f}%")

Entrez.email = "liwen.wu0821@outlook.com"

# 5'UTR
utr5_file = "/home/liwenwu/files/mRNA_diff/data/raw_data/5utr_sequences.fasta"
utr5_id_file = "/home/liwenwu/files/mRNA_diff/data/raw_data/5utr_ids.txt"
utr5_ids = get_id_list(utr5_id_file)

print(f"fetching 5'UTR..." + f"\ntotal: {len(utr5_ids)}")
fetch_sequences(utr5_ids, utr5_file, feature_type="5'UTR", batch_size=10)

# # 3'UTR
# utr3_file = "../data/3utr_sequences.fasta"
# utr3_id_file = "../data/3utr_ids.txt"
# utr3_ids = get_id_list(utr3_id_file)
#
# print(f"fetching 3'UTR..." + f"\ntotal: {len(utr3_ids)}")
# fetch_sequences(utr3_ids, utr3_file, feature_type="3'UTR")

# cds
# cds_file = "../data/cds_sequences.fasta"
# cds_id_file = "../data/cds_ids.txt"
# cds_ids = get_id_list(cds_id_file)
#
# print(f"fetching CDS..." + f"\ntotal: {len(cds_ids)}")
# fetch_sequences(cds_ids, cds_file, feature_type="CDS")
#
# print("completed.")

