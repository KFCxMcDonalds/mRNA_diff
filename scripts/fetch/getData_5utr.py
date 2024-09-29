import time
import os

from loguru import logger
from Bio import Entrez, SeqIO

# 设置Entrez的邮箱地址
Entrez.email = "liwen.wu0821@outlook.com"

# params
mRNA_ids = []
total_num = 37251
batch_size = total_num  # query number every batch
start = 0
search_term = "mRNA[Filter] AND 5'UTR[Feature Key]"
# store path
log_file_path = "../data/5utr_id_log.txt"
id_file_path = "../data/5utr_ids.txt"
# set start param
if os.path.exists(log_file_path):
    # restart = start(logged)
    with open(log_file_path, "r") as log:
        lines = log.readlines()
        if lines:
            last_line = lines[-1].strip()
            start = int(last_line.split("=")[1])
        else:
            start = 0
else:
    start = 0

# existing ids
existing_ids = set()
if os.path.exists(id_file_path):
    with open(id_file_path, "r") as id_file:
        for line in id_file:
            existing_ids.add(line.strip())

while True:
    if len(existing_ids) == total_num:
        break
    try:
        print(f"start fetching at {start}\n {len(existing_ids) / total_num * 100:.2f}% completed.\n")
        handle = Entrez.esearch(db="nucleotide", term=search_term, restart=start, retmax=batch_size)
        record = Entrez.read(handle)
        handle.close()

        # obtain ids
        ids = record["IdList"]
        if not ids:
            break

        # non-duplicate
        new_ids = set(ids) - existing_ids
        if new_ids:
            with open(id_file_path, "a") as id_file:
                for new_id in new_ids:
                    id_file.write(new_id + "\n")
            existing_ids.update(new_ids)