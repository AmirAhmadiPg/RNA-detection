from Bio import SeqIO
import re
import matplotlib.pyplot as plt

def orf_finder(file_name):
    ORF = []
    orf_length = []
    orf_ratio = []
    orf_targets = []
    orf_cRNA = []
    orf_ncRNA = []
    orf_cRNA_length = []
    orf_ncRNA_length = []
    with open(file_name) as fn:
        for record in SeqIO.parse(fn, 'fasta'):
            stop_to_stop_codons = re.split(r'TAA|TAG|TGA', str(record.seq))
            # print(stop_to_stop_codons)
            codon_len = [len(seq) for seq in stop_to_stop_codons]
            codon_len = codon_len[1:-2]
            # print(codon_len)
            for loop in codon_len:
                index_max = codon_len.index(max(codon_len))
                if (max(codon_len) % 3 == 0) & (max(codon_len) != 0):
                    orf = stop_to_stop_codons[index_max + 1]
                    # print(orf)
                    break
                codon_len[index_max] = 0
                orf = '0'
                # print(codon_len[index_max])
            # orf = torch.tensor(orf, dtype=torch.long)
            ORF.append(orf)
            orf_length.append(codon_len[index_max])
            orf_ratio.append(codon_len[index_max]/len(record.seq))
            #label = 0 if 'CDS' in record.id else 1
            if 'CDS' in record.id:
                label = 0
                orfc = orf
                orfc_length = codon_len[index_max]
            else:
                label = 1
                orfnc = orf
                orfnc_length = codon_len[index_max]
            orf_ncRNA.append(orfnc)
            orf_ncRNA_length.append(orfnc_length)

            orf_targets.append(label)
            # orf_targets = torch.tensor(label, dtype=torch.long)
        orf_lst = [ORF, orf_targets, orf_length, orf_ratio, orf_cRNA, orf_ncRNA, orf_cRNA_length, orf_ncRNA_length]
        return orf_lst


[ORF, orf_targets, orf_length, orf_ratio, orf_cRNA, orf_ncRNA, orf_cRNA_length, orf_ncRNA_length] = orf_finder(H_train)

# the histogram of the orf
n, bins, patches = plt.hist(orf_cRNA_length, 50, density=True, facecolor='g', alpha=0.75)

plt.xlabel('ORF length')
plt.ylabel('number')
plt.title('length distribution of ORF for ncRNA')
plt.grid(True)
plt.show()