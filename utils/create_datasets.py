#!/usr/bin/env python

"""
create_datasets.py  –  Build SpliceAI‑style training data from NCBI GRCh38.p14 genome & GFF

This script:
  1. Downloads (or reads locally) the GRCh38.p14 FASTA and matching GFF from NCBI.
  2. Parses the GFF to extract all protein‑coding transcript ranges (mRNA) and their exon coordinates.
  3. For each transcript, extracts the full pre‑mRNA (introns + exons) sequence from genome.
  4. One‑hot encodes, pads to a multiple of 5,000 bp, adds flanking context S (5000) on both ends.
  5. Splits into blocks of length (5,000 + 2·S), generates per‑block donor/acceptor labels as (5000, 3) array
  6. Splits transcripts into train/val/test by chromosome (SpliceAI scheme) and writes HDF5.

Usage:
  python create_datasets.py \
    --fasta   GRCh38.p14_genomic.fna.gz \
    --gff     GCF_000001405.40_GRCh38.p14_genomic.gff.gz \
    --chrom   GCF_000001405.40_GRCh38.p14_assembly_report.txt \
    --outdir  data/spliceai_grch38p14 \
    --flank   5000

Requirements:
  Python 3.7+, numpy, pyfaidx, tqdm, biopython, pandas, requests
"""
import os
import gzip
import argparse
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
from pyfaidx import Fasta
from tqdm import tqdm
from Bio.Seq import Seq
import pandas as pd
import logging
import pickle
import subprocess

# default NCBI URLs
FASTA_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/"
    "GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz"
)
GFF_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/"
    "GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.gff.gz"
)

CHROM_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/"
    "GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_assembly_report.txt"
)


def download_if_needed(path, url):
    path = Path(path)
    if not path.exists():
        logging.info(f"Downloading {url} to {path}")
        print(f"Downloading {url} to {path}")

        urlretrieve(url, str(path))
        path = url.split
    return str(path)


def parse_gff(gff_path):
    """
    Parse GFF to return dict: {tx_id: {'chrom','strand','tx_start','tx_end','exons':[(start,end),...]}}
    Uses 'mRNA' features for transcript coords, 'exon' features to collect exon positions.
    """
    transcripts = {}
    with gzip.open(gff_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.strip().split("\t")
            chrom, src, feat, start, end, score, strand, phase, attr = cols
            attrs = dict(item.split("=", 1) for item in attr.split(";") if "=" in item)
            tx_id = attrs.get("transcript_id")
            parent = attrs.get("Parent")
            if feat == "mRNA":
                transcripts[tx_id] = {
                    "chrom": chrom,
                    "strand": strand,
                    "tx_start": int(start),
                    "tx_end": int(end),
                    "exons": [],
                    "parent": parent,
                }
            elif feat == "exon" and tx_id in transcripts:
                transcripts[tx_id]["exons"].append((int(start), int(end)))

        # multiple transcript share the same parent
        # keep only one with longest exons list
        transcripts_unique = {}
        for tx_id, meta in transcripts.items():
            if meta["parent"] in transcripts_unique:
                if len(meta["exons"]) > len(
                    transcripts_unique[meta["parent"]]["exons"]
                ):
                    transcripts_unique[meta["parent"]] = meta
            else:
                transcripts_unique[meta["parent"]] = meta

    return transcripts_unique


def one_hot_encode(seq):
    """
    One-hot encode a nucleotide sequence.
    A=0, C=1, G=2, T=3
    """
    nt_map = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
    L = len(seq)
    oh = np.zeros((L, 4), dtype=np.uint8)
    for i, base in enumerate(seq):
        idx = nt_map.get(base)
        if idx is not None:
            oh[i, idx] = 1
    return oh


def encode_and_block(transcripts, fasta, chromosome_map, flank, block_len=5000):
    """
    For each transcript (pre-mRNA), extract sequence, one-hot encode, pad & block.
    Yields dicts: {'chrom','tx_id','block_idx','sequence', 'y'}
    """
    for tx_id, meta in tqdm(transcripts.items()):
        chrom = meta["chrom"]
        strand = meta["strand"]
        tx_s = meta["tx_start"]
        tx_e = meta["tx_end"]
        exons = meta["exons"]
        chrom_id = chromosome_map[chrom]
        # skip MT
        if chrom_id == "MT":
            continue
        # extract full pre-mRNA (includes introns)
        raw = fasta[chrom][tx_s - 1 : tx_e].seq.upper()
        if strand == "-":
            raw = str(Seq(raw).reverse_complement())
        L = len(raw)
        # one-hot encode
        oh = one_hot_encode(raw)
        # pad to multiple of block_len
        pad = (-L) % block_len
        if pad:
            oh = np.pad(oh, ((0, pad), (0, 0)), constant_values=0)
        # add flank
        S = flank
        oh = np.pad(oh, ((S, S), (0, 0)), constant_values=0)
        total = oh.shape[0]
        # build splice site labels on raw (before padding): indices relative to pre-pad
        donor_idxs = []
        acceptor_idxs = []
        for s, e in exons:
            # positions relative to tx_start
            if strand == "+":
                a = s - tx_s
                d = e - tx_s
            elif strand == "-":
                d = L - (s - tx_s) - 1
                a = L - (e - tx_s) - 1
            if (d == L - 1) or (a == 0) or (s == e):
                continue
            assert (
                a != d
            ), f"Donor and acceptor at same position {a} {d} for {tx_id},\
                  {donor_idxs}, {acceptor_idxs}, {exons}, {strand}, {tx_s}, {tx_e}, {chromosome_map[chrom]}, {s}, {e}"
            assert (
                0 < a < L
            ), f"Acceptor {a} out of bounds for {tx_id}, {chromosome_map[chrom]}, {s}, {e}, {strand}, {tx_s}, {tx_e}, {raw}"
            assert 0 < d < L, f"Donor {d} out of bounds for {tx_id}"
            donor_idxs.append(d)
            acceptor_idxs.append(a)
        # print("Length", L)
        # print("START", tx_s)
        # print(f"Donor: {donor_idxs}, Acceptor: {acceptor_idxs}")
        # block slicing
        idx = 0
        blk_idx = 0
        while idx + block_len + 2 * S <= total:
            block = oh[idx : idx + block_len + 2 * S]
            length = block_len + 2 * S
            y_donor = np.zeros(length, dtype=np.uint8)
            y_acceptor = np.zeros(length, dtype=np.uint8)
            # assign labels
            for d in donor_idxs:
                rel = d + S - idx
                if 0 <= rel < length:
                    y_donor[rel] = 1
            for a in acceptor_idxs:
                rel = a + S - idx
                if 0 <= rel < length:
                    y_acceptor[rel] = 1

            y_neither = 1 - y_donor - y_acceptor
            # stack 3 y_* to single output
            y_out = np.stack([y_neither, y_donor, y_acceptor], axis=0)
            # trim Y_out to length block_len only, trim S start and end each
            y_out = y_out[:, S : S + block_len].T

            yield {
                "chrom": chrom_id,
                "tx_id": tx_id,
                "block_idx": blk_idx,
                "sequence": block,
                "y": y_out,
            }
            idx += block_len
            blk_idx += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta", default=None, help="Path or URL for GRCh38.p14 FASTA")
    p.add_argument("--gff", default=None, help="Path or URL for GFF")
    p.add_argument("--chrom", default=None, help="Path or URL for chromosome mapping")
    p.add_argument("--flank", type=int, default=5000)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    fasta_path = download_if_needed(
        args.fasta or "GRCh38.p14_genomic.fna.gz", FASTA_URL
    )
    gff_path = download_if_needed(args.gff or "GRCh38.p14_genomic.gff.gz", GFF_URL)
    chrom_path = download_if_needed(
        args.chrom or "GCF_000001405.40_GRCh38.p14_assembly_report.txt", CHROM_URL
    )

    logging.info(f"Loading chromosome mapping from {chrom_path}")
    df_chrom = pd.read_csv(chrom_path, sep="\t", comment="#", header=None)
    chromosome_map = df_chrom[[2, 6]].set_index(6).to_dict()[2]

    logging.info(f"Loading FASTA from {fasta_path}")
    # unzip if needed with gunzip
    if fasta_path.endswith(".gz"):
        logging.info(f"Unzipping {fasta_path}")
        subprocess.run(["gunzip", fasta_path])
        fasta_path = fasta_path[:-3]

    fasta = Fasta(fasta_path)

    logging.info(f"Parsing GFF from {gff_path}")
    transcripts = parse_gff(gff_path)
    logging.info(f"Parsed {len(transcripts)} transcripts")

    logging.info("Encoding and blocking transcripts")
    all_recs = list(
        encode_and_block(transcripts, fasta, chromosome_map, flank=args.flank)
    )
    logging.info(f"Encoded and blocked {len(all_recs)} records")
    # split transcripts by chromosome for train/val/test
    # train 2, 4, 6, 8, 10-22, X, Y
    # val 10% of train txs (get all blocks of a transcript)
    # test 1, 3, 5, 7, 9
    train_chrs = (
        set([str(i) for i in range(10, 23)])
        | {"X", "Y"}
        | set([str(i) for i in range(2, 10, 2)])
    )
    test_chrs = set([str(i) for i in range(1, 10, 2)])
    train = [r for r in all_recs if r["chrom"] in train_chrs]
    test = [r for r in all_recs if r["chrom"] in test_chrs]
    train_tx = set([r["tx_id"] for r in train])
    test_tx = set([r["tx_id"] for r in test])
    # get 10% of train transcripts
    val_tx = set(
        np.random.choice(list(train_tx), size=int(len(train_tx) * 0.1), replace=False)
    )
    val = [r for r in train if r["tx_id"] in val_tx]
    train = [r for r in train if r["tx_id"] not in val_tx]

    logging.info(
        f"Train: {len(train)} blocks from {len(train_tx)} transcripts from {len(train_chrs)} chromosomes"
    )
    logging.info(
        f"Val:   {len(val)} blocks from {len(val_tx)} transcripts from {len(train_chrs)} chromosomes"
    )
    logging.info(
        f"Test:  {len(test)} blocks from {len(test_tx)} transcripts from {len(test_chrs)} chromosomes"
    )

    # save as pickles
    logging.info(f"Saving train {len(train)} blocks to {args.outdir}/train.pkl")
    with open(os.path.join(args.outdir, "train.pkl"), "wb") as f:
        pickle.dump(train, f)
    logging.info(f"Saving val {len(val)} blocks to {args.outdir}/val.pkl")
    with open(os.path.join(args.outdir, "val.pkl"), "wb") as f:
        pickle.dump(val, f)
    logging.info(f"Saving test {len(test)} blocks to {args.outdir}/test.pkl")
    with open(os.path.join(args.outdir, "test.pkl"), "wb") as f:
        pickle.dump(test, f)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
