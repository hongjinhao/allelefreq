"""
Exports uniprotkb_human_ref_proteome_dict.pkl to a flat binary file the C++
prototype can read directly, with no text parsing.

Format (little-endian):
    uint32  num_proteins
    repeated num_proteins times:
        uint8   id_len
        char    id[id_len]          (ASCII, not null-terminated)
        uint32  seq_len
        char    seq[seq_len]        (ASCII, not null-terminated)
"""
import pickle
import struct
import sys

SRC = "../uniprotkb_human_ref_proteome_dict.pkl"
DST = "proteome.bin"

with open(SRC, "rb") as f:
    proteome = pickle.load(f)

with open(DST, "wb") as f:
    f.write(struct.pack("<I", len(proteome)))
    for protein_id, seq in proteome.items():
        id_bytes = protein_id.encode("ascii")
        seq_bytes = seq.encode("ascii")
        if len(id_bytes) > 255:
            sys.exit(f"protein id too long for uint8 length prefix: {protein_id}")
        f.write(struct.pack("<B", len(id_bytes)))
        f.write(id_bytes)
        f.write(struct.pack("<I", len(seq_bytes)))
        f.write(seq_bytes)

print(f"wrote {DST}: {len(proteome)} proteins")
