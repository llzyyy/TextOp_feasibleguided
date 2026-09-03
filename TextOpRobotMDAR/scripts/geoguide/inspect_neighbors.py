"""Inspect semantic/geometric neighbors for a saved latent-bank entry."""

import argparse
import sys
from pathlib import Path


ROBOTMDAR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROBOTMDAR_ROOT))

from robotmdar.guidance import LatentBank  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--mode", default="semantic_then_geometry")
    parser.add_argument("--semantic-top-m", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=16)
    args = parser.parse_args()
    bank = LatentBank.load(args.bank, args.metadata)
    query_text = None if bank.text_embeddings is None else bank.text_embeddings[args.query_index]
    result = bank.query(
        bank.latents[args.query_index],
        mode=args.mode,
        top_k=args.top_k,
        query_text_embedding=query_text,
        semantic_top_m=args.semantic_top_m,
    )
    for rank, index in enumerate(result.indices[0].tolist()):
        text = "" if bank.texts is None else bank.texts[index]
        semantic = None if result.semantic_similarities is None else float(result.semantic_similarities[0, rank])
        print({"rank": rank, "index": index, "latent_l2": float(result.latent_distances[0, rank]), "semantic": semantic, "text": text})


if __name__ == "__main__":
    main()
