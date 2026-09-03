"""Save Local PCA spectrum and basis diagnostics for a bank query."""

import argparse
import sys
from pathlib import Path

import torch


ROBOTMDAR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROBOTMDAR_ROOT))

from robotmdar.guidance import LatentBank, local_pca  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bank = LatentBank.load(args.bank, args.metadata)
    query_text = None if bank.text_embeddings is None else bank.text_embeddings[args.query_index]
    result = bank.query(
        bank.latents[args.query_index],
        mode="semantic_then_geometry" if query_text is not None else "geometry",
        top_k=args.top_k,
        query_text_embedding=query_text,
        semantic_top_m=max(args.top_k, 1024),
    )
    pca = local_pca(result.latents[0], args.rank)
    orthogonality_error = torch.linalg.matrix_norm(pca.basis.T @ pca.basis - torch.eye(args.rank))
    payload = {
        "query_index": args.query_index,
        "neighbor_indices": result.indices[0],
        "latent_distances": result.latent_distances[0],
        "basis": pca.basis,
        "eigenvalues": pca.eigenvalues,
        "explained_variance_ratio": pca.explained_variance_ratio,
        "orthogonality_error": orthogonality_error,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print({"orthogonality_error": float(orthogonality_error), "eigenvalues": pca.eigenvalues.tolist()})


if __name__ == "__main__":
    main()
