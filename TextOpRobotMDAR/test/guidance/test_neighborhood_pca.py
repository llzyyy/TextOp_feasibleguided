import torch

from robotmdar.guidance import local_pca, retrieve_neighbors


def test_semantic_then_geometry_applies_both_filters_deterministically():
    latents = torch.tensor([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0], [5.1, 0.0]])
    text = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    result = retrieve_neighbors(
        latents,
        torch.tensor([5.0, 0.0]),
        mode="semantic_then_geometry",
        top_k=2,
        bank_text_embeddings=text,
        query_text_embedding=torch.tensor([1.0, 0.0]),
        semantic_top_m=2,
    )
    assert result.indices.tolist() == [[1, 0]]
    repeated = retrieve_neighbors(
        latents,
        torch.tensor([5.0, 0.0]),
        mode="semantic_then_geometry",
        top_k=2,
        bank_text_embeddings=text,
        query_text_embedding=torch.tensor([1.0, 0.0]),
        semantic_top_m=2,
    )
    torch.testing.assert_close(result.indices, repeated.indices)


def test_local_pca_returns_orthonormal_reproducible_basis():
    generator = torch.Generator().manual_seed(7)
    neighbors = torch.randn(32, 8, generator=generator)
    first = local_pca(neighbors, rank=4)
    second = local_pca(neighbors, rank=4)
    torch.testing.assert_close(first.basis.T @ first.basis, torch.eye(4), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(first.basis, second.basis)
    assert torch.all(first.eigenvalues[:-1] >= first.eigenvalues[1:])


def test_local_pca_rejects_rank_not_supported_by_neighbors():
    try:
        local_pca(torch.randn(4, 8), rank=4)
    except ValueError as exc:
        assert "rank" in str(exc)
    else:
        raise AssertionError("expected rank validation")
