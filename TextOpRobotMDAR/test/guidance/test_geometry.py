import torch

from robotmdar.guidance import clip_to_trust_region, decoder_metric, solve_geometry_step


def test_decoder_jvp_metric_matches_linear_map():
    latent = torch.tensor([[[1.0, 2.0]]], requires_grad=True)
    basis = torch.eye(2)

    def decoder(value):
        return value * torch.tensor([2.0, 3.0])

    result = decoder_metric(decoder, latent, basis)
    torch.testing.assert_close(result.matrix, torch.diag(torch.tensor([4.0, 9.0])))


def test_geometry_solver_uses_natural_step():
    gradient = torch.tensor([2.0, 4.0])
    basis = torch.eye(2)
    geometry = torch.diag(torch.tensor([2.0, 4.0]))
    result = solve_geometry_step(gradient, basis, geometry, eta=0.5)
    torch.testing.assert_close(result.coordinates, torch.tensor([-0.5, -0.5]))
    torch.testing.assert_close(result.latent_step, result.coordinates)


def test_trust_region_enforces_geometry_norm_without_changing_direction():
    coordinates = torch.tensor([3.0, 4.0])
    geometry = torch.eye(2)
    result = clip_to_trust_region(coordinates, geometry, rho=2.0)
    assert result.clipped
    torch.testing.assert_close(result.geometry_norm, torch.tensor(2.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(result.coordinates / result.coordinates[0], coordinates / coordinates[0])
