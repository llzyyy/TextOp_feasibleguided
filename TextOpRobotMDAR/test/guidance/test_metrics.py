import torch

from robotmdar.evaluation.geoguide.motion_metrics import foot_sliding


def test_foot_sliding_aggregates_time_and_feet():
    positions = torch.zeros(1, 3, 2, 3)
    positions[:, 1, 0, 0] = 1.0
    positions[:, 2, 0, 0] = 2.0
    contacts = torch.ones(1, 3, 2)
    result = foot_sliding(positions, contacts, fps=2.0)
    # Two transitions at speed 2 for one foot and 0 for the other.
    torch.testing.assert_close(result, torch.tensor([1.0]))
