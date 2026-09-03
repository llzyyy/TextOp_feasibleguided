import torch

from robotmdar.guidance import GeoGuide, TaskContext, TaskSpec, task_cost


def make_context():
    return TaskContext(
        robot_world_position=torch.zeros(1, 3),
        robot_world_yaw=torch.zeros(1),
        generator_position=torch.zeros(1, 3),
        generator_yaw=torch.zeros(1),
        motion_fps=50.0,
    )


def test_waypoint_cost_uses_reconstructed_future_root_xy():
    future = torch.zeros(1, 3, 57)
    future[..., 7] = 0.5
    future[..., 10] = 0.77
    task = TaskSpec(type="waypoint", frame="world", position=(1.0, 0.0, 0.0))
    cost = task_cost(future, make_context(), task)
    torch.testing.assert_close(cost, torch.tensor(0.0))


def test_raw_task_gradient_step_lowers_cost():
    def decoder(clean_latent, history):
        del history
        future = torch.zeros(clean_latent.shape[0], 3, 57, device=clean_latent.device)
        future[..., 7] = clean_latent[:, 0, 0].unsqueeze(-1)
        future[..., 10] = 0.77
        return future

    config = {
        "enabled": True,
        "task": {"enabled": True, "type": "waypoint", "target": {"frame": "world", "position": [2, 0, 0]}},
        "subspace": {"mode": "none"},
        "solver": {"eta": 0.1},
        "schedule": {"guided_steps": [1]},
    }
    guide = GeoGuide(config, decoder)
    latent = torch.zeros(1, 1, 4)
    history = torch.zeros(1, 0, 57)
    updated = guide.guide(latent, torch.tensor([1]), history, None, make_context())
    assert guide.last_stats is not None
    assert guide.last_stats.gradient_norm > 0
    assert guide.last_stats.task_cost_after < guide.last_stats.task_cost_before
    assert updated[0, 0, 0] > 0


def test_robot_start_target_is_rotated_into_world():
    context = TaskContext(
        robot_world_position=torch.tensor([[10.0, 20.0, 0.0]]),
        robot_world_yaw=torch.tensor([torch.pi / 2]),
        generator_position=torch.tensor([[10.0, 20.0, 0.0]]),
        generator_yaw=torch.tensor([torch.pi / 2]),
    )
    target = context.target_to_world(torch.tensor([2.0, 0.0, 0.0]), "robot_start")
    torch.testing.assert_close(target, torch.tensor([[10.0, 22.0, 0.0]]), atol=1e-6, rtol=0)


def test_generator_velocity_is_rotated_without_translation():
    context = TaskContext(
        robot_world_position=torch.tensor([[10.0, 20.0, 0.0]]),
        robot_world_yaw=torch.tensor([torch.pi / 2]),
        generator_position=torch.tensor([[3.0, 4.0, 0.0]]),
        generator_yaw=torch.tensor([0.25]),
    )
    velocity = context.vector_to_world(torch.tensor([2.0, 0.0, 0.0]), "generator")
    torch.testing.assert_close(velocity, torch.tensor([[0.0, 2.0, 0.0]]), atol=1e-6, rtol=0)
