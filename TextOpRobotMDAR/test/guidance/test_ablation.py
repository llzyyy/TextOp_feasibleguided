from pathlib import Path

from robotmdar.evaluation.geoguide import ABLATION_PRESETS, build_ablation_plan, validate_preset


def test_all_a0_a7_presets_are_valid_and_plannable():
    for name, preset in ABLATION_PRESETS.items():
        validate_preset(name, preset)
    plan = build_ablation_plan(sorted(ABLATION_PRESETS), [0, 1], Path("outputs"))
    assert len(plan) == 16
    assert {item["ablation_id"] for item in plan} == set(ABLATION_PRESETS)


def test_h_only_is_global_identity_geometry():
    preset = ABLATION_PRESETS["A3"]
    assert preset["subspace"] == "identity"
    assert preset["decoder_h"]
    assert not preset["support_h"]
