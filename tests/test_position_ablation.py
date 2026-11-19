import pytest
import torch

from vision_transformer.ablate_positions import conditions, initial_statistics
from vision_transformer.embedding import VisionTransformerEmbedding
from vision_transformer.model import VisionTransformer


def make_model(**kwargs):
    return VisionTransformer(8, 2, 1, 2, 16, 4, 32, 1, **kwargs)


def test_defaults_preserve_initialization_and_forward():
    torch.manual_seed(12)
    default = make_model()
    torch.manual_seed(12)
    explicit = make_model(position_init_std=0.02, position_scale=1.0)
    for name, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[name], rtol=0, atol=0)
    images = torch.randn(2, 1, 8, 8)
    torch.testing.assert_close(default(images), explicit(images), rtol=0, atol=0)


@pytest.mark.parametrize('std', [0.0, 0.02, 0.25, 1.0])
def test_normal_initialization_matches_same_normal_draws(std):
    torch.manual_seed(9)
    unit = VisionTransformerEmbedding(32, 2, 1, 64, position_init_std=1.0)
    torch.manual_seed(9)
    scaled = VisionTransformerEmbedding(32, 2, 1, 64, position_init_std=std)
    torch.testing.assert_close(scaled.position_embedding, std * unit.position_embedding)
    assert unit.position_embedding.mean().abs().item() < 0.03
    assert unit.position_embedding.std().item() == pytest.approx(1.0, abs=0.03)


def test_scaling_affects_tokens_and_position_gradient():
    embedding = VisionTransformerEmbedding(4, 2, 1, 3, position_scale=2.5)
    images = torch.ones(2, 1, 4, 4)
    patches = embedding.patch_embedding(images)
    tokens = torch.cat([embedding.cls_token.expand(2, -1, -1), patches], dim=1)
    output = embedding(images)
    torch.testing.assert_close(output, tokens + 2.5 * embedding.position_embedding)
    output.sum().backward()
    torch.testing.assert_close(embedding.position_embedding.grad,
                               torch.full_like(embedding.position_embedding, 5.0))


def test_common_parameters_and_bc_effective_positions_match_but_gradients_differ():
    models = {}
    for key, config in conditions(16).items():
        torch.manual_seed(11)
        models[key] = make_model(**config)
    baseline = models['A'].state_dict()
    for model in models.values():
        for name, value in model.state_dict().items():
            if name != 'embedding.position_embedding':
                torch.testing.assert_close(value, baseline[name], rtol=0, atol=0)
    images = torch.randn(2, 1, 8, 8)
    b, c = models['B'], models['C']
    torch.testing.assert_close(b.embedding(images), c.embedding(images), rtol=0, atol=0)
    b(images).sum().backward()
    c(images).sum().backward()
    torch.testing.assert_close(c.embedding.position_embedding.grad,
                               4 * b.embedding.position_embedding.grad)


def test_statistics_exclude_cls_and_do_not_change_rng():
    model = make_model(position_scale=3.0)
    with torch.no_grad():
        model.embedding.position_embedding[:, 0].fill_(1e6)
        model.embedding.cls_token.fill_(1e6)
    images = torch.randn(2, 1, 8, 8)
    state = torch.get_rng_state().clone()
    stats = initial_statistics(model, images)
    assert torch.equal(state, torch.get_rng_state())
    patches = model.embedding.patch_embedding(images)
    positions = model.embedding.position_embedding[:, 1:] * 3
    assert stats['patch_std'] == pytest.approx(patches.std(correction=0).item())
    assert stats['scaled_position_std'] == pytest.approx(positions.std(correction=0).item())
    assert stats['position_patch_std_ratio'] == pytest.approx(
        positions.std(correction=0).item() / patches.std(correction=0).item())
    assert stats['patch_mean_l2'] == pytest.approx(patches.norm(dim=-1).mean().item())
    assert stats['scaled_position_mean_l2'] == pytest.approx(positions.norm(dim=-1).mean().item())


def test_condition_definitions():
    pairs = [(0.02, 1), (1, 1), (0.125, 8), (0.125, 1), (0.02, 8), (0.125, 64**0.25)]
    for config, (std, scale) in zip(conditions(64).values(), pairs):
        assert config == {'position_init_std': std, 'position_scale': scale}
