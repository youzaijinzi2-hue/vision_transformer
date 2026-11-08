import pytest
import torch
import torch.nn.functional as F

from vision_transformer.encoder import TransformerEncoder
from vision_transformer.model import VisionTransformer


def make_model():
    return VisionTransformer(
        image_size=8, patch_size=4, in_channels=3, num_classes=3,
        d_model=8, num_heads=2, d_ff=16, num_layers=2,
    )


def test_encoder_stacks_independent_layers_and_applies_final_norm():
    encoder = TransformerEncoder(8, 2, 16, num_layers=2)
    assert (
        encoder.layers[0].attention.W_Q.weight
        is not encoder.layers[1].attention.W_Q.weight
    )
    x = torch.randn(2, 5, 8)
    output, maps = encoder(x)

    expected, _ = encoder.layers[0](x)
    expected, _ = encoder.layers[1](expected)
    expected = F.layer_norm(expected, (8,), encoder.norm.gamma, encoder.norm.beta)
    torch.testing.assert_close(output, expected)
    assert len(maps) == 2
    assert all(weights.shape == (2, 2, 5, 5) for weights in maps)


def test_model_classifies_cls_token_and_returns_raw_logits():
    model = make_model()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.tensor([-2., 0., 3.]))

    logits = model(torch.randn(2, 3, 8, 8))

    torch.testing.assert_close(logits, torch.tensor([[-2., 0., 3.], [-2., 0., 3.]]))

    # Headが受け取るのは最終Norm後のCLSだけ。
    model = make_model()
    images = torch.randn(2, 3, 8, 8)
    encoded, _ = model.encoder(model.embedding(images))
    torch.testing.assert_close(model(images), model.head(encoded[:, 0]))


def test_classification_loss_backpropagates_and_optimizer_updates_parameters():
    torch.manual_seed(3)
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    images = torch.randn(2, 3, 8, 8, requires_grad=True)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    loss = F.cross_entropy(model(images), torch.tensor([0, 2]))
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(images.grad).all()
    assert images.grad.abs().sum() > 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name

    optimizer.step()
    for name in (
        "embedding.cls_token", "embedding.position_embedding",
        "encoder.layers.0.attention.W_Q.weight",
        "encoder.layers.1.ffn.linear1.weight", "encoder.norm.gamma", "head.weight",
    ):
        assert not torch.equal(before[name], dict(model.named_parameters())[name]), name


@pytest.mark.parametrize("num_layers", [0, -1])
def test_encoder_rejects_non_positive_depth(num_layers):
    with pytest.raises(ValueError, match="num_layers must be positive"):
        TransformerEncoder(8, 2, 16, num_layers)


def test_model_rejects_non_positive_class_count():
    with pytest.raises(ValueError, match="num_classes must be positive"):
        VisionTransformer(8, 4, 3, 0, 8, 2, 16, 2)
