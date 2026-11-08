import pytest
import torch
import torch.nn.functional as F

from vision_transformer.embedding import VisionTransformerEmbedding
from vision_transformer.encoder import EncoderBlock


def test_block_matches_pre_norm_reference():
    torch.manual_seed(0)
    block = EncoderBlock(d_model=8, num_heads=2, d_ff=16)
    x = torch.randn(2, 5, 8)

    # PyTorchの標準演算を使った独立の参照計算。
    norm = F.layer_norm(x, (8,), block.norm1.gamma, block.norm1.beta)
    attention = block.attention
    q, k, v = [
        layer(norm).reshape(2, 5, 2, 4).transpose(1, 2)
        for layer in (attention.W_Q, attention.W_K, attention.W_V)
    ]
    expected_attention = F.scaled_dot_product_attention(q, k, v)
    y = x + attention.W_O(expected_attention.transpose(1, 2).reshape(2, 5, 8))
    norm = F.layer_norm(y, (8,), block.norm2.gamma, block.norm2.beta)
    expected = y + block.ffn.linear2(F.gelu(block.ffn.linear1(norm)))

    output, weights = block(x)

    torch.testing.assert_close(output, expected)
    assert weights.shape == (2, 2, 5, 5)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 2, 5))


def test_zero_branches_preserve_residual_without_final_norm():
    block = EncoderBlock(8, 2, 16)
    with torch.no_grad():
        block.attention.W_O.weight.zero_()
        block.ffn.linear2.weight.zero_()
        block.ffn.linear2.bias.zero_()
    x = torch.randn(2, 5, 8) * 3 + 7

    output, _ = block(x)

    torch.testing.assert_close(output, x)


def test_cls_loss_backpropagates_through_embedding_and_block():
    torch.manual_seed(1)
    embedding = VisionTransformerEmbedding(4, 2, 3, 8)
    block = EncoderBlock(8, 2, 16)
    images = torch.randn(2, 3, 4, 4, requires_grad=True)

    output, _ = block(embedding(images))
    output[:, 0].square().mean().backward()

    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert images.grad.abs().sum() > 0
    for module in (embedding, block):
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_dropout_is_disabled_in_eval_mode():
    block = EncoderBlock(8, 2, 16, dropout=1.0)
    x = torch.randn(2, 5, 8)
    output, _ = block(x)
    torch.testing.assert_close(output, x)

    block.eval()
    output, _ = block(x)
    assert not torch.allclose(output, x)
    torch.testing.assert_close(output, block(x)[0])


@pytest.mark.parametrize("arguments", [(8, 0, 16), (0, 2, 16), (8, 3, 16), (8, 2, 0)])
def test_block_rejects_invalid_dimensions(arguments):
    with pytest.raises(ValueError):
        EncoderBlock(*arguments)
