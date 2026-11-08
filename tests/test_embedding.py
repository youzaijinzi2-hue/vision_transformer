import pytest
import torch

from vision_transformer.embedding import VisionTransformerEmbedding


def test_embedding_output_shape():
    embedding = VisionTransformerEmbedding(32, 8, 3, 64)

    assert embedding(torch.randn(2, 3, 32, 32)).shape == (2, 17, 64)


def test_embedding_prepends_shared_cls_and_adds_positions():
    embedding = VisionTransformerEmbedding(4, 2, 1, 1)
    with torch.no_grad():
        embedding.patch_embedding.projection.weight.fill_(1.0)
        embedding.patch_embedding.projection.bias.zero_()
        embedding.cls_token.fill_(10.0)
        embedding.position_embedding.copy_(
            torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
        )
    images = torch.stack([torch.zeros(1, 4, 4), torch.ones(1, 4, 4)])

    tokens = embedding(images)

    expected = torch.tensor([
        [[10.], [1.], [2.], [3.], [4.]],
        [[10.], [5.], [6.], [7.], [8.]],
    ])
    torch.testing.assert_close(tokens, expected)


def test_cls_and_positions_are_registered_and_receive_gradients():
    embedding = VisionTransformerEmbedding(4, 2, 1, 3)
    parameters = dict(embedding.named_parameters())
    assert parameters["cls_token"] is embedding.cls_token
    assert parameters["position_embedding"] is embedding.position_embedding

    embedding(torch.ones(2, 1, 4, 4)).sum().backward()

    torch.testing.assert_close(
        embedding.cls_token.grad, torch.full_like(embedding.cls_token, 2.0)
    )
    torch.testing.assert_close(
        embedding.position_embedding.grad,
        torch.full_like(embedding.position_embedding, 2.0),
    )
    torch.testing.assert_close(
        embedding.patch_embedding.projection.weight.grad, torch.full((3, 4), 8.0)
    )


def test_dropout_is_disabled_in_eval_mode():
    embedding = VisionTransformerEmbedding(4, 2, 1, 3, dropout=1.0)
    images = torch.ones(1, 1, 4, 4)
    with torch.no_grad():
        embedding.cls_token.fill_(1.0)

    assert torch.count_nonzero(embedding(images)) == 0
    embedding.eval()
    assert torch.count_nonzero(embedding(images)) > 0


@pytest.mark.parametrize("image_size,patch_size", [(0, 2), (-4, 2), (4, 0), (4, -2), (5, 2)])
def test_embedding_rejects_invalid_sizes(image_size, patch_size):
    with pytest.raises(ValueError):
        VisionTransformerEmbedding(image_size, patch_size, 3, 8)


@pytest.mark.parametrize("shape", [(1, 3, 8, 4), (1, 3, 4, 8), (1, 3, 2, 8)])
def test_embedding_rejects_different_image_geometry(shape):
    embedding = VisionTransformerEmbedding(4, 2, 3, 8)

    with pytest.raises(ValueError, match="must match image_size"):
        embedding(torch.zeros(shape))


def test_embedding_rejects_non_image_dimensions():
    embedding = VisionTransformerEmbedding(4, 2, 3, 8)

    with pytest.raises(ValueError, match="images must have shape"):
        embedding(torch.zeros(3, 4, 4))
