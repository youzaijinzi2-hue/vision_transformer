import pytest
import torch

from vision_transformer.patch_embedding import PatchEmbedding, patchify


def test_patchify_orders_patches_from_left_to_right_then_top_to_bottom():
    images = torch.arange(16).reshape(1, 1, 4, 4)

    patches = patchify(images, patch_size=2)

    expected = torch.tensor([
        [[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]
    ])
    torch.testing.assert_close(patches, expected)


@pytest.mark.parametrize("patch_size", [1, 2, 4])
def test_patchify_preserves_pixels_in_batched_rgb_rectangles(patch_size):
    images = torch.arange(2 * 3 * 8 * 12).reshape(2, 3, 8, 12)

    patches = patchify(images, patch_size)

    assert patches.shape == (
        2, (8 // patch_size) * (12 // patch_size), 3 * patch_size**2
    )
    # スライスを使い、reshape/permuteとは独立にパッチの内容を確認する。
    index = 0
    for row in range(0, 8, patch_size):
        for col in range(0, 12, patch_size):
            expected = images[
                :, :, row:row + patch_size, col:col + patch_size
            ].flatten(1)
            torch.testing.assert_close(patches[:, index], expected)
            index += 1


def test_patchify_propagates_gradients():
    images = torch.randn(2, 3, 8, 12, requires_grad=True)

    patchify(images, patch_size=4).square().sum().backward()

    torch.testing.assert_close(images.grad, 2 * images.detach())


@pytest.mark.parametrize("shape", [(4, 4), (3, 4, 4), (1, 1, 3, 4, 4)])
def test_patchify_rejects_non_image_dimensions(shape):
    with pytest.raises(ValueError, match="images must have shape"):
        patchify(torch.zeros(shape), patch_size=2)


@pytest.mark.parametrize("patch_size", [0, -1])
def test_patchify_rejects_non_positive_patch_size(patch_size):
    with pytest.raises(ValueError, match="patch_size must be positive"):
        patchify(torch.zeros(1, 3, 4, 4), patch_size)


@pytest.mark.parametrize("shape", [(1, 3, 5, 4), (1, 3, 4, 5)])
def test_patchify_rejects_dimensions_not_divisible_by_patch_size(shape):
    with pytest.raises(ValueError, match="H and W must be divisible"):
        patchify(torch.zeros(shape), patch_size=2)


def test_patch_embedding_output_shape():
    embedding = PatchEmbedding(in_channels=3, patch_size=8, d_model=64)

    tokens = embedding(torch.randn(2, 3, 32, 32))

    assert tokens.shape == (2, 16, 64)


def test_patch_embedding_applies_shared_projection_to_each_patch():
    embedding = PatchEmbedding(in_channels=1, patch_size=2, d_model=2)
    with torch.no_grad():
        embedding.projection.weight.copy_(torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
        ]))
        embedding.projection.bias.copy_(torch.tensor([1.0, -1.0]))
    images = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)

    tokens = embedding(images)

    expected = torch.tensor([[[1., 9.], [3., 13.], [9., 25.], [11., 29.]]])
    torch.testing.assert_close(tokens, expected)


def test_patch_embedding_backpropagates_to_images_and_parameters():
    embedding = PatchEmbedding(in_channels=1, patch_size=2, d_model=2)
    with torch.no_grad():
        embedding.projection.weight.fill_(1.0)
    images = torch.ones(1, 1, 4, 4, requires_grad=True)

    embedding(images).sum().backward()

    torch.testing.assert_close(images.grad, torch.full_like(images, 2.0))
    torch.testing.assert_close(
        embedding.projection.weight.grad, torch.full((2, 4), 4.0)
    )
    torch.testing.assert_close(embedding.projection.bias.grad, torch.full((2,), 4.0))


@pytest.mark.parametrize("arguments", [
    (0, 2, 8), (3, 0, 8), (3, 2, 0),
    (-1, 2, 8), (3, -1, 8), (3, 2, -1),
])
def test_patch_embedding_rejects_non_positive_dimensions(arguments):
    with pytest.raises(ValueError, match="must be positive"):
        PatchEmbedding(*arguments)


def test_patch_embedding_rejects_channel_mismatch():
    embedding = PatchEmbedding(in_channels=3, patch_size=2, d_model=8)

    with pytest.raises(ValueError, match="channels must match"):
        embedding(torch.zeros(1, 1, 4, 4))
