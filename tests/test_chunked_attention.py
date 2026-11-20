import pytest
import torch
from neural_lam.models import edm_networks_2

from clara_inference import chunked_attention


@pytest.fixture
def restore_unet_block():
    forward = edm_networks_2.UNetBlock.forward
    yield
    edm_networks_2.UNetBlock.forward = forward
    if hasattr(edm_networks_2.UNetBlock, "_clara_attention_chunk"):
        del edm_networks_2.UNetBlock._clara_attention_chunk


@pytest.mark.parametrize("chunk", [1, 7, 30, 1000])
def test_chunked_attention_is_unchanged(restore_unet_block, chunk):
    torch.manual_seed(0)
    block = edm_networks_2.UNetBlock(
        in_channels=120,
        out_channels=120,
        channels_per_head=60,
        emb_channels=16,
        attention=True,
        init_zero=dict(init_weight=0.3),
    ).eval()
    x = torch.randn(2, 120, 5, 6)
    emb = torch.randn(2, 16)
    with torch.no_grad():
        expected = block(x.clone(), emb)
        chunked_attention.patch_unet_attention(chunk)
        got = block(x.clone(), emb)
    assert block.num_heads == 2
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_blocks_without_attention_are_unchanged(restore_unet_block):
    torch.manual_seed(0)
    block = edm_networks_2.UNetBlock(
        in_channels=60, out_channels=60, emb_channels=16
    ).eval()
    x, emb = torch.randn(1, 60, 4, 4), torch.randn(1, 16)
    with torch.no_grad():
        expected = block(x.clone(), emb)
        chunked_attention.patch_unet_attention(3)
        got = block(x.clone(), emb)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
