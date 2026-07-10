import pytest

torch = pytest.importorskip("torch")

from src.models.cmmca import CMMCA


def make_model():
    return CMMCA(
        dim=32,
        text_dim=24,
        vision_dim=16,
        num_query_tokens=16,
        num_layers=2,
        n_heads=4,
        out_context_dim=32,
    )


def test_shape_and_hidden_states():
    model = make_model()
    text = torch.randn(2, 7, 24)
    controls = [torch.randn(2, 9, 16), torch.randn(2, 9, 16)]
    mask = torch.tensor([[1, 0], [1, 1]])
    states = model(text, controls, mask, return_hidden_states=True)
    assert len(states) == 2
    assert states[-1].shape == (2, 16, 32)


def test_masked_control_cannot_change_output():
    torch.manual_seed(7)
    model = make_model().eval()
    text = torch.randn(1, 7, 24)
    first = torch.randn(1, 9, 16)
    second = torch.randn(1, 9, 16)
    mask = torch.tensor([[1, 0]])
    output_a = model(text, [first, second], mask)
    output_b = model(text, [first, second + 1000], mask)
    torch.testing.assert_close(output_a, output_b)
