import torch


def test_torch_imports_and_reports_version() -> None:
    assert isinstance(torch.__version__, str)
    assert len(torch.__version__) > 0


def test_matmul_is_correct() -> None:
    # Hand-checkable 2x2 matrix multiply.
    #   [[1, 2],   [[5, 6],     [[1*5+2*7, 1*6+2*8],     [[19, 22],
    #    [3, 4]] @  [7, 8]]  =   [3*5+4*7, 3*6+4*8]]  =   [43, 50]]
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    assert torch.allclose(a @ b, expected)


def test_allclose_tolerance_behaves() -> None:
    x = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(x, x + 1e-8)
    assert not torch.allclose(x, x + 1.0)
