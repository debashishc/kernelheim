import torch

@torch.compile
def compile_square(matrix: torch.Tensor) -> torch.Tensor:
    return torch.square(matrix)

# if not using decorator @torch.compile
# compiled_square = torch.compile(compile_square,) 


if __name__ == "__main__":
    # x = torch.Tensor([[1,2,3], [3,2,1]]).to(torch.device("cuda"))
    # or using torch.randn
    x = torch.randn((25, 25), device="cuda")
    print(compile_square(x))