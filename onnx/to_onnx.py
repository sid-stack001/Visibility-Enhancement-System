import torch
import torch.onnx
from models.networks import ResnetGenerator

# 1. Create generator exactly as in training
# Change n_blocks if your model used 6 (for 128x128) instead of 9
model = ResnetGenerator(
    input_nc=3,
    output_nc=3,
    ngf=64,
    norm_layer=torch.nn.InstanceNorm2d,
    use_dropout=False,
    n_blocks=9
)

# 2. Load weights
checkpoint = torch.load("path\model.pth", map_location="cpu")  # change path & filename
model.load_state_dict(checkpoint)
model.eval()

# 3. Dummy input (match your image size)
dummy_input = torch.randn(1, 3, 256, 256)

# 4. Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "cyclegan_gen_A.onnx",  # name of your ONNX file
    export_params=True,
    opset_version=11,         # good for TensorRT
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)

print("✅ Conversion complete: cyclegan_gen_A.onnx created")
