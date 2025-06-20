import torch
from NeuralNN import NeuralNN  # your defined model

# Step 1: Recreate model architecture
model = NeuralNN()

# Step 2: Load weights into the model
model.load_state_dict(torch.load("myModel.pth", map_location="cpu"))
model.eval()

# Step 3: Define dummy input (must match input size, here 4 features)
dummy_input = torch.randn(1, 4)

# Step 4: Export to ONNX
torch.onnx.export(
    model,                          # model
    dummy_input,                    # dummy input tensor
    "myModel.onnx",                 # output file name
    input_names=['input'],          # name of the input node
    output_names=['output'],        # name of the output node
    dynamic_axes={                  # optional: make batch size dynamic
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    export_params=True,
    opset_version=11,               # compatible with STM32Cube.AI
    do_constant_folding=True        # optimize model
)

print("✅ Exported ONNX model: myModel.onnx")
