import onnxruntime as ort
import numpy as np

# Run with ONNX Runtime
session = ort.InferenceSession("myModel.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

sample = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
output = session.run([output_name], {input_name: sample})
print("ONNX output:", output)

