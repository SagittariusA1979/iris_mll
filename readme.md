<PIP>
    pip install gymnasium
    pip install "gymnasium[box2d]"
</>

<REQuirment>
    pip freeze > requirements.txt
    pip install -r requirements.txt
</>

<GIT>
    git rm --cached path/to/file.onnx
    git rm -r --cached __pycache__
</>

<To tested onnx model>
    pip install onnx
    pip install onnxruntime onnxsim netron
    ---
    onnxruntime:    Run ONNX models directly for testing.
    onnxsim:        Simplifies the model before deploying to embedded systems.
    netron:         GUI tool to visualize ONNX architecture (netron myModel.onnx). 
</>


