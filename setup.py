from setuptools import setup, find_packages

setup(
    name="qnn_sample_apps",
    version="0.1.0",
    author="Derrick Johnson",
    description="Sample applications demonstrating QNN models",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "src.models": ["*.onnx"],  # Include all .onnx files in models directory
    },
    extras_require={
        "dev": ["pytest"]
    }
)