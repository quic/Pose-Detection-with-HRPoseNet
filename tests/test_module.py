import pytest
import os

import onnxruntime as ort

from unittest.mock import Mock, patch
from src.hrnet_pose.model_loader import ModelLoader

class TestModelLoader():

    mock_execution_config = {
                            "NPU": "QNNExecutionProvider",
                            "CPU": "CPUExecutionProvider"
                           }
    mock_models_config = {
                        "MODELS":{
                                    "HRNET_POSE": "models/hrnet_pose.onnx",
                                    "HRNET_POSE_W44": "models/hrnet_pose_w44.onnx"
                                }
                        }
    
    def test_get_executioner(self):
        processor = "cpu"
        iLoad = ModelLoader(Mock(),processor=processor)
        # model_inference.processor = "CPU"
        iLoad._get_config = Mock(return_value=TestModelLoader.mock_execution_config)
        result = iLoad._get_executioner()
        assert result == "CPUExecutionProvider"

    def test_get_executioner_invalid_processor(self):
        processor = "something_made_in_2045"
        iLoad = ModelLoader(Mock(),processor=processor)
        iLoad._get_config = Mock(return_value=TestModelLoader.mock_execution_config)
        with pytest.raises(ValueError):
            iLoad._get_executioner()

    def test_get_model_path(self):
        model = "hrnet_pose"
        iLoad = ModelLoader(model=model, processor="placeholder")
        iLoad._get_config = Mock(return_value=TestModelLoader.mock_models_config)
        result = iLoad._get_model_path(model_name=model)
        assert result == os.path.abspath("models/hrnet_pose.onnx")

    def test_get_model_invalid_path(self):
        model = "the_model_that_actually_achieves_agi"
        iLoad = ModelLoader(model=model, processor="placeholder")
        iLoad._get_config = Mock(return_value=TestModelLoader.mock_models_config)
        with pytest.raises(TypeError):
            iLoad._get_model_path(model_name=model)

    @patch("onnxruntime.InferenceSession")
    def test_load_model(self, mock_inference_session):
        model = "hrnet_pose"
        processor = "cpu"
        iLoad = ModelLoader(model=model, processor=processor)
        iLoad._get_model_path = Mock(return_value=TestModelLoader.mock_models_config["MODELS"][model.upper()])
        iLoad._get_executioner = Mock(return_value=TestModelLoader.mock_execution_config[processor.upper()])
        model_path = TestModelLoader.mock_models_config["MODELS"][model.upper()]
        session = iLoad.load_model()

        iLoad._get_model_path.assert_called_once_with(model_name=model)
        iLoad._get_executioner.assert_called_once()

        mock_inference_session.assert_called_once_with(
            path_or_bytes=model_path,
            providers=[TestModelLoader.mock_execution_config[processor.upper()]]
        )

        assert session == mock_inference_session.return_value
        



                                           
