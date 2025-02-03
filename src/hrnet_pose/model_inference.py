import torch
import os

import cv2 as cv
import numpy as np

from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from onnxruntime import InferenceSession, NodeArg
from PIL import Image
from typing import List,Tuple

class ModelInference():
    """
    A class for handling model inference for pose estimation (HRnet_pose) using OpenCV and PyTorch.
    """
    def __init__(self, session: InferenceSession) -> None:
        """
        Initialize the ModelInference class.

        Parameters:
        ----------
        session : InferenceSession
            The ONNX inference session instance to be used for model inference.
        """
        self.session = session
        self._expected_inputs = session.get_inputs()[0]
        self.expected_shape = self._expected_inputs.shape
        self.device = torch.device("cuda" if torch.cuda.is_available else "cpu")

        self.transformer = v2.Compose([
            v2.Resize(size=(self.expected_shape[2], self.expected_shape[3]), interpolation=InterpolationMode.BICUBIC),
            v2.PILToTensor(),
            v2.ConvertImageDtype(torch.float32),                                                # Converts to float32 AND Scales
        ])
    
    def _image_transformer(self, frame: Image) -> Tuple[np.array, np.array]:
        """
        Transforms the input image to match the model's expected input format.

        Parameters:
        ----------
        frame : Image
            A PIL Image to be transformed.

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray]
            The transformed image in NumPy format (H, W, C) and the model-compatible tensor (B, C, H, W).
        """
        
        transformed_frame = self.transformer(frame)
        transformed_frame_np = transformed_frame.cpu().numpy() if self.device == "gpu" else transformed_frame.numpy()
        transformed_frame_inference = np.expand_dims(transformed_frame_np,axis=0)
        final_frame = np.transpose(transformed_frame_np, (1,2,0))

        return (final_frame, transformed_frame_inference)
    
    def inference(self, camera: int=1) -> None:                                                              # Will generalize this even more to work with all models
        """
        Conducts inference by capturing frames from a camera, processing them, and displaying results.

        Parameters:
        ----------
        camera : int, optional
            The camera index for OpenCV to capture from (default is 1).
        """
        cap = cv.VideoCapture(camera)
        if not cap.isOpened():                                                                  # Can also use this to automatically select a camera, probably a better option to alleviate any frustrations                 
            raise ValueError(f"Error while trying to open camera - {self.available_cameras}")   
        output_height, _ = self._frame_shape(cap=cap)
        scaler = self.expected_shape[2]/output_height
        
        while True: 
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Can't receive frame. Exiting....")
            frame, inference_frame = self._frame_processor(frame)
            keypoints = self._inference_onnx(inference_frame=inference_frame, scaler=scaler)                         
            frame = frame.copy()                                                                # To prevent errors due to memory fragmentation during processing
            for (y,x) in keypoints:
                cv.circle(frame, (x,y), radius=3, color=(0,0,255), thickness=-1)
            frame = cv.resize(frame, (640,480), interpolation=cv.INTER_CUBIC)
            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
           
    def _inference_onnx(self, inference_frame: np.array, scaler: int=None, solo_scaler: bool=False) -> List[Tuple[int,int]]:
        """
        Performs inference on the input frame using the ONNX model and scales keypoints.

        Parameters:
        ----------
        inference_frame : np.ndarray
            The input frame formatted for inference.
        scaler : int, optional
            The scaling factor to adjust keypoints to the original frame size.
        solo_scaler : bool, optional
            If True, returns the output shape for scaling only.

        Returns:
        -------
        List[Tuple[int, int]]
            A list of keypoints as (y, x) tuples.
        
        Additional:
        ----------
        Each keypoint represents a different bodypart identifier
        https://www.researchgate.net/figure/Key-points-for-human-poses-according-to-the-COCO-output-format-R-L-right-left_fig3_353746430
        """
        keypoint_coordinates: List[Tuple[int,int]] = []
        outputs = self.session.run(None, {self._expected_inputs.name:inference_frame})
        output_numpy = np.squeeze(np.squeeze(np.array(outputs), axis=0))

        if solo_scaler:
            out_height, out_width = output_numpy.shape[1],output_numpy.shape[2]
            keypoint_coordinates.append((out_height,out_width))
            return keypoint_coordinates
    
        for keypoint in range(output_numpy.shape[0]): 
            max_val =  np.argmax(output_numpy[keypoint])                                    
            img_height, img_width =  np.unravel_index(max_val, output_numpy[keypoint].shape)
            coordinates_scaled = (int(img_height*scaler), int(img_width*scaler))
            keypoint_coordinates.append(coordinates_scaled)
        
        return keypoint_coordinates



    def _frame_shape(self, cap: cv.VideoCapture) -> Tuple[int,int]:
        """
        Determines the output shape of a frame after inference scaling.

        Parameters:
        ----------
        cap : cv.VideoCapture
            The OpenCV video capture object.

        Returns:
        -------
        Tuple[int, int]
            The height and width of the frame after scaling.
        """
        _, frame = cap.read()
        frame, frame_transform = self._frame_processor(frame)
        output_shape = self._inference_onnx(inference_frame=frame_transform,solo_scaler=True)

        return output_shape[0]

    
    def _frame_processor(self, frame: np.array) -> Tuple[np.array, np.array]:
        """
        Processes the input frame to match model input expectations.

        Parameters:
        ----------
        frame : np.ndarray
            The input frame in NumPy array format (H, W, C).

        Returns:
        -------
        Tuple[np.ndarray, np.ndarray]
            The processed frame in (H, W, C) format and the model-ready format (B, C, H, W).
        """
        frame = Image.fromarray(frame)
        frame, frame_transform = self._image_transformer(frame=frame)

        return (frame,frame_transform)
        

    @property
    def available_cameras(self, max_cameras: int=5) -> List[int]:
        """
        Lists available camera indices.

        Parameters:
        ----------
        max_cameras : int, optional
            Maximum number of cameras to check (default is 10).

        Returns:
        -------
        str
            A string listing available camera indices.
        """        
        #Log suppression is currently not working, need to figure out why
        os.environ["OPENCV_LOG_LEVEL"] = "FATAL"                                                # Suppress logging to ignore any error that's not fatal

        available_cameras: List[int] = []
        for cam in range(max_cameras):
            cap = cv.VideoCapture(cam)
            if cap.isOpened():
                available_cameras.append(str(cam))
                cap.release()

        os.environ["OPENCV_LOG_LEVEL"] = "INFO"                                                 # Restore logging back to normal

        return f"Available Cameras: {' | '.join(available_cameras)}"

    @property
    def expected_inputs(self) -> NodeArg:
        """
        Returns the expected inputs for the model.

        Returns:
        -------
        Any
            The expected inputs configuration for the ONNX model.
        """
    
        return self._expected_inputs

if __name__=="__main__":
    from model_loader import ModelLoader

    model_name = "hrnet_pose"
    system = "windows"
    processor = "cpu"
    iLoad = ModelLoader(model=model_name, processor=processor)
    session = iLoad.load_model()

    iInfer = ModelInference(session)
    print(iInfer.expected_inputs)
    # iInfer.inference(camera=1)




    

