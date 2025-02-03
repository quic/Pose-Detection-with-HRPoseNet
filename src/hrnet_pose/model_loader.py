import onnxruntime as ort
import os
import json

class ModelLoader:
    def __init__(self, model: str, processor: str) -> None:
        """
        Initializes an instance with the specified model and processor.

        This constructor sets up the model name, processor type, and configuration directory path.
        The configuration directory is set to the directory two levels up from the current file location.

        Args:
            model (str): The name of the model to be used.
            processor (str): The type of processor (e.g., "cpu", "npu").

        Attributes:
            model (str): Stores the name of the specified model.
            processor (str): Stores the processor type.
            config_dir (str): The absolute path to the configuration directory, located two levels up 
                            from the current file directory.
        """
        self.processor = processor.upper()
        self.model = model
        self.config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

    def _get_config(self, filename: str) -> json:
        """
        Constructs the absolute path to a configuration file based on its filename.

        Args:
            filename (str): The name of the configuration file.

        Returns:
            str: The absolute path to the configuration file.
        """
        config_path = os.path.abspath(os.path.join(self.config_dir,f"{filename}"))

        with open(config_path, "r") as f:
            config = json.load(f)

        return config 

    def _get_model_path(self, model_name: str) -> ort.InferenceSession:
        """
        Retrieves the absolute path of an ONNX model based on the model name provided, using the path specified within models.json configuration file.

        Args:
            model_name (str): The name of the model to load. This should correspond to a key in the models.json configuration file.

        Returns:
            str: The absolute path to the ONNX model file.

        Raises:
            ValueError: If the specified model name is not found in the models.json configuration file.
        """

        config = self._get_config(filename="models.json")

        model_path = os.path.abspath(config["MODELS"].get(model_name.upper()))

        if not model_path:
            raise ValueError(f"Model path for {model_name} not found in config")

        return model_path

    def _get_dll_path(self, system: str, processor: str, runtime: str="qnn") -> str:
        """
        //DEPRECATED//

        Retrieves the absolute path of a specified DLL file based on the system and processor type.
        The path is determined from the dll.json configuration file.

        Args:
            dll_name (str): The name of the DLL to load. 

        Returns:
            str: The absolute path to the DLL file.

        Raises:
            KeyError: If the specified system or processor type is not found in the dll.json configuration file.
        """
        config_file_name = "dll.json"
        config = self._get_config(filename= config_file_name)

        dll_path = os.path.abspath(config[system.upper()].get(processor))

        if not os.path.exists(dll_path):
            raise ValueError(f"Processor: {processor} or System: {system.upper()} not found within config")

        return dll_path
    
    def _get_executioner(self) -> str:
        """
        Retrieves the ExecutionProvider configuration for a specified processor.

        This function loads a configuration file (executioner.json) containing available processors 
        and their corresponding ExecutionProvider. It checks if the specified processor is available 
        in the configuration file. If the processor is not listed, it raises an error indicating 
        the available options. If the processor is found, it returns the relevant ExecutionProvider 
        for the selected processor.

        Returns:
            str: The ExecutionProvider for the specified processor.

        Raises:
            ValueError: If the specified processor does not have a supported ExecutionProvider.
        """
        config_file_name = "executioner.json"
        config = self._get_config(filename= config_file_name)
        if self.processor not in config.keys():
            raise ValueError(f"Selected processor ({self.processor}) not available. Please select {' | '.join(list(config.keys()))}")
        return config[self.processor]



    def load_model(self) -> ort.InferenceSession:
        """
        Loads an ONNX model with specified system and processor settings.

        Args:
            model_name (str): The name of the model to load.

        Returns:
            ort.InferenceSession: An ONNX Runtime InferenceSession configured with the specified model, 
                                execution provider.
        """

        model_path = self._get_model_path(model_name=self.model)
        executioner = self._get_executioner()
        session = ort.InferenceSession(path_or_bytes=model_path, providers=[executioner])
        return session
    
    @property
    def executioners(self) -> str:
        """
        Retrieves and lists available executioners from a configuration file.

        This property reads from the `executioner.json` configuration file to obtain
        key-value pairs of executioners and their settings, formats them into a list,
        and returns a formatted string listing each executioner and its associated value.

        Returns:
        -------
        str
            A formatted string where each line represents an executioner in the format
            "key: value", with one executioner per line.

        Notes:
        ------
        The configuration file `executioner.json` should be located in the expected
        directory, and the `_get_config` method should correctly handle reading the
        JSON file and returning its contents as a dictionary.
        """
        config_file_name = "executioner.json"
        config = self._get_config(filename= config_file_name)
        available_executioners = [f"{key}: {value}" for key,value in config.items()]
        return "\n".join(available_executioners)


if __name__=="__main__":
    
    model_name = "hrnet_pose"
    system = "windows"
    processor = "cpu"
    iLoad = ModelLoader(model=model_name, processor=processor)
    session = iLoad.load_model()
    print(session.get_inputs()[0])
    print(iLoad.executioners)
