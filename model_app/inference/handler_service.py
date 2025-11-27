from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_pytorch_serving_container.default_pytorch_inference_handler import DefaultPytorchInferenceHandler

class HandlerService(DefaultHandlerService):
    def __init__(self):
        super().__init__(DefaultPytorchInferenceHandler())