from sagemaker_pytorch_serving_container import default_pytorch_inference_handler
from inference_handler import LinknetHandler, model_fn, input_fn, predict_fn, output_fn
import os

class HandlerService(LinknetHandler):
    def __init__(self):
        super(HandlerService, self).__init__()
        self._model = None
    
    def initialize(self, context):
        """
        Initialize the handler with model loading
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self._model = model_fn(model_dir)
    
    def handle(self, data, context):
        """
        TorchServe handler entry point
        """
        # Ensure model is loaded
        if self._model is None:
            self.initialize(context)
        
        # Process input - TorchServe expects application/json by default
        content_type = 'application/json'
        if hasattr(context, 'get_request_header'):
            try:
                content_type = context.get_request_header('Content-Type') or 'application/json'
            except:
                content_type = 'application/json'
        
        # Use the functions from inference_handler
        input_data = input_fn(data[0]['body'], content_type)  # TorchServe passes data as list of dicts
        prediction = predict_fn(input_data, self._model)
        
        # Return output
        accept_type = 'application/json'
        if hasattr(context, 'get_request_header'):
            try:
                accept_type = context.get_request_header('Accept') or 'application/json'
            except:
                accept_type = 'application/json'
                
        result, content_type = output_fn(prediction, accept_type)
        return [result]  # TorchServe expects a list          