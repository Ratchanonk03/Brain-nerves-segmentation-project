from sagemaker_pytorch_serving_container import torchserve
import handler_service 

HANDLER_SERVICE = handler_service.__file__
    
def _start_torchserve():
    torchserve.start_torchserve(handler_service=HANDLER_SERVICE)
    
if __name__ == "__main__":
    _start_torchserve()