import sys
import os
import torch
import torch.nn as nn

from ncu import ncu_profiler

def test_ncu_profiler():
    """
    Test the ncu_profiler function with a simple ReLU model.
    """
    
    model_code = '''
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return torch.relu(x)
'''

    batch_size = 16
    dim = 16384
    
    def get_inputs():
        x = torch.randn(batch_size, dim)
        return [x]

    def get_init_inputs():
        return []

    inputs = get_inputs()
    init_inputs = get_init_inputs()
    
    print("Testing NCU Profiler...")
    print(f"Input tensor shape: {inputs[0].shape}")
    print(f"Input tensor dtype: {inputs[0].dtype}")
    print("="*60)
    
    try:
        result = ncu_profiler(
            code=model_code,
            inputs=inputs,
            init_inputs=init_inputs
        )
        
        print(f"Profiling Status: {result['status']}")
        print("="*60)
        
        if result['detailed_metrics']:
            print("Detailed Metrics:")
            for key, value in result['detailed_metrics'].items():
                print(f"  {key}: {value}")
            print("="*60)
        
        if result['derived_metrics']:
            print("Derived Metrics:")
            for key, value in result['derived_metrics'].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2e}")
                else:
                    print(f"  {key}: {value}")
            print("="*60)
        
        if result['csv_metrics']:
            print("CSV Metrics (first kernel):")
            for key, value in result['csv_metrics'][0].items():
                print(f"  {key}: {value}")
            print("="*60)
        
        if result['status'] == 'error':
            print(f"Error: {result.get('error', 'Unknown error')}")
            print(f"stderr: {result['raw_stderr']}")
        
        if '--verbose' in sys.argv:
            print("Raw NCU Output:")
            print(result['detailed_stdout'])
            print("="*60)
            print("Raw CSV Output:")
            print(result['raw_stdout'])
        
    except Exception as e:
        print(f"Error calling ncu_profiler: {e}")
        import traceback
        traceback.print_exc()

def check_requirements():
    """
    Check if required tools and packages are available.
    """
    print("Checking requirements...")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. NCU profiling requires CUDA.")
        return False
    
    print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    
    import subprocess
    try:
        result = subprocess.run(['ncu', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"NCU is available: {result.stdout.strip()}")
            return True
        else:
            print("ERROR: NCU is not available or not working properly")
            return False
    except FileNotFoundError:
        print("ERROR: NCU (Nsight Compute) is not installed or not in PATH")
        return False

if __name__ == "__main__":
    print("NCU Profiler Test Script")
    print("="*60)
    
    if not check_requirements():
        print("Requirements check failed. Exiting.")
        sys.exit(1)
    
    print("="*60)
    test_ncu_profiler()
    print("="*60)
    print("Test completed!") 