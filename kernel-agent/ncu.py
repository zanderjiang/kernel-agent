from mcp.server.fastmcp import FastMCP
import subprocess
import os
import tempfile
import json
import re
import pickle

mcp = FastMCP("NCU Profiler Server")

def parse_ncu_output(stdout: str) -> dict:
    metrics = {}
    
    patterns = {
        'achieved_occupancy': r'Achieved Occupancy\s+%\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'theoretical_occupancy': r'Theoretical Occupancy\s+%\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'memory_throughput': r'Memory Throughput\s+%\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'dram_throughput': r'DRAM Throughput\s+%\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'compute_sm_throughput': r'Compute \(SM\) Throughput\s+%\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'duration_us': r'Duration\s+us\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'sm_frequency_ghz': r'SM Frequency\s+Ghz\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'dram_frequency_ghz': r'DRAM Frequency\s+Ghz\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'elapsed_cycles': r'Elapsed Cycles\s+cycle\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'sm_active_cycles': r'SM Active Cycles\s+cycle\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'achieved_active_warps_per_sm': r'Achieved Active Warps Per SM\s+warp\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'theoretical_active_warps_per_sm': r'Theoretical Active Warps per SM\s+warp\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'l1_tex_cache_throughput': r'L1/TEX Cache Throughput\s+%\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'l2_cache_throughput': r'L2 Cache Throughput\s+%\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'grid_size': r'Grid Size\s+(\d+\.?\d*)',
        'block_size': r'Block Size\s+(\d+\.?\d*)',
        'registers_per_thread': r'Registers Per Thread\s+register/thread\s+(\d+\.?\d*)',
        'threads': r'Threads\s+thread\s+[\d.]+\s+[\d.]+\s+([\d.]+)',
        'waves_per_sm': r'Waves Per SM\s+[\d.]+\s+[\d.]+\s+([\d.]+)'
    }
    
    for metric, pattern in patterns.items():
        match = re.search(pattern, stdout, re.IGNORECASE)
        if match:
            if len(match.groups()) == 2:
                value, unit = match.groups()
                metrics[metric] = {'value': float(value), 'unit': unit}
            else:
                metrics[metric] = float(match.group(1))
    
    kernel_match = re.search(r'Kernel:\s+(.+)', stdout)
    if kernel_match:
        metrics['kernel_name'] = kernel_match.group(1).strip()
    
    return metrics

def calculate_derived_metrics(csv_data: list) -> dict:
    derived = {}
    
    if not csv_data:
        return derived
    
    data = csv_data[0] if csv_data else {}
    
    try:
        fadd_ops = float(data.get('sm__sass_thread_inst_executed_op_fadd_pred_on.sum', 0))
        fmul_ops = float(data.get('sm__sass_thread_inst_executed_op_fmul_pred_on.sum', 0))
        ffma_ops = float(data.get('sm__sass_thread_inst_executed_op_ffma_pred_on.sum', 0))
        dadd_ops = float(data.get('sm__sass_thread_inst_executed_op_dadd_pred_on.sum', 0))
        dmul_ops = float(data.get('sm__sass_thread_inst_executed_op_dmul_pred_on.sum', 0))
        dfma_ops = float(data.get('sm__sass_thread_inst_executed_op_dfma_pred_on.sum', 0))
        
        total_flops = fadd_ops + fmul_ops + (ffma_ops * 2) + dadd_ops + dmul_ops + (dfma_ops * 2)
        
        duration_ns = float(data.get('gpu__time_duration.avg', 0))
        if duration_ns > 0:
            flops_per_second = total_flops / (duration_ns * 1e-9)
            derived['total_flops'] = total_flops
            derived['flops_per_second'] = flops_per_second
            derived['gflops_per_second'] = flops_per_second / 1e9
        
        # memory bandwidth
        total_bytes = float(data.get('dram__bytes.sum', 0))
        if duration_ns > 0 and total_bytes > 0:
            bandwidth_bytes_per_sec = total_bytes / (duration_ns * 1e-9)
            derived['memory_bandwidth_bytes_per_sec'] = bandwidth_bytes_per_sec
            derived['memory_bandwidth_gb_per_sec'] = bandwidth_bytes_per_sec / 1e9
            derived['total_memory_bytes'] = total_bytes
        
        bytes_read = float(data.get('dram__bytes_read.sum', 0))
        bytes_write = float(data.get('dram__bytes_write.sum', 0))
        derived['memory_bytes_read'] = bytes_read
        derived['memory_bytes_write'] = bytes_write
        
        dram_throughput_pct = data.get('gpu__dram_throughput.avg.pct_of_peak', 0)
        compute_throughput_pct = data.get('sm__throughput.avg.pct_of_peak', 0)
        if dram_throughput_pct:
            derived['memory_bandwidth_utilization_pct'] = float(dram_throughput_pct)
        if compute_throughput_pct:
            derived['compute_utilization_pct'] = float(compute_throughput_pct)
            
    except (ValueError, KeyError) as e:
        derived['calculation_error'] = str(e)
    
    return derived

@mcp.tool()
def ncu_profiler(code: str, inputs: list = None, init_inputs: list = None) -> dict:
    """
    Profiles CUDA kernels using NVIDIA Nsight Compute (ncu).
    
    The input code should contain:
    - A nn.Module with a forward method
    
    Args:
        code (str): Python code string containing the custom CUDA kernel
        inputs (list): List of input tensors for the forward method
        init_inputs (list): List of initialization inputs for the model constructor
    Returns:
        dict: Dictionary containing parsed profiling metrics and raw output
    """
    
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "profile_kernel.py")
        inputs_path = os.path.join(tmpdir, "inputs.pkl")
        init_inputs_path = os.path.join(tmpdir, "init_inputs.pkl")
        
        if inputs is not None:
            with open(inputs_path, 'wb') as f:
                pickle.dump(inputs, f)
        
        if init_inputs is not None:
            with open(init_inputs_path, 'wb') as f:
                pickle.dump(init_inputs, f)
        
        script_content = f"""
import torch
import torch.nn as nn
import os
import sys
import pickle

os.environ["TORCH_USE_CUDA_DSA"] = "1"

exec('''
{code}
''')

def main():
    init_inputs = None
    inputs = None
    
    if os.path.exists("{init_inputs_path}"):
        with open("{init_inputs_path}", 'rb') as f:
            init_inputs = pickle.load(f)
    
    if os.path.exists("{inputs_path}"):
        with open("{inputs_path}", 'rb') as f:
            inputs = pickle.load(f)
    
    if init_inputs:
        init_inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs]
        model = Model(*init_inputs)
    else:
        model = Model()
    
    model = model.cuda()
    model.eval()
    
    if inputs:
        inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]
    else:
        inputs = []
    
    with torch.no_grad():
        for _ in range(3):
            _ = model(*inputs)
    
    torch.cuda.synchronize()
    
    with torch.no_grad():
        output = model(*inputs)
    
    torch.cuda.synchronize()
    return output

if __name__ == "__main__":
    main()
"""
        
        with open(script_path, "w") as f:
            f.write(script_content)
        
        ncu_cmd = [
            "ncu",
            "--metrics", 
            "gpu__time_duration.avg,sm__cycles_elapsed.avg,smsp__cycles_elapsed.avg",
            "--csv",
            "--log-file", os.path.join(tmpdir, "ncu.log"),
            "python", script_path
        ]
        
        try:
            result = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=120)
            
            csv_data = []
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                # Filter out lines that start with == (NCU status messages)
                csv_lines = [line for line in lines if not line.startswith('==')]
                if len(csv_lines) > 1:
                    import csv
                    import io
                    csv_content = '\n'.join(csv_lines)
                    reader = csv.DictReader(io.StringIO(csv_content))
                    csv_data = list(reader)
            
            ncu_detailed_cmd = [
                "ncu",
                "--print-summary", "per-kernel",
                "python", script_path
            ]
            
            detailed_result = subprocess.run(ncu_detailed_cmd, capture_output=True, text=True, timeout=120)
            detailed_metrics = parse_ncu_output(detailed_result.stdout)
            
            # flops, memory bandwidth calculations
            derived_metrics = calculate_derived_metrics(csv_data)
            
            return {
                "status": "success" if result.returncode == 0 else "warning",
                "csv_metrics": csv_data,
                "detailed_metrics": detailed_metrics,
                "derived_metrics": derived_metrics,
                "raw_stdout": result.stdout,
                "raw_stderr": result.stderr,
                "detailed_stdout": detailed_result.stdout,
                "detailed_stderr": detailed_result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": "Profiling timed out after 120 seconds",
                "csv_metrics": [],
                "detailed_metrics": {},
                "derived_metrics": {},
                "raw_stdout": "",
                "raw_stderr": ""
            }
        except subprocess.CalledProcessError as e:
            return {
                "status": "error", 
                "error": f"ncu command failed with return code {e.returncode}",
                "csv_metrics": [],
                "detailed_metrics": {},
                "derived_metrics": {},
                "raw_stdout": e.stdout if e.stdout else "",
                "raw_stderr": e.stderr if e.stderr else ""
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Unexpected error: {str(e)}",
                "csv_metrics": [],
                "detailed_metrics": {},
                "derived_metrics": {},
                "raw_stdout": "",
                "raw_stderr": ""
            }

if __name__ == "__main__":
    mcp.run(transport='stdio')