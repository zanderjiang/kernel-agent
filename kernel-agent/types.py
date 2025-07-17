from pydantic import BaseModel
from typing import Dict, Any, Optional

class BenchmarkResult(BaseModel):
    speedup: float
    ncu_results: Dict[str, Any]
    compilation_success: bool = True
    correctness_passed: bool = True
    error_message: Optional[str] = None

class OptimizationIdea(BaseModel):
    idea_text: str
    round_number: int
    category: Optional[str] = None

class KernelResult(BaseModel):
    idea: OptimizationIdea
    generated_code: str
    benchmark_result: BenchmarkResult
    round_number: int

class DeviceInfo(BaseModel):
    name: str
    compute_capability: str
    memory_bandwidth_gb_s: float
    peak_fp32_tflops: float
    peak_fp16_tflops: float
    memory_size_gb: float
    sm_count: int
    max_threads_per_block: int
    shared_memory_per_block_kb: float