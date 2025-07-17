import asyncio
import json
import os
import sys
import re
import json
from typing import Optional, List, Dict, Tuple, Any
from contextlib import AsyncExitStack
from pydantic import BaseModel
import subprocess
import platform

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv


from .types import DeviceInfo, OptimizationIdea, KernelResult, BenchmarkResult
from .prompts import idea_generation_prompt, code_generation_prompt
from .RAG.chroma_db import RAGDatabase

load_dotenv()

class KernelAgent:
    def __init__(
        self,
        openai_api_key: str,
        mcp_server_path: str,
        branching_factor: int = 12,
        num_rounds: int = 5,
        selection_percentage: float = 0.2,
        implementations_per_idea: int = 3,
        idea_db=None,
        code_db=None,
        model_name: str = "o1"
    ):
        """
        Initialize the Kernel Optimization Agent
        
        Args:
            openai_api_key: OpenAI API key
            mcp_server_path: Path to the MCP benchmarking server
            branching_factor: Number of ideas to generate per round
            num_rounds: Number of optimization rounds
            selection_percentage: Percentage of top kernels to carry forward
            implementations_per_idea: Number of code implementations per idea
            idea_db: RAG database for idea generation (None if not used)
            code_db: RAG database for code generation (None if not used)
            model_name: OpenAI model to use
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.mcp_server_path = mcp_server_path
        self.branching_factor = branching_factor
        self.num_rounds = num_rounds
        self.selection_percentage = selection_percentage
        self.implementations_per_idea = implementations_per_idea
        self.idea_db = idea_db
        self.code_db = code_db
        self.model_name = model_name
        
        if code_db is not None:
            self.code_db = code_db
        
        # Cache for storing results across rounds
        self.kernel_cache: List[KernelResult] = []
        self.winning_kernels: List[KernelResult] = []
        self.building_blocks: Dict[str, str] = {}
        
        self.device_info = self._detect_device_info()
        
        self.idea_generation_prompt = idea_generation_prompt
        self.code_generation_prompt = code_generation_prompt
        
        self.mcp_session = None

    def _detect_device_info(self) -> DeviceInfo:
        """Auto-detect GPU device information"""
        try:
            # Try to get GPU info using nvidia-ml-py or nvidia-smi
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = lines[0].split(', ')
                name = gpu_info[0]
                memory_mb = float(gpu_info[1])
                compute_cap = gpu_info[2]
                
                # Hardcoded values based on common GPUs (should be expanded)
                device_specs = {
                    "A100": {"bandwidth": 1555, "fp32_tflops": 19.5, "fp16_tflops": 78, "sm_count": 108},
                    "H100": {"bandwidth": 3352, "fp32_tflops": 51, "fp16_tflops": 204, "sm_count": 132},
                    "RTX 4090": {"bandwidth": 1008, "fp32_tflops": 35, "fp16_tflops": 70, "sm_count": 128},
                    "L40S": {"bandwidth": 864, "fp32_tflops": 38, "fp16_tflops": 76, "sm_count": 142}
                }
                
                # Match device name to specs
                matched_specs = None
                for device_key, specs in device_specs.items():
                    if device_key.lower() in name.lower():
                        matched_specs = specs
                        break
                
                if matched_specs:
                    return DeviceInfo(
                        name=name,
                        compute_capability=compute_cap,
                        memory_bandwidth_gb_s=matched_specs["bandwidth"],
                        peak_fp32_tflops=matched_specs["fp32_tflops"],
                        peak_fp16_tflops=matched_specs["fp16_tflops"],
                        memory_size_gb=memory_mb / 1024,
                        sm_count=matched_specs["sm_count"],
                        max_threads_per_block=1024,
                        shared_memory_per_block_kb=48
                    )
        except Exception as e:
            print(f"Warning: Could not detect GPU info: {e}")
        
        return DeviceInfo(
            name="Unknown GPU",
            compute_capability="8.0",
            memory_bandwidth_gb_s=900,
            peak_fp32_tflops=20,
            peak_fp16_tflops=40,
            memory_size_gb=16,
            sm_count=80,
            max_threads_per_block=1024,
            shared_memory_per_block_kb=48
        )

    async def _initialize_mcp_session(self):
        if self.mcp_session is None:
            server_params = StdioServerParameters(
                command=self.mcp_server_path,
                args=[]
            )
            self.exit_stack = AsyncExitStack()
            self.mcp_session = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )

    async def _cleanup_mcp_session(self):
        if hasattr(self, 'exit_stack'):
            await self.exit_stack.aclose()

    async def _query_rag_database(self, db, query: str, top_k: int = 5) -> List[str]:
        if db is None:
            return []
        
        try:
            results = await db.semantic_search(query, top_k)
            
            documents = []
            for result in results:
                documents.append(result["text"])
            
            return documents
            
        except Exception as e:
            print(f"Error querying RAG database: {e}")
            return []

    async def _build_idea_generation_context(self, pytorch_code: str, round_num: int) -> str:
        """Build context for idea generation including device info and previous results"""
        context = f"""
Device Information:
- GPU: {self.device_info.name}
- Compute Capability: {self.device_info.compute_capability}
- Memory Bandwidth: {self.device_info.memory_bandwidth_gb_s} GB/s
- Peak FP32 Performance: {self.device_info.peak_fp32_tflops} TFLOPS
- Peak FP16 Performance: {self.device_info.peak_fp16_tflops} TFLOPS
- Memory Size: {self.device_info.memory_size_gb} GB
- SM Count: {self.device_info.sm_count}
- Max Threads per Block: {self.device_info.max_threads_per_block}
- Shared Memory per Block: {self.device_info.shared_memory_per_block_kb} KB

PyTorch Code to Optimize:
{pytorch_code}

Round: {round_num}/{self.num_rounds}
"""
        
        if round_num > 1 and self.winning_kernels:
            context += "\nPrevious Round Results:\n"
            for kernel in self.winning_kernels[-10:]:  # Last 10 winning kernels
                context += f"- Idea: {kernel.idea.idea_text}\n"
                context += f"  Speedup: {kernel.benchmark_result.speedup:.2f}x\n"
                context += f"  Category: {kernel.idea.category or 'Unknown'}\n\n"
        
        # Add RAG context if available
        if self.idea_db is not None:
            try:
                rag_docs = await self._query_rag_database(self.idea_db, pytorch_code, top_k=5)
                if rag_docs:
                    context += "\nRelevant Optimization Literature:\n"
                    context += "\n".join(rag_docs)
            except Exception as e:
                print(f"Error querying idea RAG database: {e}")
        
        return context

    async def _build_code_generation_context(self, idea: OptimizationIdea, pytorch_code: str) -> str:
        """Build context for code generation including relevant building blocks"""
        context = f"""
Optimization Idea: {idea.idea_text}

PyTorch Reference Code:
{pytorch_code}

Device Information:
- GPU: {self.device_info.name}
- Compute Capability: {self.device_info.compute_capability}
- Memory Bandwidth: {self.device_info.memory_bandwidth_gb_s} GB/s
"""
        
        if self.building_blocks:
            context += "\nRelevant Building Blocks:\n"
            for name, code in list(self.building_blocks.items())[:3]:  # Top 3 relevant blocks
                context += f"\n{name}:\n{code}\n"
        
        if self.code_db is not None:
            try:
                rag_docs = await self._query_rag_database(self.code_db, idea.idea_text, top_k=3)
                if rag_docs:
                    context += "\nRelevant Implementation Examples:\n"
                    context += "\n".join(rag_docs)
            except Exception as e:
                print(f"Error querying code RAG database: {e}")
        
        return context

    async def _generate_ideas(self, pytorch_code: str, round_num: int) -> List[OptimizationIdea]:
        """Generate optimization ideas using OpenAI"""
        context = await self._build_idea_generation_context(pytorch_code, round_num)
        
        ideas = []
        for i in range(self.branching_factor):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.idea_generation_prompt},
                        {"role": "user", "content": f"{context}\n\nGenerate optimization idea #{i+1}:"}
                    ],
                    temperature=0.8 # TODO: experiment with temperature
                )
                
                idea_text = response.choices[0].message.content.strip()
                ideas.append(OptimizationIdea(
                    idea_text=idea_text,
                    round_number=round_num,
                    category=self._categorize_idea(idea_text)
                ))
                
            except Exception as e:
                print(f"Error generating idea {i+1}: {e}")
                continue
        
        return ideas

    def _categorize_idea(self, idea_text: str) -> str:
        """Categorize optimization idea based on keywords"""
        categories = {
            "memory": ["memory", "shared", "global", "cache", "bandwidth"],
            "compute": ["tensor", "wmma", "mma", "fp16", "compute"],
            "parallelism": ["warp", "block", "thread", "occupancy", "parallel"],
            "algorithm": ["gemm", "convolution", "attention", "algorithm", "implicit"],
            "fusion": ["fuse", "fusion", "kernel", "combine"],
            "async": ["async", "pipeline", "overlap", "latency", "hiding"]
        }
        
        idea_lower = idea_text.lower()
        for category, keywords in categories.items():
            if any(keyword in idea_lower for keyword in keywords):
                return category
        
        return "other"

    async def _generate_code(self, idea: OptimizationIdea, pytorch_code: str) -> List[str]:
        """Generate code implementations for an optimization idea"""
        context = await self._build_code_generation_context(idea, pytorch_code)
        
        implementations = []
        for i in range(self.implementations_per_idea):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.code_generation_prompt},
                        {"role": "user", "content": f"{context}\n\nGenerate implementation variant #{i+1}:"}
                    ],
                    temperature=0.3 # TODO: experiment with temperature
                )
                
                code = response.choices[0].message.content.strip()
                implementations.append(code)
                
            except Exception as e:
                print(f"Error generating code implementation {i+1} for idea: {e}")
                continue
        
        return implementations

    async def _benchmark_kernel(self, generated_code: str, reference_code: str, inputs: list = None, init_inputs: list = None) -> BenchmarkResult:
        """Benchmark generated kernel against PyTorch reference"""
        try:
            await self._initialize_mcp_session()
            
            print("Benchmarking reference PyTorch code...")
            reference_result = await self.mcp_session.call_tool(
                "ncu_profiler",
                {
                    "code": reference_code,
                    "inputs": inputs,
                    "init_inputs": init_inputs
                }
            )
            
            reference_data = json.loads(reference_result.content[0].text)
            
            if reference_data.get("status") != "success":
                return BenchmarkResult(
                    speedup=0.0,
                    ncu_results={},
                    compilation_success=False,
                    correctness_passed=False,
                    error_message=f"Reference code failed: {reference_data.get('error', 'Unknown error')}"
                )
            
            print("Benchmarking generated kernel code...")
            generated_result = await self.mcp_session.call_tool(
                "ncu_profiler",
                {
                    "code": generated_code,
                    "inputs": inputs,
                    "init_inputs": init_inputs
                }
            )
            
            generated_data = json.loads(generated_result.content[0].text)
            
            compilation_success = generated_data.get("status") == "success"
            if not compilation_success:
                return BenchmarkResult(
                    speedup=0.0,
                    ncu_results=generated_data,
                    compilation_success=False,
                    correctness_passed=False,
                    error_message=f"Generated code failed: {generated_data.get('error', 'Unknown error')}"
                )
            
            ref_time = None
            gen_time = None
            
            if reference_data.get("csv_metrics") and generated_data.get("csv_metrics"):
                ref_metrics = reference_data["csv_metrics"][0] if reference_data["csv_metrics"] else {}
                gen_metrics = generated_data["csv_metrics"][0] if generated_data["csv_metrics"] else {}
                
                ref_time = float(ref_metrics.get("gpu__time_duration.avg", 0))
                gen_time = float(gen_metrics.get("gpu__time_duration.avg", 0))
            
            speedup = 0.0
            if ref_time and gen_time and gen_time > 0:
                speedup = ref_time / gen_time
            
            return BenchmarkResult(
                speedup=speedup,
                ncu_results={
                    "reference_metrics": reference_data,
                    "generated_metrics": generated_data,
                    "reference_time_ns": ref_time,
                    "generated_time_ns": gen_time
                },
                compilation_success=True,
                correctness_passed=True,
                error_message=None
            )
            
        except Exception as e:
            return BenchmarkResult(
                speedup=0.0,
                ncu_results={},
                compilation_success=False,
                correctness_passed=False,
                error_message=str(e)
            )

    def _update_cache(self, round_results: List[KernelResult]):
        """Update kernel cache with round results"""
        self.kernel_cache.extend(round_results)
        
        valid_results = [r for r in round_results if r.benchmark_result.compilation_success 
                        and r.benchmark_result.correctness_passed and r.benchmark_result.speedup > 0]
        
        if valid_results:
            valid_results.sort(key=lambda x: x.benchmark_result.speedup, reverse=True)
            num_winners = max(1, int(len(valid_results) * self.selection_percentage))
            
            round_winners = valid_results[:num_winners]
            self.winning_kernels.extend(round_winners)
            
            for winner in round_winners[:3]:
                block_name = f"round_{winner.round_number}_{winner.idea.category}_{winner.benchmark_result.speedup:.2f}x"
                self.building_blocks[block_name] = winner.generated_code
            
            if len(self.building_blocks) > 20:
                sorted_blocks = sorted(self.building_blocks.items(), 
                                     key=lambda x: float(x[0].split('_')[-1][:-1]), reverse=True)
                self.building_blocks = dict(sorted_blocks[:15])

    async def _run_optimization_round(self, pytorch_code: str, round_num: int, inputs: list = None, init_inputs: list = None) -> List[KernelResult]:
        """Run a single optimization round"""
        print(f"\n=== Round {round_num}/{self.num_rounds} ===")
        
        print(f"Generating {self.branching_factor} optimization ideas...")
        ideas = await self._generate_ideas(pytorch_code, round_num)
        print(f"Generated {len(ideas)} ideas")
        
        round_results = []
        for i, idea in enumerate(ideas):
            print(f"Processing idea {i+1}/{len(ideas)}: {idea.idea_text[:50]}...")
            
            implementations = await self._generate_code(idea, pytorch_code)
            
            for j, code in enumerate(implementations):
                print(f"  Benchmarking implementation {j+1}/{len(implementations)}...")
                
                benchmark_result = await self._benchmark_kernel(code, pytorch_code, inputs, init_inputs)
                
                kernel_result = KernelResult(
                    idea=idea,
                    generated_code=code,
                    benchmark_result=benchmark_result,
                    round_number=round_num
                )
                
                round_results.append(kernel_result)
                
                if benchmark_result.compilation_success and benchmark_result.correctness_passed:
                    print(f"    Speedup: {benchmark_result.speedup:.2f}x")
                else:
                    print(f"    Failed: {benchmark_result.error_message}")
        
        return round_results

    async def generate_kernel(self, pytorch_code: str, inputs: list = None, init_inputs: list = None) -> Tuple[str, Dict]:
        """
        Main entry point: Generate optimized kernel for given PyTorch code
        
        Args:
            pytorch_code: String containing PyTorch nn.Module forward function
            inputs: List of input tensors for the forward method
            init_inputs: List of initialization inputs for the model constructor
            
        Returns:
            Tuple of (best_kernel_code, benchmark_results_dict)
        """
        try:
            print(f"Starting kernel optimization for device: {self.device_info.name}")
            print(f"Configuration: {self.num_rounds} rounds, {self.branching_factor} ideas per round")
            
            for round_num in range(1, self.num_rounds + 1):
                round_results = await self._run_optimization_round(pytorch_code, round_num, inputs, init_inputs)
                self._update_cache(round_results)
                
                valid_results = [r for r in round_results if r.benchmark_result.compilation_success 
                               and r.benchmark_result.correctness_passed and r.benchmark_result.speedup > 0]
                
                if valid_results:
                    best_round = max(valid_results, key=lambda x: x.benchmark_result.speedup)
                    print(f"Round {round_num} best: {best_round.benchmark_result.speedup:.2f}x speedup")
                    print(f"Idea: {best_round.idea.idea_text[:100]}...")
                else:
                    print(f"Round {round_num}: No successful implementations")
            
            if self.winning_kernels:
                best_kernel = max(self.winning_kernels, key=lambda x: x.benchmark_result.speedup)
                
                return best_kernel.generated_code, {
                    "speedup": best_kernel.benchmark_result.speedup,
                    "ncu_results": best_kernel.benchmark_result.ncu_results,
                    "optimization_idea": best_kernel.idea.idea_text,
                    "round_discovered": best_kernel.round_number,
                    "category": best_kernel.idea.category,
                    "total_kernels_evaluated": len(self.kernel_cache),
                    "successful_kernels": len([k for k in self.kernel_cache 
                                             if k.benchmark_result.compilation_success 
                                             and k.benchmark_result.correctness_passed])
                }
            else:
                return "", {
                    "speedup": 0.0,
                    "error": "No successful kernel implementations found",
                    "total_kernels_evaluated": len(self.kernel_cache)
                }
                
        except Exception as e:
            return "", {
                "speedup": 0.0,
                "error": f"Optimization failed: {str(e)}",
                "total_kernels_evaluated": len(self.kernel_cache)
            }
        
        finally:
            await self._cleanup_mcp_session()

async def main():
    agent = KernelAgent(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        mcp_server_path="./ncu.py",
        branching_factor=8,
        num_rounds=3,
        selection_percentage=0.25,
        implementations_per_idea=2
    )
    
    pytorch_code = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
    
    def forward(self, x):
        return self.conv(x)
"""
    
    inputs = [torch.randn(1, 3, 224, 224)]
    init_inputs = []
    
    best_kernel, results = await agent.generate_kernel(pytorch_code, inputs, init_inputs)
    print(f"Best speedup: {results['speedup']:.2f}x")
    print(f"Optimization: {results.get('optimization_idea', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(main())