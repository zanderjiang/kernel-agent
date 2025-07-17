#!/usr/bin/env python3

import asyncio
import os
import sys
# Add the kernel-agent directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'kernel-agent'))

from agent import KernelAgent

async def test_kernel_agent():
    """Test the kernel agent with a simple example"""
    
    # Example PyTorch code to optimize
    pytorch_code = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
    
    def forward(self, x):
        return self.conv(x)
"""
    
    # Initialize agent
    agent = KernelAgent(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        mcp_server_path="python kernel-agent/ncu.py",
        branching_factor=2,  # Small for testing
        num_rounds=1,       # Just 1 round for testing
        selection_percentage=0.5,
        implementations_per_idea=1,
        idea_db=None,  # Leave as None for now
        code_db=None,  # Will use default RAG
        model_name="gpt-4"  # Use a more accessible model for testing
    )
    
    print("Starting kernel optimization test...")
    
    # Generate kernel
    best_kernel, results = await agent.generate_kernel(pytorch_code)
    
    print(f"\nResults:")
    print(f"Best speedup: {results.get('speedup', 0):.2f}x")
    print(f"Optimization idea: {results.get('optimization_idea', 'N/A')}")
    print(f"Total kernels evaluated: {results.get('total_kernels_evaluated', 0)}")
    print(f"Successful kernels: {results.get('successful_kernels', 0)}")
    
    if best_kernel:
        print(f"\nBest kernel code preview:")
        print(best_kernel[:500] + "..." if len(best_kernel) > 500 else best_kernel)
    else:
        print("\nNo successful kernel generated")
        print(f"Error: {results.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(test_kernel_agent()) 