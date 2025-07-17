# Kernel Agent - Automated CUDA Kernel Optimization

This is agent utilizes only test time optimizations (RAG, Search) on existing LLM APIs for Kernel Generation, it follows the KernelBench convention, where the reference kernel is defined as pytorch modules with a forward method, along with a inputs and init inputs for testing.