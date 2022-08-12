
FROM ubuntu:18.04


# Install wget
RUN apt-get update -y && apt-get install -y wget

# Install Python and pip
RUN apt-get update -y && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y && apt-get install -y python3.8
RUN apt-get install -y python3-pip
RUN python3.8 -m pip install --upgrade pip

# Install Python dependencies
COPY requires.txt /
RUN python3.8 -m pip install -r requires.txt

# Install Julia
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.6/julia-1.6.7-linux-x86_64.tar.gz
RUN tar -xvzf julia-1.6.7-linux-x86_64.tar.gz
RUN ln -s /julia-1.6.7/bin/julia /bin/julia

# Install Julia dependencies
RUN julia --project=/ -e "using Pkg; Pkg.add(\"CUDA\"); Pkg.add(\"Flux\"); Pkg.add(\"HDF5\"); Pkg.add(\"JSON\"); Pkg.add(\"LinearAlgebra\"); Pkg.add(\"ScikitLearnBase\"); Pkg.add(\"StatsBase\")"
RUN julia --project=/ -e "using Pkg; Pkg.add(url=\"https://github.com/dpmerrell/MatFac.jl\")"
RUN julia --project=/ -e "using Pkg; Pkg.add(url=\"https://github.com/dpmerrell/PathwayMultiomics.jl\")"

