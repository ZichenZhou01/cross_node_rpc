# setup.py
import os
import subprocess
import sys
import logging
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CppExtension,
    CUDAExtension,
    BuildExtension,
    CUDA_HOME,
    include_paths,
)
import torch
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_cuda_arch_flags():
    # Override via env: export TORCH_EXTENSION_ARCH_LIST="6.1;7.0"
    arch_list = os.getenv("TORCH_EXTENSION_ARCH_LIST")
    if arch_list:
        flags = [f"-gencode=arch=compute_{a.replace('.', '')},code=sm_{a.replace('.', '')}" for a in arch_list.split(';')]
        logger.info(f"Using architecture(s) from TORCH_EXTENSION_ARCH_LIST={arch_list!r}")
        return flags

    if not CUDA_HOME or not torch.cuda.is_available():
        default = "70"  # Volta
        logger.warning(f"CUDA not found or unavailable—defaulting to sm_{default}")
        return [f"-arch=sm_{default}"]

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    logger.info(f"Detected GPU compute capability {major}.{minor} → sm_{arch}")
    return [f"-arch=sm_{arch}"]


def make_extensions():
    debug = bool(int(os.getenv("DEBUG", "0")))
    use_cuda = CUDA_HOME is not None and torch.cuda.is_available()
    ext_cls = CUDAExtension if use_cuda else CppExtension

    common_args = {
        "extra_compile_args": {
            "cxx": ["-O0" if debug else "-O3", "-fdiagnostics-color=always"] + (["-g"] if debug else []),
            "nvcc": (["-O0" if debug else "-O3", "-v"] + get_cuda_arch_flags() + (["-g"] if debug else []))
            if use_cuda
            else [],
        },
        "extra_link_args": ["-g", "-lrdmacm", "-libverbs"] if debug else ["-lrdmacm", "-libverbs"],
        "include_dirs": include_paths(cuda=use_cuda),
    }

    # find all .cpp/.cu under csrc/
    here = os.path.dirname(__file__)
    src_dir = os.path.join(here, "csrc")
    sources = []
    for ext in ("cpp", "cu"):
        sources += list(Path(src_dir).rglob(f"*.{ext}"))

    return [
        ext_cls(
            "rpc_rdma",
            sources=[str(s) for s in sources],
            **common_args,
        )
    ]


class BuildExtWithStubs(BuildExtension):
    def run(self):
        super().run()
        self._generate_stubs()

    def _generate_stubs(self):
        logger.info("Generating .pyi stubs via pybind11-stubgen…")
        try:
            torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
            ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
            os.environ["LD_LIBRARY_PATH"] = f"{torch_lib_path}:{ld_library_path}"

            subprocess.run(
                [sys.executable, "-m", "pybind11_stubgen", "rpc_rdma_util", "--output-dir", "."],
                check=True,
            )
            stub = "rpc_rdma_util.pyi"
            if os.path.exists(stub):
                logger.info(f"Formatting {stub} with black…")
                subprocess.run(["black", stub], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Stub generation failed: {e}")


setup(
    name="rpc_rdma_util",
    version="0.0.2",
    description="PyTorch C++/CUDA RDMA-enabled extension",
    packages=find_packages(),
    ext_modules=make_extensions(),
    cmdclass={"build_ext": BuildExtWithStubs},
    install_requires=["torch"],
    zip_safe=False,
)
