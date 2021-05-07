from setuptools import setup
import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

sources = glob.glob("src/*.cpp")+glob.glob("src/*.cu")
setup(
    name='deform_conv',
    ext_modules=[
        CUDAExtension('dcn_cuda', sources),
    ],
    cmdclass={'build_ext': BuildExtension})