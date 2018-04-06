#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import subprocess
import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext


def locate_cuda():
    # first check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = os.path.join(home, 'bin', 'nvcc')

        if not os.path.exists(nvcc):
            raise EnvironmentError('The CUDA path could not be located')
    else:
        # otherwise, search for NVCC
        nvcc = subprocess.check_output('which nvcc'.split()).decode()

        if not nvcc:
            raise EnvironmentError('The CUDA path could not be located')

        home = os.path.abspath(os.path.dirname(nvcc) + '/..')

    cudaconfig = {'home': home, 'nvcc': nvcc}

    return cudaconfig


cuda = locate_cuda()


def customize_compiler_for_nvcc(self):
    """inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on."""

    # tell the compiler it can processes .cu
    self.src_extensions.append('.cu')

    # save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):

        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', str(cuda['nvcc']))
            # use only a subset of the extra_postargs, which are 1-1 translated
            # from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # inject our redefined _compile method into the class
    self._compile = _compile


class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


opencv_libs_str = subprocess.check_output('pkg-config --libs opencv'.split()).decode()
opencv_incs_str = subprocess.check_output('pkg-config --cflags opencv'.split()).decode()

opencv_libs = [str(lib) for lib in opencv_libs_str.strip().split()]
opencv_incs = [str(inc) for inc in opencv_incs_str.strip().split()]

eppm_src = ['EPPM/bao_pmflow_census_kernel.cu', 'EPPM/bao_pmflow_refine_kernel.cu',
            'EPPM/bao_flow_patchmatch_multiscale_cuda.cpp',
            'EPPM/bao_flow_patchmatch_multiscale_kernel.cu', 'EPPM/bao_pmflow_kernel.cu',
            'EPPM/basic/bao_basic_cuda.cpp']

extensions = [
    Extension('optflow',
              sources=['optflow.pyx'] + eppm_src,
              include_dirs=[numpy.get_include(), cuda['home'] + '/include', 'EPPM', 'EPPM/basic'] + opencv_incs,
              language='c++',
              extra_link_args=opencv_libs + ['-lcudart', '-L' + cuda['home'] + '/lib', '-L' + cuda['home'] + '/lib64',
                                             '-L' + cuda['home'] + '/lib/x86_64-linux-gnu', '-g'],
              extra_compile_args={'gcc': ['-g'],
                                  'nvcc': ['-arch=sm_30', '--ptxas-options=-v', '-c', '--compiler-options', "'-fPIC'"]},
              )
]

setup(
    name='optflow',
    version='1.0',
    ext_modules=cythonize(extensions),
    cmdclass={'build_ext': custom_build_ext},
)
