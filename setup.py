#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os
import sys
import subprocess
from os.path import join


def install_package(package_name):
    """Install a package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
except ImportError:
    print("PyTorch is not installed. Installing PyTorch first...")
    install_package("torch")  # You can specify a version if needed
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension



def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

def parse_requirements(fname='requirements.txt', with_version=True):
    # https://github.com/open-mmlab/mmdetection
    """
    Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import sys
    from os.path import exists
    import re
    require_fpath = fname

    def parse_line(line):
        """
        Parse information from a line in a requirements text file
        """
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages

def get_ext_modules(cur_dir):
    # if encounter compilation issues, please refer to https://github.com/HuiZeng/Image-Adaptive-3DLUT?tab=readme-ov-file#build.
    if torch.cuda.is_available():
        if torch.__version__ >= '1.11.0':
            print('torch version >= 1.11.0')
            trilinear_cuba_abs_path = join(cur_dir, 'libcom/image_harmonization/source/trilinear_cpp_torch1.11/src/trilinear_cuda.cpp')
            trilinear_kernel_abs_path = join(cur_dir, 'libcom/image_harmonization/source/trilinear_cpp_torch1.11/src/trilinear_kernel.cu')
            return CUDAExtension('trilinear', [trilinear_cuba_abs_path, trilinear_kernel_abs_path])
        else:
            trilinear_cuba_abs_path = join(cur_dir, 'libcom/image_harmonization/source/trilinear_cpp/src/trilinear_cuda.cpp')
            trilinear_kernel_abs_path = join(cur_dir, 'libcom/image_harmonization/source/trilinear_cpp/src/trilinear_kernel.cu')
            return CUDAExtension('trilinear', [trilinear_cuba_abs_path, trilinear_kernel_abs_path])
    else:
        trilinear_abs_path = join(cur_dir, 'libcom/image_harmonization/source/trilinear_cpp/src/trilinear.cpp')
        src_abs_path = join(cur_dir, 'libcom/image_harmonization/source/trilinear_cpp/src')
        return CppExtension('trilinear', [trilinear_abs_path], include_dirs=[src_abs_path]) 

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    setup(
        name='libcom',
        version='0.1.0',
        description='Image Composition Toolbox',
        long_description=readme(),
        long_description_content_type='text/markdown',
        url='https://github.com/bcmi/libcom',
        author='BCMI Lab',
        author_email='bernard-zhang@hotmail.com',
        keywords='computer vision, image composition',
        packages=find_packages(exclude=('tests', 'demo')),
        package_data={'': ['*.yaml','*.txt']},
        classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.8',
        ],
        license='Apache License 2.0',
        setup_requires=parse_requirements('requirements/build.txt'),
        tests_require=parse_requirements('requirements/tests.txt'),
        install_requires=parse_requirements('requirements/runtime.txt'),
        ext_modules=[get_ext_modules(cur_dir)],
        cmdclass={'build_ext': BuildExtension},
        extras_require={
            'all': parse_requirements('requirements.txt'),
            'build': parse_requirements('requirements/build.txt'),
            'test': parse_requirements('requirements.txt'),
        },
    )
