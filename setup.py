#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from setuptools.command.install import install
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
    install_package("torch")
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension:

def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content

def parse_requirements(fname='requirements.txt', with_version=True):
    # ... (保持不变)

    packages = list(gen_packages_items())
    return packages

class PostInstallCommand(install):
    def run(self):
        super().run()
        print("\n" + "=" * 60)
        print("  Welcome to visit our libcom online workbench")
        print("  https://libcom.ustcnewly.com/")
        print("=" * 60 + "\n")


if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    setup(
        name='libcom',
        version='0.1.5.post1',
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
        install_requires=parse_requirements('requirements.txt'),
        cmdclass={'build_ext': BuildExtension, 'install': PostInstallCommand},
    )
