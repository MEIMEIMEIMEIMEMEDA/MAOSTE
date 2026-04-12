"""Env setup.py."""
import os
import re

from setuptools import find_packages
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension


def parse_requirements(requie_file='requirements.txt'):
  assert os.path.exists(requie_file), f"{requie_file} doesn't exists!"
  with open(requie_file, 'r') as f:
    data = [
        line.strip('\n')
        for line in f.readlines()
        if not re.match(r'^\s*(-i|--extra-index-url).*', line)
    ]
  return data


if __name__ == '__main__':
  setup(
      name='maoste',
      version='0.0.0',
      description='MAOSTE: Multimodal Aspect-Object-Sentiment Triples Extraction',
      author='haomei',
      author_email='',
      url='https://github.com/MEIMEIMEIMEIMEMEDA/MAOSTE',
      packages=find_packages(),
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
      setup_requires=parse_requirements('requirements.txt'),
      cmdclass={'build_ext': BuildExtension},
  )