from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
  name="miso",
  version="0.1.0",
  description="Resolving tissue complexity by multi-modal spatial omics modeling with MISO",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/kpcoleman/miso",
  author="Kyle Coleman",
  author_email="kpcoleman87@gmail.com",
  packages=find_packages(),
  #include_package_data=True,
  python_requires="==3.11.*",
  package_data = {'miso': ['checkpoints/*.pth'],},
  install_requires=['einops==0.8.1','numpy==2.2.6','opencv-python==4.10.0.84','Pillow==11.2.1','scanpy==1.11.2','scikit-image==0.25.2','scikit-learn==1.7.0','scipy==1.15.2','setuptools==75.8.2','torch==2.4.0','torchvision==0.19.0','tqdm==4.67.1'],
  #install_requires=["scikit-learn==1.0.2","scikit_image==0.19.3","torch==1.13.1","torchvision==0.14.1","numpy==1.21.6","Pillow>=6.1.0","opencv-python==4.6.0.66","scipy==1.7.3","einops==0.6.0","scanpy==1.9.1","tqdm==4.64.1"],
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Operating System :: OS Independent",
  ]
)


