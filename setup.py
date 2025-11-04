from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="MotionHeatmapGenerator",
    version="0.1.0",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="motion heatmap video analysis computer-vision",
    url="https://github.com/ylp1455/MotionHeatmapGenerator",
    author="Yasiru Perera",
    author_email="yasiruperera681@gmail.com",
    description="A Python package for generating motion heatmaps from video sequences.",
    long_description_content_type="text/markdown",
    long_description=long_description,
    install_requires=[
        "opencv-python>=4.0.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=False,
)
