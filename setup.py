from setuptools import setup, find_packages

setup(
    name="MotionHeatmapGenerator",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="motion heatmap",
    url="https://github.com/YasirUperera/MotionHeatmapGenerator",
    author="Yasir Uperera",
    author_email="yasiruperera681@gmail.com",
    description="A Python package for generating motion heatmaps from video sequences.",
    long_description_content_type="text/markdown",
    long_description=open("README.md").read(),
    install_requires=[
        "opencv-python",
        "numpy",
        "scipy",
        "scikit-image",
    ],
    python_requires=">=3.6",
    include_package_data=True,
    zip_safe=False,
)
