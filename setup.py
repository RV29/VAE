import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="VAE-RV29", # Replace with your own username
    version="0.0.2",
    author="Christine Shen, Vidvat Ramachandran",
    author_email="vidvatrmc@gmail.com",
    description="A simple implementation of Variational Autoencoder for MNIST dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RV29/VAE",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
