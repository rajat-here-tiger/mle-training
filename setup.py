from distutils.core import setup


def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc

        return pypandoc.convert("README.md", "rst")
    except (IOError, ImportError):
        with open("README.md") as readme_file:
            return readme_file.read()


setup(
    name="mle_training",
    version="0.1",
    description="Estimating Median House Values",
    long_description=readme(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    url="https://github.com/rajat-here-tiger/mle-training",
    author="Rajat Gupta",
    author_email="rajat.gupta@tigeranalytics.com",
    license="MIT",
    packages=["mle_training", "mle_training.utils"],
    install_requires=[
        "pypandoc>=1.4",
        "pytest>=4.3.1",
        "pytest-runner>=4.4",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
