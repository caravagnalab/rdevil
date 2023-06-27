pydevil <- basilisk::BasiliskEnvironment(
  envname = "pydevil", pkgname = "rdevil",
  packages = c(
    "python==3.10",
    "pandas==1.5.3",
    "seaborn==0.12.2",
    "numpy==1.24.2",
    "matplotlib==3.7",
    "scikit-learn==1.2.1",
    "scipy==1.10.1",
    "statsmodels==0.14.0",
    "tqdm==4.65.0"
  ),
  pip = c("pyro-ppl==1.8.5"),
  channels = c("bioconda", "conda-forge"),
  path = "pydevil"
)
