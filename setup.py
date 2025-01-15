from setuptools import find_packages, setup

setup(
    name="customer_churn_mlops",
    version="0.1.0",
    description="An end-to-end MLOps project for customer churn prediction using CI/CD, DVC, MLflow, and Databricks.",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/customer-churn-mlops",
    packages=find_packages(),
    install_requires=[
        "scikit-learn",
        "pandas",
        "numpy",
        "mlflow",
        "dvc",
        "evidently",
        "databricks-feature-store",
        "pytest",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
