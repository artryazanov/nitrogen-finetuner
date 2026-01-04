import os
import sys

import pytest

# Ensure 'src' is in the python path
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from src.config import DataArguments, ModelArguments, TrainingArguments


@pytest.fixture
def model_args():
    return ModelArguments()


@pytest.fixture
def data_args():
    return DataArguments()


@pytest.fixture
def training_args():
    return TrainingArguments()
