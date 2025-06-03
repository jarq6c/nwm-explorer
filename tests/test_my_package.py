import pytest
import my_package
from my_package import my_module

def test_package_version():
    # Check version
    assert my_package.__version__ == "0.1.0"

@pytest.fixture
def default_dataframe():
    return my_module.make_dataframe()

def test_default_dataframe(default_dataframe):
    # Test defaults
    assert len(default_dataframe.columns) == 4
    assert len(default_dataframe.index) == 5
