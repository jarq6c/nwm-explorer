import pytest
import nwm_explorer
from nwm_explorer import my_module

def test_package_version():
    # Check version
    assert nwm_explorer.__version__ == "0.1.0"

@pytest.fixture
def default_dataframe():
    return my_module.make_dataframe()

def test_default_dataframe(default_dataframe):
    # Test defaults
    assert len(default_dataframe.columns) == 4
    assert len(default_dataframe.index) == 5
