import pytest
from click.testing import CliRunner
from my_package.cli import run
from my_package._version import __version__

@pytest.fixture
def runner_result():
    runner = CliRunner()
    return runner.invoke(run)

def test_hello_world(runner_result):
    # Check exit code
    assert runner_result.exit_code == 0

    # Check first line of output
    first_line = runner_result.output.split("\n")[0]
    assert first_line == f"my_package {__version__}"
