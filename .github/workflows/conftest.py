def pytest_addoption(parser):
    parser.addoption(
        "--dataset",
        action="append",
        default=[],
        help="list of datasets to pass to test_data_loading",
    )


def pytest_generate_tests(metafunc):
    if "dataset" in metafunc.fixturenames:
        metafunc.parametrize("dataset", metafunc.config.getoption("dataset"))
