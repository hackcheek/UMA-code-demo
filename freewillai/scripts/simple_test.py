"""
This exists just because pytest is not working yet
due to this issue: 
    DeprecationWarning: This library has been renamed to `eth-abi`.  The `ethereum-abi-utils` package will no longer recieve updates.  Please update your dependencies accordingly.

Test run `pytest` in console and if it is not working 
try call here your test functions and run `python -m tests.simple_test`
"""
from tests.test_common import TestModelWrapper, TestDatasetWrapper


def run_tests(_class, match=''):
    test_methods = filter(
        lambda m: m.startswith('test_') and match in m, dir(_class)
    )
    
    for method in test_methods:
        print(f"\n[*] Testing {_class.__name__}.{method}")
        try:
            getattr(_class, method)(_class)
            print(f"  > success")
        except AssertionError:
            print(f"  > failed")


if __name__ == "__main__":
    run_tests(TestModelWrapper)
    run_tests(TestDatasetWrapper)
