import pytest

import batchlas as bl

# A vendor-free build has no route for the vendor-only ops (hemm, herk, her2k, complex gram,
# some syr2k/syev/getrf shapes); NoRouteError there is the documented contract. Only that
# exception, and only in a vendor-free build, becomes a skip: anywhere else it still fails.
_VENDOR_FREE = bool(bl.compiled_features().get("vendor_free", False))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if (_VENDOR_FREE and call.when == "call" and report.failed and call.excinfo is not None
            and call.excinfo.errisinstance(bl.NoRouteError)):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, f"vendor-free: {call.excinfo.value}")
