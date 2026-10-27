from packaging.version import Version

import clara_inference


def test_version():
    # from the git tag (setuptools-scm), or the fallback without git
    assert Version(clara_inference.__version__) >= Version("0.0.0")
