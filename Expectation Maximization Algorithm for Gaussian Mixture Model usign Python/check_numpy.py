try:
    print("before raising exception")
    raise Exception
    print("numpy is installed")
except ImportError:
    print("numpy is not installed")