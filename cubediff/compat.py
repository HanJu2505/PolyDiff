def ensure_torch_xpu_compat() -> None:
    try:
        import torch
    except Exception:
        return

    if not hasattr(torch, "xpu"):
        class _DummyXPU:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def empty_cache():
                return None

            @staticmethod
            def device_count():
                return 0

            @staticmethod
            def manual_seed(seed):
                return None

            @staticmethod
            def reset_peak_memory_stats(*args, **kwargs):
                return None

            @staticmethod
            def max_memory_allocated(*args, **kwargs):
                return 0

            @staticmethod
            def synchronize(*args, **kwargs):
                return None

            @staticmethod
            def current_device():
                return 0

            @staticmethod
            def get_device_properties(index):
                raise RuntimeError("XPU is not available in this environment")

            @staticmethod
            def get_device_capability():
                raise RuntimeError("XPU is not available in this environment")

        torch.xpu = _DummyXPU()

    if not hasattr(torch, "mps"):
        class _DummyMPS:
            @staticmethod
            def is_available():
                return False

            @staticmethod
            def empty_cache():
                return None

            @staticmethod
            def manual_seed(seed):
                return None

        torch.mps = _DummyMPS()
