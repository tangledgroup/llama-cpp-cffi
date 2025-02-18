__all__ = [
    'is_cuda_available',
    'is_vulkan_available',
]

import subprocess

"""
try:
    from numba import cuda
except Exception:
    cuda = None

try:
    import vulkan as vk
except Exception:
    vk = None


def is_cuda_available() -> bool:
    cuda_available: bool = False

    if cuda is None:
        return cuda_available

    try:
        cuda_available = cuda.is_available()
    except Exception:
        pass

    return cuda_available


def is_vulkan_available() -> bool:
    vulkan_available: bool = False

    if vk is None:
        return vulkan_available

    try:
        # Load the Vulkan library and create an instance
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Vulkan Check",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="No Engine",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )

        instance_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
        )

        # Try creating a Vulkan instance
        instance = vk.vkCreateInstance(instance_info, None)

        # If we reach this point, Vulkan is available
        vulkan_available = True

        # Clean up the Vulkan instance
        vk.vkDestroyInstance(instance, None)
    except Exception:
        pass

    return vulkan_available
"""

def is_cuda_available() -> bool:
    """
    Check if CUDA is available by verifying the presence of the `nvidia-smi` command.
    Returns True if CUDA is available, False otherwise.
    """
    try:
        # Run the `nvidia-smi` command to check for CUDA support
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If `nvidia-smi` is not found or fails, CUDA is not available
        return False

    return True


def is_vulkan_available() -> bool:
    """
    Check if Vulkan is available by verifying the presence of the `vulkaninfo` command.
    Returns True if Vulkan is available, False otherwise.
    """
    try:
        # Run the `vulkaninfo` command to check for Vulkan support
        result = subprocess.run(['vulkaninfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If `vulkaninfo` is not found or fails, Vulkan is not available
        return False

    return True
