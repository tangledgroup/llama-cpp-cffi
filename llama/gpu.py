__all__ = [
    'is_cuda_available',
    'is_vulkan_available',
]

try:
    from numba import cuda
except Exception:
    cuda = None

try:
    import vulkan as vk
except Exception:
    vk = None


def is_cuda_available():
    cuda_available: bool = False

    if cuda is None:
        return cuda_available

    try:
        cuda_available = cuda.is_available()
    except Exception:
        pass

    return cuda_available


def is_vulkan_available():
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
