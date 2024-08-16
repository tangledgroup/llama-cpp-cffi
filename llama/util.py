__all__ = ['is_cuda_available', 'is_vulkan_available']

try:
    from numba import cuda
except Exception:
    pass

try:
    import vulkan as vk
except Exception:
    pass


def is_cuda_available():
    r: bool = False
    
    try:
        r = cuda.is_available()
    except Exception:
        r = False

    return r


def is_vulkan_available():
    vulkan_available: bool = False

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
