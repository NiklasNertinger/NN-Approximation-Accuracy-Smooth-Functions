def standard_architecture(depth, width):
    """
    Creates an architecture with fixed layer width.

    Arguments:
        depth (int): Number of layers in the network.
        width (int): Number of neurons per layer.

    Returns:
        list: 
            - layers (list): List with the number of neurons per layer.
            - name (str): Name of the architecture in the form 'standard{width}'.
    """
    layers = [width] * depth
    name = f"standard{width}"
    return [layers, name]

def increasing_architecture(depth, width):
    """
    Creates a triangular architecture where the width of the layers increases with depth.

    Arguments:
        depth (int): Number of layers in the network.
        width (int): Base width, multiplied by the layer number.

    Returns:
        list: 
            - layers (list): List with the number of neurons per layer.
            - name (str): Name of the architecture in the form 'increasing{width}'.
    """
    layers = [(i + 1) * width for i in range(depth)]
    name = f"increasing{width}"
    return [layers, name]

def decreasing_architecture(depth, width):
    """
    Creates a mirrored triangular architecture where the width of the layers decreases with depth.

    Arguments:
        depth (int): Number of layers in the network.
        width (int): Base width, multiplied by the remaining depth.

    Returns:
        list: 
            - layers (list): List with the number of neurons per layer.
            - name (str): Name of the architecture in the form 'decreasing{width}'.
    """
    layers = [(depth - i) * width for i in range(depth)]
    name = f"decreasing{width}"
    return [layers, name]

def diamond_architecture(depth, width):
    """
    Creates a diamond-shaped architecture where the width first increases and then decreases.

    Arguments:
        depth (int): Number of layers in the network.
        width (int): Base width, scaled for the layers.

    Returns:
        list: 
            - layers (list): List with the number of neurons per layer.
            - name (str): Name of the architecture in the form 'diamond{width}'.
    """
    layers = [2 * min(i + 1, depth - i) * width for i in range(depth)]
    layers[0], layers[-1] = width, width
    name = f"diamond{width}"
    return [layers, name]

def sandglass_architecture(depth, width):
    """
    Creates a sandglass-shaped architecture where the width decreases towards the middle and then increases again.

    Arguments:
        depth (int): Number of layers in the network.
        width (int): Base width, scaled for the layers.

    Returns:
        list: 
            - layers (list): List with the number of neurons per layer.
            - name (str): Name of the architecture in the form 'sandglass{width}'.

    Special cases:
        - If `depth` is odd, there is a single middle layer with the smallest width (`width`).
        - If `depth` is even, there are two middle layers with the smallest width (`width`).
    """
    layers = []

    if depth % 2 == 1:
        # There is a unique middle
        mid = depth // 2
        for i in range(depth):
            if i <= mid:
                layers.append((mid - i + 1) * width * 2)
            else:
                layers.append((i - mid + 1) * width * 2)
        layers[mid] = width
    else:
        # There are two middle layers
        mid1, mid2 = (depth // 2) - 1, depth // 2
        for i in range(depth):
            layers.append(2 * min(abs(mid1 - i) + 1, abs(mid2 - i) + 1) * width)
        layers[mid1], layers[mid2] = width, width

    name = f"sandglass{width}"
    return [layers, name]
