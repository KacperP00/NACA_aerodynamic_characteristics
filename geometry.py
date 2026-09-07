import numpy as np

def generate_naca4(naca_code, n_points=100):
    # Generation of the chordwise position vector.
    # Use of a cosine distribution for clustering at the leading and trailing edges.
    beta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1.0 - np.cos(beta))
    
    # Extraction of parameters from the NACA code.
    m = int(naca_code[0]) / 100.0
    p = int(naca_code[1]) / 10.0
    t = int(naca_code[2:]) / 100.0

    # Calculation of the thickness distribution.
    # Coefficient -0.1015 (blunt trailing edge)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    # Calculation of the skeleton line and its derivative.
    if p > 0:
        front = x <= p
        back = x > p
        yc[front] = (m / p**2) * (2 * p * x[front] - x[front]**2)
        dyc_dx[front] = (2 * m / p**2) * (p - x[front])
        yc[back] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[back] - x[back]**2)
        dyc_dx[back] = (2 * m / (1 - p)**2) * (p - x[back])
        
    theta = np.arctan(dyc_dx)
    
    # Calculation of the upper and lower surface coordinates.
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # Concatenation of vectors: from the lower surface (trailing edge), through the leading edge, to the upper surface (trailing edge).
    x_coords = np.concatenate((xl[::-1], xu[1:]))
    y_coords = np.concatenate((yl[::-1], yu[1:]))
    
    return x_coords, y_coords