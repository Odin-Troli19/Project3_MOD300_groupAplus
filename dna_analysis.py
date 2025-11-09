import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

class SimulationBox:
    """
    We created this class to represent a 3D box where our Monte Carlo simulation happens.
    Like a container in 3D space where we will throw random points.
    
    This class stores information about:
        x_min, x_max: The left and right walls of the box (in Angstrom units)
        y_min, y_max: The front and back walls of the box (in Angstrom units)
        z_min, z_max: The bottom and top walls of the box (in Angstrom units)
        volume: How much space is inside the box (in Angstrom^3)
    
    Note: Angstrom (Å) is a unit for measuring very small distances(perfect for atoms and molecules).
    1 Angstrom = 0.0000000001 meters.
    """
    
    def __init__(self, x_range: Tuple[float, float], 
                 y_range: Tuple[float, float], 
                 z_range: Tuple[float, float]):
       
        """
        We created this initialization function to set up our simulation box.
        When we create a new box, and then we tell it how big it should be in each direction.
        
        Parameters:
            x_range: A pair of numbers (min, max) telling us the box size in x direction
            y_range: A pair of numbers (min, max) telling us the box size in y direction
            z_range: A pair of numbers (min, max) telling us the box size in z direction
        
        """
        # We store the min and max positions for each direction
        # This tells us where the walls of our box are located
        self.x_min, self.x_max = x_range # Unpack the x boundaries
        self.y_min, self.y_max = y_range # Unpack the y boundaries
        self.z_min, self.z_max = z_range # Unpack the z boundaries
        
        # We calculate how wide the box is in each direction
        # This is simply: maximum position minus minimum position
        self.x_size = self.x_max - self.x_min # Width of box in x direction
        self.y_size = self.y_max - self.y_min # Width of box in y direction
        self.z_size = self.z_max - self.z_min # Width of box in z direction
        
        # We calculate the total volume of the box
        # Volume of a rectangular box = length × width × height
        # We multiply the size in all three directions
        self.volume = self.x_size * self.y_size * self.z_size
    
    def __repr__(self):
        """
        We created this function to show information about our box in a readable way.
        When we print the box object, Python will use this function to display it nicely.
        
        This is called a "string representation" - it converts our box data into text.
        """
        # We use f-strings (the f before the quotes) to insert our values into the text
        # The :.2f means "show this number with 2 digits after the decimal point"
        # Å³ is the symbol for cubic Angstroms (volume unit)
        return (f"SimulationBox(x=[{self.x_min:.2f}, {self.x_max:.2f}], "
                f"y=[{self.y_min:.2f}, {self.y_max:.2f}], "
                f"z=[{self.z_min:.2f}, {self.z_max:.2f}], "
                f"Volume={self.volume:.2f} Å³)")
    


class Sphere:
    
    """
    We created this class to represent a sphere (a perfect ball shape) in 3D space.
    In our project, each sphere represents one atom in the DNA molecule.
      Attributes:
        x, y, z: Center coordinates (Angstrom)
        radius: Sphere radius (Angstrom)
        atom_type: Optional atom type identifier
    
    This class stores information about:
        x, y, z: The position of the sphere's center in 3D space (in Angstrom units)
        radius: How big the sphere is - distance from center to surface (in Angstrom)
        atom_type: What kind of atom this sphere represents (optional)
                   Examples: 'H' for Hydrogen, 'O' for Oxygen, 'C' for Carbon
    """
    
    def __init__(self, x: float, y: float, z: float, 
                 radius: float, atom_type: Optional[str] = None):
        
        """
        We created this initialization function to set up a new sphere.
        When we create a sphere, we need to tell it where it is and how big it is.
        
        Parameters:
            x, y, z: Three numbers that tell us where the center of the sphere is located
                     - x: left/right position
                     - y: front/back position  
                     - z: up/down position
                     (All measured in Angstrom units)
            
            radius: One number that tells us how big the sphere is
                    This is the distance from the center to the edge of the sphere
                    (Measured in Angstrom units)
            
            atom_type: Optional text that tells us what type of atom this is
                       We write "Optional" because we don't always need to provide this
                       The "= None" means if we don't give an atom type, it will be None
        
        Example: Sphere(5.0, 3.0, 2.0, 1.2, 'H') creates a Hydrogen atom
                 at position (5, 3, 2) with radius 1.2 Angstroms
        """
        # We store the x position (left/right) of the sphere's center
        self.x = x

        # We store the y position (front/back) of the sphere's center
        self.y = y

        # We store the z position (up/down) of the sphere's center
        self.z = z

        # We store how big the sphere is (its radius)
        # The radius is the distance from the center to any point on the surface
        self.radius = radius

        # We store what type of atom this sphere represents
        # This could be 'H' (Hydrogen), 'O' (Oxygen), 'C' (Carbon), etc.
        # If no atom type is given, this will be None
        self.atom_type = atom_type



class Sphere:
    """
    We created this class to represent a sphere (a perfect ball shape) in 3D space.
    In our project, each sphere represents one atom in the DNA molecule.
    
    This class stores information about:
        x, y, z: The position of the sphere's center in 3D space (in Angstrom units)
        radius: How big the sphere is (in Angstrom)
        atom_type: What kind of atom this sphere represents
    """
    
    def __init__(self, x: float, y: float, z: float, 
                 radius: float, atom_type: Optional[str] = None):
        """
        We created this initialization function to set up a new sphere.
        """
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.atom_type = atom_type
    
    def is_point_inside(self, point_x: float, point_y: float, 
                       point_z: float) -> bool:
        """
        We created this function to check if a random point is inside this sphere or not.
        
        How it works:
        1. We calculate the distance from the point to the sphere's center
        2. If the distance is less than or equal to the radius, the point is inside
        3. If the distance is greater than the radius, the point is outside
        
        Parameters (what we give to the function):
            point_x: The x position of the point we want to check
            point_y: The y position of the point we want to check
            point_z: The z position of the point we want to check
            
        Returns (what the function gives back):
            True if the point is inside the sphere
            False if the point is outside the sphere
        
        Mathematical explanation:
        - We use the 3D distance formula: distance² = (x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²
        - We compare distance² with radius² (we use squared values to avoid calculating 
          square roots, which makes the code faster)
        - If distance² ≤ radius², the point is inside or on the surface
        """


        # Calculate the squared distance from the point to the sphere's center
        # We subtract each coordinate of the sphere's center from the point's coordinates
        # Then we square each difference and add them together
        distance_squared = ((point_x - self.x)**2 + # (x difference)²
                          (point_y - self.y)**2 +   # (y difference)²
                          (point_z - self.z)**2)    # (z difference)²
        
        # Check if the squared distance is less than or equal to squared radius
        # We use <= (less than or equal) to include points exactly on the surface
        # This returns True (inside) or False (outside)
        return distance_squared <= self.radius**2
    
    def volume(self) -> float:
        """
        We created this function to calculate the exact volume of the sphere.
        
        The volume formula for a sphere is: V = (4/3) × π × r³
        where r is the radius
        
        Returns:
            The volume of the sphere in cubic Angstroms (Å³)
        
        Example: A sphere with radius 2 Å has volume ≈ 33.5 Å³
        """
        # Apply the sphere volume formula
        # (4.0 / 3.0) is the fraction 4/3
        # np.pi is the number pi (≈ 3.14159...)
        # self.radius**3 means radius × radius × radius (radius cubed)
        return (4.0 / 3.0) * np.pi * self.radius**3
    
    def __repr__(self):
        """
        We created this function to show information about our sphere in a readable way.
        When we print a sphere, Python uses this function to display it nicely.
        """
        # If the sphere has an atom type, we add it to the display
        # Otherwise, we show an empty string (nothing)
        atom_str = f", {self.atom_type}" if self.atom_type else ""

        # Create a nice text representation showing the sphere's properties
        # The :.2f means "show 2 digits after the decimal point"
        return (f"Sphere(center=({self.x:.2f}, {self.y:.2f}, {self.z:.2f}), "
                f"radius={self.radius:.2f}{atom_str})")


def generate_random_points(box: SimulationBox, n_points: int) -> np.ndarray:
    """
    We created this function to generate many random points inside our simulation box.
    This is like throwing darts randomly into a box - each dart lands at a random position.
    
    This is the core of our Monte Carlo method: we throw random points and check
    how many land inside the DNA spheres. From this, we can calculate the DNA volume.
    
    Parameters (what we give to the function):
        box: The SimulationBox object that defines our 3D space
        n_points: How many random points we want to create
        
    Returns (what the function gives back):
        A numpy array with shape (n_points, 3)
        Each row contains [x, y, z] coordinates for one random point
        
    Example: If n_points = 1000, we get 1000 random points, each with x, y, z coordinates
    """
    # Generate n_points random x coordinates between box.x_min and box.x_max
    # np.random.uniform creates random numbers that are equally likely anywhere
    # in the range (this is called "uniform distribution")
    x_points = np.random.uniform(box.x_min, box.x_max, n_points)

    # Generate n_points random y coordinates between box.y_min and box.y_max
    y_points = np.random.uniform(box.y_min, box.y_max, n_points)

    # Generate n_points random z coordinates between box.z_min and box.z_max
    z_points = np.random.uniform(box.z_min, box.z_max, n_points)
    
    # Combine the x, y, z coordinates into one array
    # np.column_stack puts the three arrays side by side as columns
    # Result: [[x₁, y₁, z₁], [x₂, y₂, z₂], [x₃, y₃, z₃], ...]
    return np.column_stack([x_points, y_points, z_points])


def create_random_sphere(box: SimulationBox, 
                        min_radius: float = 1.0, 
                        max_radius: float = 5.0) -> Sphere:
    """
    We created this function to make a sphere with random position and random size.
    This is useful for testing our Monte Carlo code with simple spheres before
    moving to real DNA data.
    
    Parameters (what we give to the function):
        box: The SimulationBox where we want to place the sphere
        min_radius: The smallest radius the sphere can have (default = 1.0 Angstrom)
        max_radius: The largest radius the sphere can have (default = 5.0 Angstrom)
        
    Returns (what the function gives back):
        A new Sphere object with random position and random size
        
    Note: The "= 1.0" and "= 5.0" are default values. If we don't specify min_radius
    and max_radius when calling the function, it will use these values automatically.
    """
    # Generate a random x position anywhere inside the box
    # np.random.uniform picks a random number between x_min and x_max
    x = np.random.uniform(box.x_min, box.x_max)

    # Generate a random y position anywhere inside the box
    y = np.random.uniform(box.y_min, box.y_max)

    # Generate a random z position anywhere inside the box
    z = np.random.uniform(box.z_min, box.z_max)
    
    # Generate a random radius between min_radius and max_radius
    # This determines how big the sphere will be
    radius = np.random.uniform(min_radius, max_radius)
    
    # Create and return a new Sphere object with the random values we generated
    # We don't specify atom_type, so it will be None
    return Sphere(x, y, z, radius)


def count_points_in_spheres(points: np.ndarray, 
                            spheres: List[Sphere]) -> int:
    """
    We created this function to count how many random points are inside any of the spheres.
    This is the key step in our Monte Carlo method for calculating volume.
    
    How it is suppose to work:
    1. We look at each random point one by one
    2. For each point, we check if it's inside any of the spheres
    3. If a point is inside at least one sphere, we count it (only once)
    4. We return the total count
    
    Parameters (what we give to the function):
        points: A numpy array of shape (n_points, 3) containing all our random points
                Each row is [x, y, z] coordinates of one point
        spheres: A list containing all the Sphere objects we want to check
        
    Returns (what the function gives back):
        An integer: the number of points that are inside at least one sphere
        
    Example: If we throw 10,000 random points and 3,000 land inside the spheres,
             this function returns 3,000
    
    Important: If a point is inside multiple overlapping spheres, we only count it ONCE.
               This is correct because we want to know if the point is inside "any DNA atom",
               not "how many DNA atoms" it's inside.
    """

    # Start with a counter at zero
    # We will increase this counter each time we find a point inside a sphere
    n_inside = 0
    
    # Loop through each random point
    # "for point in points" means: take each row from the points array one by one
    for point in points:
        # Extract the x, y, z coordinates from this point
        # This unpacks the array [x, y, z] into three separate variables
        px, py, pz = point

        # Check if this point is inside any of the spheres
        # We go through each sphere in our list
        for sphere in spheres:

            # Use the sphere's is_point_inside function to check
            # This returns True if the point is inside, False if outside
            if sphere.is_point_inside(px, py, pz):

                # The point is inside this sphere!
                # Increase our counter by 1
                n_inside += 1

                # Use "break" to stop checking other spheres
                # We already found that this point is inside, so we don't need
                # to check the remaining spheres. This makes the code faster
                # and ensures we count each point only once.
                break  # Exit the inner loop and move to the next point
    
    # Return the total number of points we found inside the spheres
    return n_inside

# We took inspiration from the Monte Carlo function shown in class
# Source: MC_area_volume_central_theorem.ipynb (cells 3-5) from class-004
# The teacher showed us a basic "hit-and-miss" Monte Carlo method
# We adapted that idea to work with our DNA spheres

def monte_carlo_volume(box: SimulationBox, 
                       spheres: List[Sphere], 
                       n_points: int) -> Tuple[float, float]:
    """
    Calculate volume of spheres using Monte Carlo integration.
    
    Parameters:
        box: SimulationBox for sampling
        spheres: List of Sphere objects
        n_points: Number of random points to sample
        
    Returns:
        Tuple of (estimated_volume, fraction_inside)
    """
    points = generate_random_points(box, n_points)
    n_inside = count_points_in_spheres(points, spheres)
    
    fraction_inside = n_inside / n_points
    estimated_volume = fraction_inside * box.volume
    
    return estimated_volume, fraction_inside


def calculate_pi_from_sphere(n_points: int, radius: float = 1.0) -> float:
    """
    Calculate pi using Monte Carlo method with a sphere.
    
    Parameters:
        n_points: Number of random points to use
        radius: Radius of sphere (default 1.0)
        
    Returns:
        Estimated value of pi
    """
    # Create a box that contains the sphere
    box = SimulationBox((-radius, radius), (-radius, radius), (-radius, radius))
    sphere = Sphere(0, 0, 0, radius)
    
    # Monte Carlo sampling
    volume_estimate, _ = monte_carlo_volume(box, [sphere], n_points)
    
    # V_sphere = (4/3) * pi * r^3
    # pi = 3 * V_sphere / (4 * r^3)
    pi_estimate = 3 * volume_estimate / (4 * radius**3)
    
    return pi_estimate


# Atomic radii dictionary (in picometers, will convert to Angstrom)
ATOMIC_RADII_PM = {
    'H': 120,  # Hydrogen
    'C': 170,  # Carbon
    'N': 155,  # Nitrogen
    'O': 152,  # Oxygen
    'P': 180,  # Phosphorus
}

def pm_to_angstrom(pm: float) -> float:
    """Convert picometers to Angstrom."""
    return pm / 100.0


def read_dna_coordinates(filename: str) -> List[Sphere]:
    """
    Read DNA atomic coordinates from file and create Sphere objects.
    
    File format: atom_type  x  y  z (coordinates in Angstrom)
    
    Parameters:
        filename: Path to coordinate file
        
    Returns:
        List of Sphere objects representing atoms
    """
    spheres = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                atom_type = parts[0]
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                
                # Get atomic radius (convert from pm to Angstrom)
                radius_pm = ATOMIC_RADII_PM.get(atom_type, 170)  # Default to C
                radius = pm_to_angstrom(radius_pm)
                
                spheres.append(Sphere(x, y, z, radius, atom_type))
    
    return spheres


def get_bounding_box(spheres: List[Sphere], 
                     padding: float = 5.0) -> SimulationBox:
    """
    Create a simulation box that contains all spheres with padding.
    
    Parameters:
        spheres: List of Sphere objects
        padding: Extra space around molecules (Angstrom)
        
    Returns:
        SimulationBox that contains all spheres
    """
    # Find min/max coordinates considering sphere radii
    x_coords = [s.x for s in spheres]
    y_coords = [s.y for s in spheres]
    z_coords = [s.z for s in spheres]
    radii = [s.radius for s in spheres]
    
    x_min = min(x_coords) - max(radii) - padding
    x_max = max(x_coords) + max(radii) + padding
    y_min = min(y_coords) - max(radii) - padding
    y_max = max(y_coords) + max(radii) + padding
    z_min = min(z_coords) - max(radii) - padding
    z_max = max(z_coords) + max(radii) + padding
    
    return SimulationBox((x_min, x_max), (y_min, y_max), (z_min, z_max))


def convergence_analysis(box: SimulationBox, 
                         spheres: List[Sphere],
                         n_points_list: List[int],
                         n_repeats: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze convergence of Monte Carlo volume calculation.
    
    Parameters:
        box: SimulationBox for sampling
        spheres: List of Sphere objects
        n_points_list: List of different n_points to test
        n_repeats: Number of repetitions for each n_points
        
    Returns:
        Tuple of (mean_volumes, std_volumes) for each n_points
    """
    mean_volumes = []
    std_volumes = []
    
    for n_points in n_points_list:
        volumes = []
        for _ in range(n_repeats):
            vol, _ = monte_carlo_volume(box, spheres, n_points)
            volumes.append(vol)
        
        mean_volumes.append(np.mean(volumes))
        std_volumes.append(np.std(volumes))
    
    return np.array(mean_volumes), np.array(std_volumes)


# ============================================================================
# TOPIC 2: Random Walk for Accessible Volume
# ============================================================================

class RandomWalker:
    """
    Represents a random walker in 3D space.
    
    Attributes:
        position: Current position [x, y, z]
        trajectory: History of positions
        step_size: Size of each random step
    """
    
    def __init__(self, start_position: np.ndarray, step_size: float = 1.0):
        """
        Initialize a random walker.
        
        Parameters:
            start_position: Starting [x, y, z] coordinates
            step_size: Size of each random step (Angstrom)
        """
        self.position = np.array(start_position, dtype=float)
        self.trajectory = [self.position.copy()]
        self.step_size = step_size
    
    def take_step(self):
        """Take one random step in 3D space."""
        # Random direction: choose uniformly from 6 directions (+/- x, y, z)
        direction = np.random.randint(0, 6)
        
        step = np.zeros(3)
        if direction == 0:
            step[0] = self.step_size
        elif direction == 1:
            step[0] = -self.step_size
        elif direction == 2:
            step[1] = self.step_size
        elif direction == 3:
            step[1] = -self.step_size
        elif direction == 4:
            step[2] = self.step_size
        else:  # direction == 5
            step[2] = -self.step_size
        
        self.position += step
        self.trajectory.append(self.position.copy())
    
    def is_inside_spheres(self, spheres: List[Sphere]) -> bool:
        """
        Check if current position is inside any sphere.
        
        Parameters:
            spheres: List of Sphere objects to check
            
        Returns:
            True if inside any sphere, False otherwise
        """
        px, py, pz = self.position
        for sphere in spheres:
            if sphere.is_point_inside(px, py, pz):
                return True
        return False


def generate_random_walkers(n_walkers: int, 
                            box: SimulationBox,
                            step_size: float = 1.0) -> List[RandomWalker]:
    """
    Generate multiple random walkers at random starting positions.
    
    Parameters:
        n_walkers: Number of walkers to create
        box: SimulationBox for initial positions
        step_size: Step size for each walker
        
    Returns:
        List of RandomWalker objects
    """
    walkers = []
    for _ in range(n_walkers):
        start_x = np.random.uniform(box.x_min, box.x_max)
        start_y = np.random.uniform(box.y_min, box.y_max)
        start_z = np.random.uniform(box.z_min, box.z_max)
        
        walker = RandomWalker([start_x, start_y, start_z], step_size)
        walkers.append(walker)
    
    return walkers


def random_walk_fast(n_walkers: int,
                     n_steps: int,
                     box: SimulationBox,
                     step_size: float = 1.0) -> np.ndarray:
    """
    Fast vectorized implementation of random walk for multiple walkers.
    
    Parameters:
        n_walkers: Number of walkers
        n_steps: Number of steps per walker
        box: SimulationBox for initial positions
        step_size: Step size for each walker
        
    Returns:
        Array of shape (n_walkers, n_steps+1, 3) with all trajectories
    """
    # Initialize positions
    positions = np.random.uniform(
        low=[box.x_min, box.y_min, box.z_min],
        high=[box.x_max, box.y_max, box.z_max],
        size=(n_walkers, 3)
    )
    
    # Store all trajectories
    trajectories = np.zeros((n_walkers, n_steps + 1, 3))
    trajectories[:, 0, :] = positions
    
    # Generate all random steps at once
    # Random directions: 0-5 for +/- x, y, z
    directions = np.random.randint(0, 6, size=(n_walkers, n_steps))
    
    # Convert directions to steps
    for step in range(n_steps):
        step_vectors = np.zeros((n_walkers, 3))
        
        # +x direction (0)
        mask = directions[:, step] == 0
        step_vectors[mask, 0] = step_size
        
        # -x direction (1)
        mask = directions[:, step] == 1
        step_vectors[mask, 0] = -step_size
        
        # +y direction (2)
        mask = directions[:, step] == 2
        step_vectors[mask, 1] = step_size
        
        # -y direction (3)
        mask = directions[:, step] == 3
        step_vectors[mask, 1] = -step_size
        
        # +z direction (4)
        mask = directions[:, step] == 4
        step_vectors[mask, 2] = step_size
        
        # -z direction (5)
        mask = directions[:, step] == 5
        step_vectors[mask, 2] = -step_size
        
        positions += step_vectors
        trajectories[:, step + 1, :] = positions
    
    return trajectories


def accessible_volume_random_walk(spheres: List[Sphere],
                                 box: SimulationBox,
                                 n_walkers: int = 1000,
                                 n_steps: int = 1000,
                                 step_size: float = 0.5) -> float:
    """
    Calculate accessible volume using random walk method.
    
    Strategy: Start walkers at random positions. If they enter a sphere
    (DNA atom), mark that position as inaccessible. Calculate fraction
    of accessible space.
    
    Parameters:
        spheres: List of Sphere objects (DNA atoms)
        box: SimulationBox for sampling
        n_walkers: Number of random walkers
        n_steps: Number of steps per walker
        step_size: Step size (Angstrom)
        
    Returns:
        Estimated accessible volume (Angstrom^3)
    """
    # Generate trajectories
    trajectories = random_walk_fast(n_walkers, n_steps, box, step_size)
    
    # Count accessible positions
    n_accessible = 0
    n_total = 0
    
    for walker_traj in trajectories:
        for position in walker_traj:
            n_total += 1
            px, py, pz = position
            
            # Check if position is inside any sphere
            inside = False
            for sphere in spheres:
                if sphere.is_point_inside(px, py, pz):
                    inside = True
                    break
            
            if not inside:
                n_accessible += 1
    
    # Calculate accessible volume
    fraction_accessible = n_accessible / n_total
    accessible_vol = fraction_accessible * box.volume
    
    return accessible_vol


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_convergence(n_points_list: List[int],
                    mean_volumes: np.ndarray,
                    std_volumes: np.ndarray,
                    true_volume: Optional[float] = None,
                    title: str = "Monte Carlo Convergence"):
    """
    Plot convergence of Monte Carlo volume calculation.
    
    Parameters:
        n_points_list: List of n_points values
        mean_volumes: Mean volumes for each n_points
        std_volumes: Standard deviations for each n_points
        true_volume: Optional true volume for comparison
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(n_points_list, mean_volumes, yerr=std_volumes,
                marker='o', capsize=5, label='MC estimate')
    
    if true_volume is not None:
        plt.axhline(y=true_volume, color='r', linestyle='--',
                   label=f'True volume = {true_volume:.2f} Å³')
    
    plt.xlabel('Number of Points')
    plt.ylabel('Volume (Å³)')
    plt.title(title)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_fraction_vs_points(n_points_list: List[int],
                           fractions: List[float],
                           title: str = "Fraction of Points Inside"):
    """
    Plot fraction of points inside vs number of points.
    
    Parameters:
        n_points_list: List of n_points values
        fractions: Fraction of points inside for each n_points
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(n_points_list, fractions, marker='o')
    
    plt.xlabel('Number of Points')
    plt.ylabel('Fraction Inside')
    plt.title(title)
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Simple test
    print("DNA Analysis Module")
    print("=" * 50)
    
    # Test simulation box
    box = SimulationBox((-10, 10), (-10, 10), (-10, 10))
    print(f"\n{box}")
    
    # Test sphere
    sphere = Sphere(0, 0, 0, 5.0, 'C')
    print(f"\n{sphere}")
    print(f"Sphere volume: {sphere.volume():.2f} Å³")
    
    # Test point generation
    points = generate_random_points(box, 10)
    print(f"\nGenerated {len(points)} random points")
    
    print("\nModule loaded successfully!")
