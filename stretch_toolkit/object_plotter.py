import matplotlib
matplotlib.use('TkAgg')  # Use Tk backend for virtual desktop compatibility
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class ObjectPlotter:
    """Real-time 3D visualization of camera and object positions relative to robot base."""
    
    def __init__(self):
        """Initialize the 3D plot with interactive mode enabled."""
        plt.ion()  # Enable interactive mode
        
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Create plot artists (initialized with dummy data)
        self.base_scatter = self.ax.scatter([0], [0], [0], c='green', marker='o', s=100, label='Base Origin')
        self.cam_scatter = self.ax.scatter([0], [0], [0], c='blue', marker='s', s=100, label='Camera')
        
        # Create line artist for base to camera
        self.base_to_cam_line, = self.ax.plot([0, 0], [0, 0], [0, 0], 'b--', alpha=0.5)
        
        # Camera direction arrow (quiver)
        self.cam_arrow = self.ax.quiver(0, 0, 0, 0, 0, 1, 
                                        length=0.3, color='cyan', 
                                        arrow_length_ratio=0.3, linewidth=2)
        
        # Dynamic pool of object artists (scatter + line)
        self.obj_artists = []  # List of (scatter, line) tuples
        
        # Set labels and formatting
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.legend()
        self.ax.set_title('Object Position (Base Frame)')
        
        # Set fixed axis limits
        self.ax.set_xlim([-1.0, 1.0])
        self.ax.set_ylim([-1.0, 1.0])
        self.ax.set_zlim([0, 2.0])
        
        # Initial draw
        plt.draw()
        plt.pause(0.001)
    
    def _create_object_artists(self):
        """Create a new object artist tuple (scatter, line).
        
        Returns:
            Tuple of (scatter, line) artists
        """
        scatter = self.ax.scatter([0], [0], [0], c='red', marker='^', s=100)
        line, = self.ax.plot([0, 0], [0, 0], [0, 0], color='red', linestyle='--', alpha=0.5)
        return (scatter, line)
    
    def update(self, cam_T, obj_Ts):
        """Update the plot with new camera and object transforms.
        
        Args:
            cam_T: 4x4 camera-to-base transformation matrix (or None)
            obj_Ts: Single 4x4 object transform, or list of 4x4 object-to-base transformation matrices (or empty list)
        """
        if cam_T is None:
            return
        
        # Handle single object or list of objects
        if obj_Ts is None:
            obj_Ts = []
        elif isinstance(obj_Ts, np.ndarray):
            # Single transform matrix - wrap in list
            obj_Ts = [obj_Ts]
        
        # Extract camera position
        cam_x, cam_y, cam_z = cam_T[0:3, 3]
        
        # Update camera scatter
        self.cam_scatter._offsets3d = ([cam_x], [cam_y], [cam_z])
        
        # Update base to camera line
        self.base_to_cam_line.set_data([0, cam_x], [0, cam_y])
        self.base_to_cam_line.set_3d_properties([0, cam_z])
        
        # Update camera direction arrow
        # Remove old quiver and create new one (quiver doesn't have update method)
        self.cam_arrow.remove()
        cam_forward = cam_T[0:3, 2]  # Third column is Z-axis (forward) in camera frame
        self.cam_arrow = self.ax.quiver(cam_x, cam_y, cam_z,
                                        cam_forward[0], cam_forward[1], cam_forward[2],
                                        length=0.3, color='cyan',
                                        arrow_length_ratio=0.3, linewidth=2)
        
        # Handle objects
        num_objects = len(obj_Ts)
        
        # Grow artist pool if needed
        while len(self.obj_artists) < num_objects:
            new_artists = self._create_object_artists()
            self.obj_artists.append(new_artists)
        
        # Update visible objects
        for i, obj_T in enumerate(obj_Ts):
            if obj_T is None:
                continue
                
            scatter, line = self.obj_artists[i]
            obj_x, obj_y, obj_z = obj_T[0:3, 3]
            
            # Update scatter position
            scatter._offsets3d = ([obj_x], [obj_y], [obj_z])
            scatter.set_visible(True)
            
            # Update line from camera to object
            line.set_data([cam_x, obj_x], [cam_y, obj_y])
            line.set_3d_properties([cam_z, obj_z])
            line.set_visible(True)
        
        # Hide unused artists
        for i in range(num_objects, len(self.obj_artists)):
            scatter, line = self.obj_artists[i]
            scatter.set_visible(False)
            line.set_visible(False)
        
        # Refresh the plot
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def close(self):
        """Close the plot window."""
        plt.close(self.fig)
    
    def is_open(self):
        """Check if the plot window is still open."""
        return plt.fignum_exists(self.fig.number)