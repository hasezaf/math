import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


#CONSTANTS
g = 9.81 
dt = 0.001

#INPUTS
velocity = float(input("Enter initial velocity (m/s): "))
angle_degrees = float(input("Enter angle of projection (degrees): "))
k = float(input("Enter drag coefficient k (e.g. 0.005): "))

#FUNCTION TO RUN SIMULATION WITH DRAG
def run_simulation(v0, angle_deg, drag_k):
    rad = np.radians(angle_deg)
    vx = v0 * np.cos(rad)
    vy = v0 * np.sin(rad)
    
    x = [0.0]
    y = [0.0]
    count = 0
    
    while y[-1] >= 0:
        count +=1
        v_total = np.sqrt(vx**2 + vy**2)
        acceleration_x = -drag_k * v_total * vx
        acceleration_y = -g - drag_k * v_total * vy
        
        vx += acceleration_x * dt
        vy += acceleration_y * dt
        
        x.append(x[-1] + vx * dt)
        y.append(y[-1] + vy * dt)
        
    return np.array(x), np.array(y), count

_,_,range_for_graph = run_simulation(velocity, angle_degrees, k)

#CALCULATE IDEAL TRAJECTORY (WITHOUT DRAG)
angle_rad = np.radians(angle_degrees)
vy0 = velocity * np.sin(angle_rad)
vx0 = velocity * np.cos(angle_rad)
t_ideal = np.linspace(0, (2 * vy0) / g, num=range_for_graph)
x_ideal = vx0 * t_ideal
y_ideal = vy0 * t_ideal - 0.5 * g * t_ideal**2

#FINDING OPTIMAL ANGLE FOR MAXIMUM RANGE WITH DRAG
x_drag, y_drag , _ = run_simulation(velocity, angle_degrees, k)
angles = np.arange(0, 91, 1)
ranges = []

for a in angles:
    x_sim, _ , _ = run_simulation(velocity, a, k)
    ranges.append(x_sim[-1])

ranges = np.array(ranges)
opt_index = np.argmax(ranges)
opt_angle = angles[opt_index]
opt_range = ranges[opt_index]

print(f"\n--- RESULTS ---")
print(f"Optimal Angle for k={k}: {opt_angle}° (Max Range: {opt_range:.2f} m)")

#Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.set_xlim(0, max(max(x_drag),max(x_ideal)) * 1.05)
ax1.set_ylim(0, max(max(y_drag),max(y_ideal)) * 1.1)
(line,) = ax1.plot([], [], "b-", lw=2, label="Trajectory(Drag)")
(point,) = ax1.plot([], [], "ro", markersize=7, label="Projectile(Drag)")
(line1,) = ax1.plot([], [], "g-", lw=2, label="Trajectory(Ideal)")
(point1,) = ax1.plot([], [], "ro", markersize=7, label="Projectile(Ideal)")

#PLOT 1: TRAJECTORY COMPARISON (WITH AND WITHOUT DRAG)
ax1.set_title(f'Trajectory at {angle_degrees}° Launch')
ax1.set_xlabel('Horizontal Distance (m)')
ax1.set_ylabel('Vertical Height (m)')
ax1.grid(True)
ax1.legend()

#PLOT 2: MAX RANGE VS LAUNCH ANGLE (WITH DRAG)
ax2.plot(angles, ranges, 'b-', lw=2, label='Max Range with Drag')
ax2.plot(opt_angle, opt_range, 'ro', markersize=7, label=f'Optimal Angle: {opt_angle}°')
ax2.set_title('Max Range vs Launch Angle')
ax2.set_xlabel('Launch Angle (degrees)')
ax2.set_ylabel('Total Horizontal Range (m)')
ax2.grid(True)
ax2.legend()

#RUN ANIMATION
step = 15
def update(frame):
    line1.set_data(x_ideal[:frame], y_ideal[:frame])
    point1.set_data([x_ideal[frame]], [y_ideal[frame]])
    line.set_data(x_drag[:frame], y_drag[:frame])
    point.set_data([x_drag[frame]], [y_drag[frame]])
    return line, point, line1, point1

step = max(1, range_for_graph//100)
anim = FuncAnimation(
    fig,
    update,
    frames=range(0, range_for_graph, step),
    interval=15,
    blit=True,
    repeat=True,
)

anim.save("trajectory.gif", writer="pillow", fps=30, dpi=80)
plt.show()





