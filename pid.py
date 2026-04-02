import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameter dari firmware
# =========================
Kp = 2.5
Ki = 0.02
Kd = 0.9

ENC_RES = 4096
DT = 0.01           # 10 ms loop
SIM_TIME = 3.0      # detik
DEADBAND = 30
PID_LIMIT = 999

TARGET_ANGLE_DEG = 180.0
TARGET_RAW = int((TARGET_ANGLE_DEG * ENC_RES) / 360.0)

# =========================
# Helper functions
# =========================
def angle_error(target, current):
    err = target - current
    if err > ENC_RES // 2:
        err -= ENC_RES
    elif err < -ENC_RES // 2:
        err += ENC_RES
    return err

def deadband(x):
    return 0 if abs(x) < DEADBAND else x

# =========================
# Simulasi
# =========================
steps = int(SIM_TIME / DT)
time = np.linspace(0, SIM_TIME, steps)

position = np.zeros(steps)
velocity = 0.0

integral = 0.0
last_error = 0.0

# kondisi awal
position[0] = 0

# parameter motor (model sederhana)
MOTOR_GAIN = 0.015
MOTOR_DAMP = 0.2

pid_out_log = np.zeros(steps)
error_log = np.zeros(steps)

for i in range(1, steps):
    err = angle_error(TARGET_RAW, position[i-1])
    error_log[i] = err

    integral += err
    integral = np.clip(integral, -20000, 20000)

    if abs(err) < 10:
        integral *= 0.5

    derivative = err - last_error
    last_error = err

    pid = (Kp * err) + (Ki * integral) + (Kd * derivative)
    pid = np.clip(pid, -PID_LIMIT, PID_LIMIT)
    pid = deadband(pid)

    pid_out_log[i] = pid

    # model motor (first order)
    acceleration = MOTOR_GAIN * pid - MOTOR_DAMP * velocity
    velocity += acceleration
    position[i] = (position[i-1] + velocity) % ENC_RES

# =========================
# Plot
# =========================
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(time, position * 360 / ENC_RES, label="Position (deg)")
plt.axhline(TARGET_ANGLE_DEG, linestyle="--")
plt.ylabel("Angle (deg)")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(time, error_log * 360 / ENC_RES)
plt.ylabel("Error (deg)")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(time, pid_out_log)
plt.ylabel("PID Output")
plt.xlabel("Time (s)")
plt.grid(True)

plt.tight_layout()
plt.show()
