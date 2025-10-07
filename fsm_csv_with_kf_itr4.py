'''
iteration3 considered the covariance matrix to be identity matrix. 
This code (iteration4) considers a data-driven Q matrix based on the dataset's acceleration noise variance.
'''

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

# =================== CONSTANTS ===================
T_0 = 288.15 #310.15 og
L = 0.0065
P_0 = 101325 #101050 og
R = 287
g = 9.80665
ELEVATION = 882

# Thresholds
LAUNCH_DETECTION = (5 * g)
LAUNCH_THRESHOLD = 5
COASTING_THRESHOLD = 3
APOGEE_THRESHOLD = 5
MAIN_ALTITUDE = 457
RECOVERY_CONDITION = 0.2
RECOVERY_THRESHOLD = 10

# =================== STATES ===================
STANDBY, LAUNCH, COASTING, APOGEE, MAIN, RECOVERY = range(6)
state_names = ["STANDBY", "LAUNCH", "COASTING", "APOGEE", "MAIN", "RECOVERY"]

# =================== FUNCTIONS ===================
def getAltitude(pressure):
    """Convert pressure(mbar) → altitude(m)"""
    #return (T_0 / L) * (1.0 - ((pressure * 100) / P_0) ** (R * L / g))
    return ((T_0 / L) * (1.0 - ((pressure * 100) / P_0) ** (R * L / g))-925)

# ---------- KALMAN FILTER ----------
class KalmanFilter1D:
    def __init__(self, dt=0.02, meas_var=50): # meas_var=20.0: hit and trial
        # State [altitude, velocity]
        self.x = np.zeros((2, 1))
        self.P = np.eye(2) * 1000.0  # large initial uncertainty
        self.dt = dt
        
        self.F = np.array([[1, dt],
                           [0, 1]])   # State transition
        self.B = np.array([[0.5 * dt * dt],
                           [dt]])     # Control input
        self.H = np.array([[1, 0]])   # Measurement matrix
        
        # ====== Data-driven Q matrix (from dataset) ======
        # acceleration noise variance ≈ 14.47 (m/s²)²
        sigma_a2 = 14.47/2   # hit and trial or manually calculated; here gave dataset and asked gpt to calculate the var, cov
        #sigma_a2 = 0.0413
        
        self.Q = np.array([
            [0.25 * dt**4 * sigma_a2, 0.5 * dt**3 * sigma_a2],
            [0.5 * dt**3 * sigma_a2, dt**2 * sigma_a2]
        ])
        
        self.R = np.array([[meas_var]])    # Measurement noise


    #PREDICTION
    def predict(self, u=0):  # u = acceleration is set to 0 by default assuming accceleration data is too noisy
        # x = F*x + B*u
        self.x = self.F @ self.x + self.B * u
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
    

    #CORRECTION AND UPDATE
    def update(self, z):
        # z = measured altitude
        y = z - (self.H @ self.x)               # residual
        S = self.H @ self.P @ self.H.T + self.R # residual cov
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x

# ---------- FSM ----------
def flightTransition(kf_alt, kf_vel, pressure, lastRead, accl, currentState, count, launchTime, time_ms):
    ax, ay, az = accl
    totalAccl = math.sqrt(ax*ax + ay*ay + az*az)
    acclAngle = math.degrees(math.atan2(ax, ay))

    if currentState == STANDBY:
        cond = (totalAccl > LAUNCH_DETECTION or pressure < lastRead)
        count = count + 1 if cond else 0
        if count >= LAUNCH_THRESHOLD:
            currentState, count = LAUNCH, 0

    elif currentState == LAUNCH:
        count = count + 1 if acclAngle < 0 else 0
        if count >= COASTING_THRESHOLD:
            currentState, count = COASTING, 0

    elif currentState == COASTING:
        # Apogee detection using KF (velocity sign change)
        cond = (kf_vel < 0)
        count = count + 1 if cond else 0
        if count >= APOGEE_THRESHOLD:
            currentState, launchTime, count = APOGEE, time_ms, 0

    elif currentState == APOGEE:
        if kf_alt < (MAIN_ALTITUDE + ELEVATION):  #yahan par directly 1500 kyun nahi likha?
            currentState = MAIN

    elif currentState == MAIN:
        count = count + 1 if abs(pressure - lastRead) < RECOVERY_CONDITION else 0
        if count >= RECOVERY_THRESHOLD:
            currentState, count = RECOVERY, 0

    return currentState, count, launchTime, acclAngle

# =================== MAIN SIMULATION ===================
def run_fsm(input_csv, output_csv, dt=0.02):
    df = pd.read_csv(input_csv)

    time_col = "Timestamp"
    pressure_col = "Pressure"
    accx_col, accy_col, accz_col = "AccX", "AccY", "AccZ"

    currentState = STANDBY
    count = 0
    launchTime = 0
    lastRead = df[pressure_col].iloc[0]

    states, altitudes_kf, velocities_kf, accAngles = [], [], [], []
    raw_altitudes, raw_velocities, raw_accels = [], [], []

    kf = KalmanFilter1D(dt=dt)

    v_raw = 0.0
    for _, row in df.iterrows():
        t = row[time_col]
        pressure = row[pressure_col]
        ax, ay, az = row[accx_col], row[accy_col], row[accz_col]

        # Convert pressure to altitude (measurement)
        z_meas = getAltitude(pressure)

        # KF predict + update
        kf.predict(u=ay - g)   # vertical accel (remove gravity)
        kf.update(z_meas)

        kf_alt = float(kf.x[0, 0])
        kf_vel = float(kf.x[1, 0])

        # Raw velocity by integrating accel
        v_raw += (ay - g) * dt

        # FSM transition using KF
        currentState, count, launchTime, acclAngle = flightTransition(
            kf_alt, kf_vel, pressure, lastRead, (ax, ay, az), currentState, count, launchTime, t
        )

        # Append data
        states.append(currentState)
        altitudes_kf.append(kf_alt)
        velocities_kf.append(kf_vel)
        accAngles.append(acclAngle)
        raw_altitudes.append(z_meas)
        raw_velocities.append(v_raw)
        raw_accels.append(ay - g)
        
        

        lastRead = pressure

    # Append results
    df["Altitude_KF"] = altitudes_kf
    df["Velocity_KF"] = velocities_kf
    df["Altitude_raw"] = raw_altitudes
    df["Velocity_raw"] = raw_velocities
    df["Accel_raw"] = raw_accels
    df["FSM_State"] = states
    df["AccAngle_deg"] = accAngles

    df.to_csv(output_csv, index=False)
    print(f"Simulation finished. Output saved to {output_csv}")

    # =================== PLOTTING ===================
    time = df[time_col]

    plt.figure(figsize=(12, 10))

    # Altitude
    plt.subplot(4, 1, 1)
    plt.plot(time, df["Altitude_raw"], label="Raw (Pressure-derived)")
    plt.plot(time, df["Altitude_KF"], label="KF Estimated")
    plt.ylabel("Altitude (m)")
    plt.legend()
    plt.grid()

    # Velocity
    plt.subplot(4, 1, 2)
    plt.plot(time, df["Velocity_raw"], label="Raw (Accel integrated)")
    plt.plot(time, df["Velocity_KF"], label="KF Estimated")
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.grid()

    # Acceleration
    plt.subplot(4, 1, 3)
    plt.plot(time, df["Accel_raw"], label="Raw Vertical Accel")
    plt.ylabel("Acceleration (m/s²)")
    plt.legend()
    plt.grid()

    # Pressure (converted altitude)
    plt.subplot(4, 1, 4)
    plt.plot(time, df["Pressure"], label="Raw Pressure (mbar)")
    plt.ylabel("Pressure (mbar)")
    plt.legend()
    plt.grid()

    plt.xlabel("Time (ms)")
    plt.tight_layout()
    plt.show()

# =================== RUN ===================
if __name__ == "__main__":
    run_fsm("VAYUVEGA_13_JUN_25_FINAL.csv", "output_fsm4.csv")
