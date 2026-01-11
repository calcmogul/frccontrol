"""DC motor class instances for common FRC motors."""

import numpy as np
from scipy.signal import StateSpace


class DCMotor:
    """Models a DC motor."""

    def __init__(
        self, nominal_voltage, stall_torque, stall_current, free_current, free_speed
    ):
        """
        Holds the constants for a DC motor.

        Parameter ``nominal_voltage``:
            Voltage at which the motor constants were measured.

        Parameter ``stall_torque``:
            Current draw when stalled in Newton-meters.

        Parameter ``stall_current``:
            Current draw when stalled in Amps.

        Parameter ``free_current``:
            Current draw under no load in Amps.

        Parameter ``free_speed``:
            Angular velocity under no load in RPM.
        """
        self.nominal_voltage = nominal_voltage
        self.stall_torque = stall_torque
        self.stall_current = stall_current
        self.free_current = free_current

        # Convert from RPM to rad/s
        self.free_speed = free_speed / 60 * (2.0 * np.pi)

        # Resistance of motor
        self.R = self.nominal_voltage / self.stall_current

        # Motor velocity constant
        self.Kv = self.free_speed / (self.nominal_voltage - self.R * self.free_current)

        # Torque constant
        self.Kt = self.stall_torque / self.stall_current


# CIM
MOTOR_CIM = DCMotor(12.0, 2.42, 133.0, 2.7, 5310.0)

# MiniCIM
MOTOR_MINI_CIM = DCMotor(12.0, 1.41, 89.0, 3.0, 5840.0)

# Bag motor
MOTOR_BAG = DCMotor(12.0, 0.43, 53.0, 1.8, 13180.0)

# 775 Pro
MOTOR_775PRO = DCMotor(12.0, 0.71, 134.0, 0.7, 18730.0)

# Andymark RS 775-125
MOTOR_AM_RS775_125 = DCMotor(12.0, 0.28, 18.0, 1.6, 5800.0)

# Banebots RS 775
MOTOR_BB_RS775 = DCMotor(12.0, 0.72, 97.0, 2.7, 13050.0)

# Andymark 9015
MOTOR_AM_9015 = DCMotor(12.0, 0.36, 71.0, 3.7, 14270.0)

# Banebots RS 550
MOTOR_BB_RS550 = DCMotor(12.0, 0.38, 84.0, 0.4, 19000.0)

# NEO
MOTOR_NEO = DCMotor(12.0, 2.6, 105.0, 1.8, 5676.0)

# NEO 550
MOTOR_NEO_550 = DCMotor(12.0, 0.97, 100.0, 1.4, 11000.0)

# Falcon 500
MOTOR_FALCON_500 = DCMotor(12.0, 4.69, 257.0, 1.5, 6380.0)


def gearbox(motor, num_motors):
    """Returns a DCMotor with the same characteristics as the specified number
    of motors in a gearbox.
    """
    return DCMotor(
        motor.nominal_voltage,
        motor.stall_torque * num_motors,
        motor.stall_current * num_motors,
        motor.free_current * num_motors,
        motor.free_speed / (2.0 * np.pi) * 60,
    )


def elevator(motor, num_motors, m, r, G):
    """
    Returns the state-space model for an elevator.

    States: [[position], [velocity]]
    Inputs: [[voltage]]
    Outputs: [[position]]

    Parameter ``motor``:
        Instance of DCMotor.

    Parameter ``num_motors``:
        Number of motors driving the mechanism.

    Parameter ``m``:
        Carriage mass in kg.

    Parameter ``r``:
        Pulley radius in meters.

    Parameter ``G``:
        Gear ratio from motor to carriage.

    Returns:
        StateSpace instance containing continuous model.
    """
    motor = gearbox(motor, num_motors)

    A = np.array([[0, 1], [0, -(G**2) * motor.Kt / (motor.R * r**2 * m * motor.Kv)]])
    B = np.array([[0], [G * motor.Kt / (motor.R * r * m)]])
    C = np.array([[1, 0]])
    D = np.array([[0]])

    return StateSpace(A, B, C, D)


def flywheel(motor, num_motors, J, G):
    """
    Returns the state-space model for a flywheel.

    States: [[angular velocity]]
    Inputs: [[voltage]]
    Outputs: [[angular velocity]]

    Parameter ``motor``:
        Instance of DCMotor.

    Parameter ``num_motors``:
        Number of motors driving the mechanism.

    Parameter ``J``:
        Flywheel moment of inertia in kg-m².

    Parameter ``G``:
        Gear ratio from motor to flywheel.

    Returns:
        StateSpace instance containing continuous model.
    """
    motor = gearbox(motor, num_motors)

    A = np.array([[-(G**2) * motor.Kt / (motor.Kv * motor.R * J)]])
    B = np.array([[G * motor.Kt / (motor.R * J)]])
    C = np.array([[1]])
    D = np.array([[0]])

    return StateSpace(A, B, C, D)


def differential_drive(motor, num_motors, m, r, rb, J, Gl, Gr):
    """
    Returns the state-space model for a differential drive.

    States: [[left velocity], [right velocity]]
    Inputs: [[left voltage], [right voltage]]
    Outputs: [[left velocity], [right velocity]]

    Parameter ``motor``:
        Instance of DCMotor.

    Parameter ``num_motors``:
        Number of motors driving the mechanism.

    Parameter ``m``:
        Mass of robot in kg.

    Parameter ``r``:
        Radius of wheels in meters.

    Parameter ``rb``:
        Radius of robot in meters.

    Parameter ``J``:
        Moment of inertia of the differential drive in kg-m².

    Parameter ``Gl``:
        Gear ratio of left side of the differential drive.

    Parameter ``Gr``:
        Gear ratio of right side of the differential drive.

    Returns:
        StateSpace instance containing continuous model.
    """
    motor = gearbox(motor, num_motors)

    C1 = -(Gl**2) * motor.Kt / (motor.Kv * motor.R * r**2)
    C2 = Gl * motor.Kt / (motor.R * r)
    C3 = -(Gr**2) * motor.Kt / (motor.Kv * motor.R * r**2)
    C4 = Gr * motor.Kt / (motor.R * r)
    A = np.array(
        [
            [(1 / m + rb**2 / J) * C1, (1 / m - rb**2 / J) * C3],
            [(1 / m - rb**2 / J) * C1, (1 / m + rb**2 / J) * C3],
        ]
    )
    B = np.array(
        [
            [(1 / m + rb**2 / J) * C2, (1 / m - rb**2 / J) * C4],
            [(1 / m - rb**2 / J) * C2, (1 / m + rb**2 / J) * C4],
        ]
    )
    C = np.eye(2)
    D = np.zeros((2, 2))

    return StateSpace(A, B, C, D)


def single_jointed_arm(motor, num_motors, J, G):
    """
    Returns the state-space model for a single-jointed arm.

    States: [[angle, angular velocity]]
    Inputs: [[voltage]]
    Outputs: [[angular velocity]]

    Parameter ``motor``:
        Instance of DCMotor.

    Parameter ``num_motors``:
        Number of motors driving the mechanism.

    Parameter ``J``:
        Arm moment of inertia in kg-m².

    Parameter ``G``:
        Gear ratio from motor to arm.

    Returns:
        StateSpace instance containing continuous model.
    """
    motor = gearbox(motor, num_motors)

    A = np.array([[0, 1], [0, -(G**2) * motor.Kt / (motor.Kv * motor.R * J)]])
    B = np.array([[0], [G * motor.Kt / (motor.R * J)]])
    C = np.array([[1, 0]])
    D = np.array([[0]])

    return StateSpace(A, B, C, D)
