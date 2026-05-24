import struct
import time
from typing import Any

import can
import numpy as np
import pytest

from i2rt.motor_drivers import dm_driver
from i2rt.motor_drivers.dm_driver import ControlMode, DMChainCanInterface, DMSingleMotorCanInterface, MotorRegister
from i2rt.motor_drivers.utils import FeedbackFrameInfo, MotorInfo, MotorType
from i2rt.robots.motor_chain_robot import MotorChainRobot


def _feedback() -> FeedbackFrameInfo:
    return FeedbackFrameInfo(
        id=4,
        error_code="0x1",
        error_message="normal",
        position=0.0,
        velocity=0.0,
        torque=0.0,
        temperature_mos=30.0,
        temperature_rotor=30.0,
    )


def test_pos_vel_control_packet_uses_profile_frame() -> None:
    iface = DMSingleMotorCanInterface.__new__(DMSingleMotorCanInterface)
    iface.control_mode = ControlMode.POS_VEL
    iface.cmd_idoffset = ControlMode.get_id_offset(ControlMode.POS_VEL)
    sent: dict[str, Any] = {}

    def send(
        frame_id: int,
        motor_id: int,
        data: bytearray,
        max_retry: int = 15,
    ) -> can.Message:
        sent["frame_id"] = frame_id
        sent["motor_id"] = motor_id
        sent["data"] = bytes(data)
        sent["max_retry"] = max_retry
        return can.Message(arbitration_id=motor_id + 16, data=bytearray(8), is_extended_id=False)

    iface._send_message_get_response = send
    iface.parse_recv_message = lambda message, motor_type: _feedback()

    iface.set_control(0x04, MotorType.DM4340, pos=1.25, vel=-0.5, kp=80.0, kd=5.0, torque=3.0)

    assert sent["frame_id"] == 0x104
    assert sent["motor_id"] == 0x04
    assert sent["data"] == struct.pack("<ff", 1.25, 0.5)
    assert sent["max_retry"] == 15


def test_vel_control_packet_uses_velocity_frame() -> None:
    iface = DMSingleMotorCanInterface.__new__(DMSingleMotorCanInterface)
    iface.control_mode = ControlMode.VEL
    iface.cmd_idoffset = ControlMode.get_id_offset(ControlMode.VEL)
    sent: dict[str, Any] = {}

    def send(
        frame_id: int,
        motor_id: int,
        data: bytearray,
        max_retry: int = 15,
    ) -> can.Message:
        sent["frame_id"] = frame_id
        sent["motor_id"] = motor_id
        sent["data"] = bytes(data)
        sent["max_retry"] = max_retry
        return can.Message(arbitration_id=motor_id + 16, data=bytearray(8), is_extended_id=False)

    iface._send_message_get_response = send
    iface.parse_recv_message = lambda message, motor_type: _feedback()

    iface.set_control(0x04, MotorType.DM4340, pos=1.25, vel=-0.5, kp=80.0, kd=5.0, torque=3.0)

    assert sent["frame_id"] == 0x204
    assert sent["motor_id"] == 0x04
    assert sent["data"] == struct.pack("<f", -0.5) + bytes(4)
    assert sent["max_retry"] == 15


def test_motor_register_writes_match_damiao_format() -> None:
    iface = DMSingleMotorCanInterface.__new__(DMSingleMotorCanInterface)
    messages: list[can.Message] = []
    ack_reads = 0

    class Bus:
        pending_ack = False

        def send(self, message: can.Message) -> None:
            self.pending_ack = True
            messages.append(message)

        def recv(self, timeout: float = 0.001) -> can.Message | None:
            nonlocal ack_reads
            if not self.pending_ack:
                return None
            self.pending_ack = False
            ack_reads += 1
            return can.Message(
                arbitration_id=0x7FF,
                data=bytearray(messages[-1].data),
                is_extended_id=False,
            )

    iface.bus = Bus()
    iface.use_buffered_reader = False

    iface.switch_control_mode(0x04, ControlMode.POS_VEL)
    iface.set_motion_profile(0x04, max_speed=0.5, acceleration=1.0, deceleration=1.0)

    assert [msg.arbitration_id for msg in messages] == [0x7FF, 0x7FF, 0x7FF, 0x7FF]
    assert ack_reads == 4
    assert bytes(messages[0].data) == struct.pack("<HBBI", 0x04, 0x55, MotorRegister.CTRL_MODE, 2)
    assert bytes(messages[1].data) == struct.pack("<HBBf", 0x04, 0x55, MotorRegister.ACC, 1.0)
    assert bytes(messages[2].data) == struct.pack("<HBBf", 0x04, 0x55, MotorRegister.DEC, -1.0)
    assert bytes(messages[3].data) == struct.pack("<HBBf", 0x04, 0x55, MotorRegister.MAX_SPD, 0.5)


def test_vel_control_mode_switch_writes_damiao_mode_value() -> None:
    iface = DMSingleMotorCanInterface.__new__(DMSingleMotorCanInterface)
    messages: list[can.Message] = []

    class Bus:
        pending_ack = False

        def send(self, message: can.Message) -> None:
            self.pending_ack = True
            messages.append(message)

        def recv(self, timeout: float = 0.001) -> can.Message | None:
            if not self.pending_ack:
                return None
            self.pending_ack = False
            return can.Message(
                arbitration_id=0x7FF,
                data=bytearray(messages[-1].data),
                is_extended_id=False,
            )

    iface.bus = Bus()
    iface.use_buffered_reader = False

    iface.switch_control_mode(0x04, ControlMode.VEL)

    assert [msg.arbitration_id for msg in messages] == [0x7FF]
    assert bytes(messages[0].data) == struct.pack("<HBBI", 0x04, 0x55, MotorRegister.CTRL_MODE, 3)


def test_motor_register_write_rejects_stale_ack() -> None:
    iface = DMSingleMotorCanInterface.__new__(DMSingleMotorCanInterface)
    messages: list[can.Message] = []

    class Bus:
        def send(self, message: can.Message) -> None:
            messages.append(message)

        def recv(self, timeout: float = 0.001) -> can.Message:
            return can.Message(
                arbitration_id=0x7FF,
                data=bytearray([0x05, 0x00, 0x55, MotorRegister.CTRL_MODE, 0, 0, 0, 0]),
                is_extended_id=False,
            )

    iface.bus = Bus()
    iface.use_buffered_reader = False

    with pytest.raises(AssertionError):
        iface.switch_control_mode(0x04, ControlMode.POS_VEL)

    assert len(messages) == 5


def test_motor_chain_rejects_non_mit_startup_mode() -> None:
    with pytest.raises(ValueError, match="must start in MIT"):
        DMChainCanInterface(
            motor_list=[(0x01, MotorType.DM4310)],
            motor_offset=np.array([0.0]),
            motor_direction=np.array([1.0]),
            control_mode=ControlMode.POS_VEL,
            start_thread=False,
        )


def test_motor_chain_syncs_hardware_mit_mode_on_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    interfaces: list[Any] = []

    class FakeSingleMotorInterface:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.mode_writes: list[tuple[int, str]] = []
            interfaces.append(self)

        def _drain_bus(self, timeout_s: float) -> int:
            return 0

        def motor_on(self, motor_id: int, motor_type: str) -> FeedbackFrameInfo:
            return _feedback()

        def switch_control_mode(self, motor_id: int, control_mode: str) -> None:
            self.mode_writes.append((motor_id, control_mode))

    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", FakeSingleMotorInterface)

    chain = DMChainCanInterface(
        motor_list=[(0x04, MotorType.DM4340)],
        motor_offset=np.array([0.0]),
        motor_direction=np.array([1.0]),
        start_thread=False,
    )

    assert interfaces[0].mode_writes == [(0x04, ControlMode.MIT)]
    assert chain.get_control_mode() == ControlMode.MIT


class FakeMotorChain:
    def __init__(self) -> None:
        self.mode = ControlMode.MIT
        self.motor_max_speed: float | None = None
        self.profile_acceleration: tuple[float, float] | None = None
        self.motion_profile: tuple[float, float, float] | None = None
        self.last_command: dict[str, np.ndarray | None] | None = None
        self.running = True

    def __len__(self) -> int:
        return 2

    def read_states(self) -> list[MotorInfo]:
        now = time.time()
        return [
            MotorInfo(id=1, error_code="0x1", pos=0.1, vel=0.0, eff=0.0, timestamp=now),
            MotorInfo(id=2, error_code="0x1", pos=-0.2, vel=0.0, eff=0.0, timestamp=now),
        ]

    def set_commands(
        self,
        torques: np.ndarray,
        pos: np.ndarray | None = None,
        vel: np.ndarray | None = None,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
    ) -> list[MotorInfo]:
        self.last_command = {
            "torques": torques.copy(),
            "pos": None if pos is None else pos.copy(),
            "vel": None if vel is None else vel.copy(),
            "kp": None if kp is None else kp.copy(),
            "kd": None if kd is None else kd.copy(),
        }
        return self.read_states()

    def set_control_mode(self, control_mode: str) -> None:
        self.mode = control_mode

    def get_control_mode(self) -> str:
        return self.mode

    def set_motor_max_speed(self, max_speed: float) -> None:
        self.motor_max_speed = max_speed

    def set_profile_acceleration(self, acceleration: float, deceleration: float) -> None:
        self.profile_acceleration = (acceleration, deceleration)

    def set_motion_profile(self, max_speed: float, acceleration: float, deceleration: float) -> None:
        self.motion_profile = (max_speed, acceleration, deceleration)

    def close(self) -> None:
        self.running = False


def test_robot_pos_vel_mode_uses_profile_velocity_in_position_commands() -> None:
    chain = FakeMotorChain()
    robot = MotorChainRobot(
        motor_chain=chain,
        xml_path=None,
        use_gravity_comp=False,
        kp=[1.0, 1.0],
        kd=[0.1, 0.1],
        joint_limits=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        zero_gravity_mode=False,
        motor_max_speed=0.4,
        profile_acceleration=0.8,
        profile_deceleration=0.7,
    )

    try:
        robot.set_motor_control_mode(ControlMode.POS_VEL)
        robot.command_joint_pos(np.array([0.3, -0.4]))

        assert chain.mode == ControlMode.POS_VEL
        assert chain.motion_profile == (0.4, 0.8, 0.7)
        with robot._command_lock:
            np.testing.assert_allclose(robot._commands.pos, [0.3, -0.4])
            np.testing.assert_allclose(robot._commands.vel, [0.4, 0.4])
    finally:
        robot.close()


def test_robot_vel_mode_accepts_velocity_only_joint_state() -> None:
    chain = FakeMotorChain()
    robot = MotorChainRobot(
        motor_chain=chain,
        xml_path=None,
        use_gravity_comp=False,
        kp=[1.0, 1.0],
        kd=[0.1, 0.1],
        joint_limits=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        zero_gravity_mode=False,
        motor_max_speed=0.4,
        profile_acceleration=0.8,
    )

    try:
        robot.set_motor_control_mode(ControlMode.VEL)
        robot.command_joint_state({"vel": np.array([0.6, -0.7])})

        assert chain.mode == ControlMode.VEL
        assert chain.motion_profile is None
        with robot._command_lock:
            np.testing.assert_allclose(robot._commands.pos, [0.0, 0.0])
            np.testing.assert_allclose(robot._commands.vel, [0.6, -0.7])
            np.testing.assert_allclose(robot._commands.kp, [0.0, 0.0])
            np.testing.assert_allclose(robot._commands.kd, [0.0, 0.0])
    finally:
        robot.close()


def test_robot_rejects_non_positive_motor_max_speed() -> None:
    chain = FakeMotorChain()
    with pytest.raises(ValueError):
        MotorChainRobot(
            motor_chain=chain,
            xml_path=None,
            use_gravity_comp=False,
            joint_limits=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
            motor_max_speed=0.0,
        )


def test_robot_updates_motor_max_speed_without_changing_pos_vel_command_speed() -> None:
    chain = FakeMotorChain()
    robot = MotorChainRobot(
        motor_chain=chain,
        xml_path=None,
        use_gravity_comp=False,
        kp=[1.0, 1.0],
        kd=[0.1, 0.1],
        joint_limits=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        zero_gravity_mode=False,
        motor_max_speed=0.4,
        position_command_max_velocity=0.2,
    )

    try:
        robot.set_motor_max_speed(0.8)
        robot.set_motor_control_mode(ControlMode.POS_VEL)
        robot.command_joint_pos(np.array([0.3, -0.4]))

        assert chain.motor_max_speed == 0.8
        with robot._command_lock:
            np.testing.assert_allclose(robot._commands.vel, [0.2, 0.2])
    finally:
        robot.close()


def test_robot_motion_profile_does_not_change_pos_vel_command_speed() -> None:
    chain = FakeMotorChain()
    robot = MotorChainRobot(
        motor_chain=chain,
        xml_path=None,
        use_gravity_comp=False,
        kp=[1.0, 1.0],
        kd=[0.1, 0.1],
        joint_limits=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        zero_gravity_mode=False,
        motor_max_speed=0.4,
        position_command_max_velocity=0.2,
    )

    try:
        robot.set_motion_profile(0.8, 1.2, 1.4)
        assert chain.motion_profile == (0.8, 1.2, 1.4)
        assert robot.get_robot_info()["position_command_max_velocity"] == 0.2
    finally:
        robot.close()
