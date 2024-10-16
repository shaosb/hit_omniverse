# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(2) control."""

import numpy as np
import weakref
from collections.abc import Callable
import os
import carb
import omni

from omni.isaac.lab.devices.device_base import DeviceBase

from hit_omniverse.utils.helper import setup_config

config = setup_config(os.environ.get("CONFIG"))

class Se2Keyboard(DeviceBase):
    r"""A keyboard controller for sending SE(2) commands as velocity commands.

    This class is designed to provide a keyboard controller for mobile base (such as quadrupeds).
    It uses the Omniverse keyboard interface to listen to keyboard events and map them to robot's
    task-space commands.

    The command comprises of the base linear and angular velocity: :math:`(v_x, v_y, \omega_z)`.

    Key bindings:
        ====================== ========================= ========================
        Command                Key (+ve axis)            Key (-ve axis)
        ====================== ========================= ========================
        Move along x-axis      Numpad 8 / Arrow Up       Numpad 2 / Arrow Down
        Move along y-axis      Numpad 4 / Arrow Right    Numpad 6 / Arrow Left
        Rotate along z-axis    Numpad 7 / X              Numpad 9 / Y
        ====================== ========================= ========================

    .. seealso::

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/python/carb.html#carb.input.Keyboard>`__.

    """

    def __init__(self, v_x_sensitivity: float = 0.8, v_y_sensitivity: float = 0.4, omega_z_sensitivity: float = 1.0):
        """Initialize the keyboard layer.

        Args:
            v_x_sensitivity: Magnitude of linear velocity along x-direction scaling. Defaults to 0.8.
            v_y_sensitivity: Magnitude of linear velocity along y-direction scaling. Defaults to 0.4.
            omega_z_sensitivity: Magnitude of angular velocity along z-direction scaling. Defaults to 1.0.
        """
        # store inputs
        self.v_x_sensitivity = v_x_sensitivity
        self.v_y_sensitivity = v_y_sensitivity
        self.omega_z_sensitivity = omega_z_sensitivity
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # command buffers
        self._base_command = np.zeros(4)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for SE(2): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tReset all commands: L\n"
        msg += "\tMove forward: Arrow Up\n"
        msg += "\tMove backward: Arrow Down\n"
        msg += "\tMove right: Arrow Right\n"
        msg += "\tMove left: Arrow Left\n"
        msg += "\tMove Up: 1\n"
        msg += "\tMove Down: 2\n"
        msg += "\tYaw positively (along +z-axis): Z\n"
        msg += "\tYaw positively (along -z-axis): X\n"
        msg += "\t60-run_HIT: m\n"
        msg += "\tslope_lone: h\n"
        msg += "\tsquat_walk: j\n"
        msg += "\tstair_full: k\n"
        msg += "\thit_save_people: i\n"
        msg += "\tforsquat_down: v\n"
        msg += "\tsquat_with_people: n\n"
        msg += "\tforsquat_up: b\n"
        msg += "\tsave and quit: t"
        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        self._base_command.fill(0.0)

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/python/carb.html?highlight=keyboardeventtype#carb.input.KeyboardInput>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> np.ndarray:
        """Provides the result from keyboard event state.

        Returns:
            3D array containing the linear (x,y) and angular velocity (z).
        """
        return self._base_command

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/docs/python/carb.html?highlight=keyboardeventtype#carb.input.KeyboardInput
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "L":
                self.reset()
            elif event.input.name in self._INPUT_KEY_MAPPING:
                self._base_command += self._INPUT_KEY_MAPPING[event.input.name]
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._INPUT_KEY_MAPPING:
                self._base_command -= self._INPUT_KEY_MAPPING[event.input.name]
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            #up and down
            "1": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0]) * self.v_x_sensitivity,
            "2": np.asarray([0.0, 0.0, 0.0, -1.0, 0.0]) * self.v_x_sensitivity,
            # forward command 1.5
            "UP": np.asarray([1., 0.0, 0.0, 0.0, 0.0]) * self.v_x_sensitivity,
            # back command -1.5
            "DOWN": np.asarray([-1., 0.0, 0.0, 0.0, 0.0]) * self.v_x_sensitivity,
            # right command 1.5
            "LEFT": np.asarray([0.0, 1., 0.0, 0.0, 0.0]) * self.v_y_sensitivity,
            # left command -1.5
            "RIGHT": np.asarray([0.0, -1., 0.0, 0.0, 0.0]) * self.v_y_sensitivity,
            # yaw command (positive) 1.0
            "X": np.asarray([0.0, 0.0, 1., 0.0, 0.0]) * self.omega_z_sensitivity,
            # yaw command (negative) -1.0
            "Z": np.asarray([0.0, 0.0, -1.0, 0.0, 0.0]) * self.omega_z_sensitivity,
            # 30-run_HIT
            "M": np.asarray([0.0, 0.0, 0.0, 0.0, config["GAIT"]["30-run_HIT"]]),
            # slope_lone
            "H": np.asarray([0.0, 0.0, 0.0, 0.0, config["GAIT"]["slope_lone"]]),
            # squat_walk
            "J": np.asarray([0.0, 0.0, 0.0, 0.0, config["GAIT"]["squat_walk"]]),
            # stair_full
            "K": np.asarray([0.0, 0.0, 0.0, 0.0, config["GAIT"]["stair_full"]]),
            # hit_save_people
            "I": np.asarray([0.0, 0.0, 0.0, 0.0, config["GAIT"]["hit_save_people"]]),
            # forsquat_down
            "V": np.asarray([0.0, 0.0, 0.0, 0.0, config["GAIT"]["forsquat_down"]]),
            # forsquat_up
            "B": np.asarray([0.0, 0.0, 0.0, 0.0, config["GAIT"]["forsquat_up"]]),
            # squat_with_people
            "N": np.asarray([0.0, 0.0, 0.0, 0.0, config["GAIT"]["squat_with_people"]]),
            # save and quid
            "T": np.asarray([0.0, 0.0, 0.0, 0.0, config["GAIT"]["save_and_quit"]]),
        }
