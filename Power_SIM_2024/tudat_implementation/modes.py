"""
Stores modes class and all desired modes.
"""

# Dictionary keys for each implemented mode. 
safeKey = 'safe'
activeKey = 'active'            # Placeholder


class conditionsDict(dict):
    """Dictionary of conditions for mode to be active. A dictionary of preset keys for use in mode
    switching conditions.
    """
    _keys = "batteryCharge sunlit timeSinceActive timeSinceLastActive".split()
    def __init__(self, valtype= int):
        for key in conditionsDict._keys:
            self[key] = valtype()
    def __setitem__(self, key, value):
        if key not in conditionsDict._keys:
            raise KeyError
        dict.__setitem__(self, key, value)


class mode:
    """Mode class. Defines the mode and conditions for switching to."""
    def __init__(self,
                 modeName: str,
                 powerActive: float,
                 activeConditions: conditionsDict,
                 iD: int,
                 ):
        """Initializes mode.
        Args:
        - modeName: string. Name of the mode.
        - powerActive: float. Power consumption while mode is active.
        - activeConditions: conditionsDict. Dictionary of conditions for being active.
        - iD: int. ID of mode."""

        self.name = modeName
        self.powerActive = powerActive
        self.activeConditions = activeConditions
        self.iD = iD

        self.isActive = False
        # Time of activation given by simulation time.
        self.timeActivated = 0.0

    def check_runtime(self,
                      currentTime):
        # Checks whether mode was previously set to active.
        if self.isActive:
            # If so, checks the active duration condition.
            # If condition > current time - time of last activation; mode can keep running.
            if self.activeConditions["timeSinceActive"] >= (currentTime - self.timeActivated):
                return True
            else:
                return False
        else:
            # If not, checks the time since last activated condition.
            # If condition < current time - time of last activation; enough time has passed for new activation.
            if self.activeConditions["timeSinceLastActive"] <= (currentTime - self.timeActivated):
                return True
            else:
                return False

    def check_sunlit(self,
                     currentSunlit):
        # Checks sunlit condition.
        # If sunlit state above condition, mode keeps running.
        if self.activeConditions["sunlit"] <= currentSunlit:
            return True
        else:
            return False

    def check_battery(self,
                      currentBatteryCharge):
        # Checks battery condition.
        # If battery charge above condition, mode keeps running.
        if self.activeConditions["batteryCharge"] <= currentBatteryCharge:
            return True
        else:
            return False

    def check_active(self,
                     batteryCharge: float,
                     sunlit: float,
                     currentTime: float):
        """Checks if conditions for mode to be active are met.
        Args:
        - conditionsCurrent: conditionsDict. Dictionary of current conditions.
        - timeStep: float. Time step of simulation.
        """

        # Initializes an array of booleans.
        activeChecklist = [False] * 3

        activeChecklist[0] = self.check_runtime(currentTime= currentTime)
        activeChecklist[1] = self.check_sunlit(currentSunlit= sunlit)
        activeChecklist[2] = self.check_battery(currentBatteryCharge= batteryCharge)

        # Checks if all elements of checklist are true.
        if all(activeChecklist):
            # Checks if mode was previously inactive.
            if not self.isActive:
                # Sets activated to true.
                self.isActive = True
                # Sets time activated to current time.
                self.timeActivated = currentTime
            return True
        else:
            # Checks if mode was previously active.
            if self.isActive:
                # Switches mode off
                self.isActive = False
                # Prompts which check caused mode switch. 
                if not activeChecklist[0]: print(f"{self.name} switched OFF. Runtime Exceeded.")
                if not activeChecklist[1]: print(f"{self.name} switched OFF. Low Sunlight.")
                if not activeChecklist[2]: print(f"{self.name} switched OFF. Low Battery.")
            return False
    def force_active(self):
        # Sets mode to active. Used in verification.
        self.isActive = True

            

