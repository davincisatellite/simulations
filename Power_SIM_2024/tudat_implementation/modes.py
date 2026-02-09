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

### Example of mode active conditions initialization.
"""exampleConditions = conditionsDict()

exampleConditions["batteryCharge"] = 100
exampleConditions["sunlit"] = 1.0
exampleConditions["timeSinceActive"] = 100
exampleConditions["timeSinceLastActive"] = 1000"""

class mode:
    """Mode class. Defines the mode and conditions for switching to."""
    def __init__(self,
                 modeName: str,
                 powerActive: float,
                 activeConditions: conditionsDict
                 ):
        """Initializes mode.
        Args:
        - modeName: string. Name of the mode.
        - powerActive: float. Power consumption while mode is active.
        - activeConditions: conditionsDict. Dictionary of conditions for being active."""

        self.name = modeName
        self.powerActive = powerActive
        self.activeConditions = activeConditions
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

    # TODO: Problem with current implementation is two modes can be active at the same time.
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
        # TODO: This can be made into some variable that actually tells you what condition/s are failing.
        activeChecklist = [False] * 3

        activeChecklist[0] = self.check_runtime(currentTime= currentTime)
        activeChecklist[1] = self.check_sunlit(sunlit= sunlit)
        activeChecklist[2] = self.check_battery(batteryCharge= batteryCharge)

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
            return False

    def force_active(self):
        # Sets mode to active. Used in verification.
        self.isActive = True


# TODO: These are all old implementations. Delete when new ones work properly.
class active_mode: 
    """Mode class.
    Args:
    - batteryCond: battery charge condition for mode to be active.
    - sunlightCond: sunlit condition for mode to be active.
    - powerDrain: float. Power drain of mode in W. 
    """
    def __init__(self, modeName:str, batteryCond=0.0 , sunlightCond=False,
                 powerDrain=0.0, switchTo:str="safe"):
        
        self.name = modeName
        self.batteryCond = batteryCond
        self.sunlightCond = sunlightCond
        self.powerDrain = powerDrain
        self.switchTo = switchTo

    def check_run(self, batteryCharge, sunlight):
        """Checks if conditions for mode to be active are met.

        Args:
        - batteryCharge: float W*h, current battery charge. 
        - sunglight: 1-0 depending on sunglight exposure. (1 = sunlit)
        Returns:
        - modeRuns: Boolean. True if running conditions are met. 
        """
        if self.batteryCond:
            if batteryCharge > self.batteryCond:
                batteryCheck = True
            else: 
                batteryCheck = False
                # Temporary log statement. 
                print(f"{self.name} mode switched due to low power.")
        else:
            batteryCheck = True
            
        if self.sunlightCond:
            if sunlight >= self.sunlightCond:
                sunlightCheck = True
            else:
                sunlightCheck = False
                # Temporary log statement. 
                print(f"{self.name} mode switched due to eclipse.")
        else:
            sunlightCheck = True

        checkRun = batteryCheck and sunlightCheck

        return checkRun, safeKey
    

class safe_mode: 
    """safe mode class. Defines the safe mode and conditions for switching to
    other (Active) modes.
    Args:
    - batteryCond: 
    - sunlightCond: 
    - powerDrain: float. Power drain of mode in W. 
    """
    def __init__(self, modeName:str, batteryCond=0.0, sunlightCond=False,
                 powerDrain=0.0):
        
        self.name = modeName
        self.batteryCond = batteryCond
        self.sunlightCond = sunlightCond
        self.powerDrain = powerDrain

    def switch_to_active(self, batteryCharge):
        """Switches to active mode. Placeholder. """
        if batteryCharge > self.batteryCond:
            batteryCheck = False
            print(f"{self.name} mode switched due to enough power.")
        else: 
            batteryCheck = True

        return batteryCheck, activeKey

    def check_run(self, batteryCharge, sunlight):
        """Checks if conditions for mode to be active are met. Currently built
        to accomodate a series of modes to switch to.

        Args:
        - batteryCharge: float W*h, current battery charge. 
        - sunglight: 1-0 depending on sunglight exposure. (1 = sunlit)
        Returns:
        - modeRuns: Boolean. True if running conditions are met. 
        """
        
        checkRun, switchMode = self.switch_to_active(
            batteryCharge= batteryCharge)

        return checkRun, switchMode
    
            
# TODO: This needs a citation. Is it the safe mode consumption??
powerPassiveBase = 0.66           # W
powerPassivePayload = 1.32        
powerPassivePing = 0.73

safeModeDrain = powerPassiveBase + powerPassivePayload + powerPassivePing

safeMode = safe_mode(powerDrain= safeModeDrain, modeName="safeMode",
                     batteryCond= 40.0)

# NOTE: Placeholder
activeModePlaceholder = active_mode(batteryCond= 5.0, sunlightCond= 0.0, powerDrain= 5.0,
                 modeName= "activeMode", switchTo= "safe")

# Defines the dictionary of modes. 
modes = {safeKey:safeMode, activeKey:activeModePlaceholder}
            

