"""
Stores modes class and all desired modes.
"""

# Dictionary keys for each implemented mode. 
safeKey = 'safe'
activeKey = 'active'            # Placeholder


class conditionsDict(dict):
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
                 activeConditions: conditionsDict
                 ):
        """Initializes mode.
        Args:
        - modeName: string. Name of the mode.
        - powerActive: float. Active power of mode in W.
        - powerPassive: float. Passive power of mode in W.
        - activeConditions: conditionsDict. Dictionary of conditions for being active."""

        self.name = modeName
        self.powerActive = powerActive
        self.activeConditions = activeConditions
        self.isActive = False
        self.timeSinceActive = 0.0
        self.timeSinceLastActive = 0.0

    # TODO: Problem with current implementation is two modes can be active at the same time.
    def check_active(self,
                     conditionsCurrent: conditionsDict,
                     timeStep: float):
        """Checks if conditions for mode to be active are met.
        Args:
        - conditionsCurrent: conditionsDict. Dictionary of current conditions.
        - timeStep: float. Time step of simulation.
        """
        # Initializes an array of booleans.
        # TODO: This can be made into some variable that actually tells you what condition/s are failing.
        activeChecklist = [False] * len(self.activeConditions)
        i = 0

        # loops through conditions.
        for key, value in self.activeConditions:
            if key == "timeSinceActive":
                if conditionsCurrent[key] < value:
                    activeChecklist[i] = True
            else:
                if conditionsCurrent[key] >= value:
                    activeChecklist[i] = True
            i =+ 1

        # Checks if all elements of checklist are true.
        if all(activeChecklist):
            # Checks if mode was previously not active.
            if not self.isActive:
                self.isActive = True
                self.timeSinceActive = 0.0
            else:
                self.timeSinceActive += timeStep
            return True
        else:
            # Checks if mode was previously active.
            if self.isActive:
                # Switches mode off and resets counters.
                self.isActive = False
                self.timeSinceActive = 0.0
                self.timeSinceLastActive = 0.0
            else:
                # Increases time since last active count.
                self.timeSinceLastActive += timeStep
            return False


payloadConditions = conditionsDict()

payloadConditions["batteryCharge"] = 100
payloadConditions["sunlit"] = 1.0
payloadConditions["timeSinceActive"] = 100
payloadConditions["timeSinceLastActive"] = 1000


modePayload = mode(
    modeName= "payload",
    powerActive= 10.0,
    activeConditions= payloadConditions
)

currentConditions = conditionsDict()

currentConditions["batteryCharge"] = 1000
currentConditions["sunlit"] = 1.0
currentConditions["timeSinceActive"] = 100
currentConditions["timeSinceLastActive"] = 1000



modePayload.check_active(
    conditionsCurrent= ,
    timeStep= 10.0
)








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
            

