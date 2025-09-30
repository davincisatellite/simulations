"""
Stores modes class and all desired modes.
"""

# Dictionary keys for each implemented mode. 
safeKey = 'safe'
activeKey = 'active'            # Placeholder


class active_mode: 
    """Mode class.
    Args:
    - batteryCond: battery charge condition for mode to be active.
    - sunlightCond: sunlit condtion for mode to be active.  
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
    def __init__(self, modeName:str, batteryCond=0.0 , sunlightCond=False,
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
            

