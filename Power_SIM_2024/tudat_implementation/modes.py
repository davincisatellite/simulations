"""
Stores modes class and all desired modes.


"""

class mode: 
    """Mode class.
    Args:
    - batteryCond: battery charge condition for mode to be active.
    - sunlightCond: sunlit condtion for mode to be active.  
    - powerDrain: float. Power drain of mode in W. 
    """
    def __init__(self, modeName:str, batteryCond=0.0 , sunlightCond=False,
                 powerDrain=0.0):
        
        self.name = modeName
        self.batteryCond = batteryCond
        self.sunlightCond = sunlightCond
        self.powerDrain = powerDrain

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

        return batteryCheck, sunlightCheck
            
safeMode = mode(powerDrain= 1.0, modeName="safeMode")
checkMode = mode(batteryCond= 100.0, sunlightCond= 1.0, powerDrain= 5.0,
                 modeName= "checkMode")

currentMode = checkMode 
currentMode.check_run(batteryCharge= 50, sunlight= 1.0)

            

