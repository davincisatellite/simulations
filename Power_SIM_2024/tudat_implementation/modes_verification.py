from modes import *

import numpy as np


# Verifies function of time active condition check
def test_runtime_check(
        testMode: mode,
        currentTime: float,
):

    # Mode is currently inactive.

    print(currentTime - testMode.timeActivated)
    print(testMode.activeConditions["timeSinceLastActive"])
    # Checks time since last activation condition check.
    if testMode.check_runtime(
        currentTime= currentTime):
        print("CHECK: Time since last active < time activated - current.")
    # Sets mode activation time to current.
    testMode.timeActivated = currentTime
    # Prints relevant values.
    print(currentTime - testMode.timeActivated)
    print(testMode.activeConditions["timeSinceLastActive"])
    # Checks detection of insufficient spacing between activations.
    if not testMode.check_runtime(
        currentTime= currentTime):
        print("CHECK: Time since last active > time activated - current.")


    # Resets mode activation time.
    testMode.timeActivated = 0.0
    # Forces mode to active.
    testMode.force_active()

    print(currentTime - testMode.timeActivated)
    print(testMode.activeConditions["timeSinceActive"])
    if not testMode.check_runtime(
        currentTime= currentTime):
        # Checks detection of exceeding active time condition.
        print("CHECK: Time since active > time activated - current.")
    # Sets mode activation time to current.
    testMode.timeActivated = currentTime
    # Prints relevant values.
    print(currentTime - testMode.timeActivated)
    print(testMode.activeConditions["timeSinceActive"])
    if testMode.check_runtime(currentTime= currentTime):
        print("CHECK: Time since active < time activated - current.")


# Initializes verification modes.
testModeConditions = conditionsDict()

testModeConditions["batteryCharge"] = 100
testModeConditions["sunlit"] = 1.0
testModeConditions["timeSinceActive"] = 100
testModeConditions["timeSinceLastActive"] = 1000

modeTest = mode(
    modeName            = "test",
    powerActive         = 10.0,         # W
    activeConditions    = testModeConditions
)

# Initializes array of time values.
times           = np.arange(start= 0.0, stop= 5000, step= 5.0)

# Initializes array of sunlit values.
sunlit          = np.ones(len(times))

# Sets starting battery value.
batteryMax      = 45                    # Watt*h
batteryStart    = batteryMax/2

# Verifies runtime check.
print("===== RUNTIME VERIFICATION =====")
test_runtime_check(
    testMode        =modeTest,
    currentTime     =1250.0,
)

print("===== BATTERY VERIFICATION =====")

print("===== SUNLIT VERIFICATION =====")

print("===== OVERALL CHECK VERIFICATION =====")