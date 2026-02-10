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

# Verifies function of battery condition check.
def test_battery_check(
        testMode: mode,
        currentBatteryCharge: float,
):
    # Prints relevant values.
    print(currentBatteryCharge)
    print(testMode.activeConditions["batteryCharge"])
    if testMode.check_battery(currentBatteryCharge= currentBatteryCharge):
        print("CHECK: Current battery > Battery requirement")

    # Sets battery to below condition.
    currentBattery = testMode.activeConditions["batteryCharge"] - 1.0
    print(currentBattery)
    print(testMode.activeConditions["batteryCharge"])
    if not testMode.check_battery(currentBatteryCharge= currentBatteryCharge):
        print("CHECK: Current battery < Battery requirement")

# Verifies sunlit condition check.
def test_sunlit_check(
        testMode: mode,
        currentSunlit: float,
):
    # Prints relevant values.
    print(currentSunlit)
    print(testMode.activeConditions["sunlit"])
    if testMode.check_sunlit(currentSunlit=currentSunlit):
        print("CHECK: Current sunlit > Sunlit requirement")
    if not testMode.check_sunlit(currentSunlit= 0.99):
        print("CHECK: Detects current sunlit < requirement")


# Initializes verification modes.
testModeConditions = conditionsDict()

testModeConditions["batteryCharge"] = 5.0
testModeConditions["sunlit"] = 1.0
testModeConditions["timeSinceActive"] = 10
testModeConditions["timeSinceLastActive"] = 20

modeTest = mode(
    modeName            = "test",
    powerActive         = 10.0,         # W
    activeConditions    = testModeConditions
)

# Initializes array of time values.
times           = np.arange(start= 0.0, stop= 50.0, step= 5.0)

# Initializes array of sunlit values.
sunlit          = np.ones(len(times))
sunlit[-3:]     = 0.0

print(times)

# Sets starting battery value.
batteryMax          = 45                    # Watt*h
batteryCurrent      = batteryMax/2


# Verifies runtime check.
if runtimeVerification := False:
    print("===== RUNTIME VERIFICATION =====")
    test_runtime_check(
        testMode        =modeTest,
        currentTime     =1250.0,
    )

# Verifies battery check.
if batteryVerification := False:
    print("===== BATTERY VERIFICATION =====")
    test_battery_check(
        testMode        =modeTest,
        currentBatteryCharge  =10.0
    )

# Verifies sunlit check.
if sunlitVerification := False:
    print("===== SUNLIT VERIFICATION =====")
    test_sunlit_check(
        testMode        =modeTest,
        currentSunlit  =1.0
    )

# Verifies active check.
if activeVerification := False:
    print("===== ACTIVE CHECK VERIFICATION =====")
    expectedBools = [
        False, False, False, False, True, True, True, False, False, False
    ]
    resultBools = []

    for i, time in enumerate(times):
        # Prints current battery and sunlit conditions.
        print(f"Battery= {batteryCurrent} Wh")
        print(f"Sunlit= {sunlit[i]}")

        if modeTest.check_active(
            batteryCharge= batteryCurrent,
            sunlit= sunlit[i],
            currentTime= time):

            # Appends true result if check active passes.
            resultBools.append(True)

            # Updates battery value.
            batteryCurrent += (1.0 - modeTest.powerActive) * (5.0/60**2)

        else:
            # Appends false.
            resultBools.append(False)

            # Updates battery value.
            batteryCurrent += 1.0 * (5.0/60**2)

    if np.all(resultBools == expectedBools):
        print("CHECK: Active check passed")
