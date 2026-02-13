# This file is meant to contain all the specific properties for each mode, along with the modes themselves.
# Main file should pull modes from here.
# - Active conditions using conditionsDict class.
# - Power consumption and name.

from modes import conditionsDict, mode

# ===== Example of mode active conditions initialization. =====
"""
exampleConditions = conditionsDict()

exampleConditions["batteryCharge"] = 100            # W*h
exampleConditions["sunlit"] = 1.0                   # -
exampleConditions["timeSinceActive"] = 100          # s
exampleConditions["timeSinceLastActive"] = 1000     # s
"""


# ===== Idle Mode =====
# NOTE: Active conditions don't really matter for idle since the logic switching always reverts to this mode.
# Done to keep a consistent initialization for all modes.
idleConditions = conditionsDict()

idleConditions["batteryCharge"]         = 0.0
idleConditions["sunlit"]                = 0.0
idleConditions["timeSinceActive"]       = 0.0
idleConditions["timeSinceLastActive"]   = 0.0

modeIdle = mode(
    modeName                            = "idle",
    # TODO: Add payload passive power when available.
    # Idle power active should be the sum of all passive power draws.
    # Comms + Payload
    powerActive                         = 0.1 + 0.0,           # W
    activeConditions                    = idleConditions
)

# ===== Comms Mode =====
commsConditions = conditionsDict()

commsConditions["batteryCharge"]        = 20.0
commsConditions["sunlit"]               = 1.0
# TODO: Needs better estimate for comms window duration.
# Currently 30 min window.
commsConditions["timeSinceActive"]      = 30*60.0
# Based on "1-2 comms windows per day" estimate. Uses 2.
commsConditions["timeSinceLastActive"]  = (24*60**2) / 2

modeComms = mode(
    modeName                            = "comms",
    # TODO: Add payload passive power when available.
    # Based on values taken from Bartek.
    powerActive                         = 4.8,
    activeConditions                    = commsConditions
)


# ===== Payload Mode =====
payloadConditions = conditionsDict()

payloadConditions["batteryCharge"]      = 20.0
payloadConditions["sunlit"]             = 1.0
# Based on 15-30s activation of dice payload. Uses 30.
payloadConditions["timeSinceActive"]    = 30
# Using 3/day dice payload activation.
payloadConditions["timeSinceLastActive"]= (24*60**2) / 12

modePayload = mode(
    modeName                            = "payload",
    # Based on ???
    # TODO: This desperately needs better values.
    powerActive                         = 4.8*2,            # 
    activeConditions                    = payloadConditions
)