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

# ===== Active Component Power Draw [W] =====
# Safety Factors
untestedSF      = 1.20
testedSF        = 1.05
# Payload components
pLDice          = 6.445 * untestedSF
pLBitflip       = 0.060 * untestedSF
pLManager       = 0.550 * testedSF
pLGnss          = 0.022 * testedSF
# On board computer components
# TODO: Missing OBC Data-board.
oBCBoard        = 0.022 * untestedSF
# EPS components
ePSModule       = 0.090 * testedSF
ePSThSensor     = 0.001 * testedSF
ePSThHeater     = 0.010 * testedSF
# ADCS components
aDCSHyperion    = 4.000 * untestedSF
# TODO: Missing a few values from hyperion atm.
aDCSSunSensor   = 0.003 * testedSF
# Communication components
commsReceiving  = 0.480 * testedSF
commsTransmitt  = 4.000 * testedSF
commsAntenna    = 0.027 * testedSF

# ===== Idle Mode =====
# NOTE: Active conditions don't really matter for idle since the logic switching always reverts to this mode.
# Done to keep a consistent initialization for all modes.
idleConditions = conditionsDict()

idleConditions["batteryCharge"]         = 0.0
idleConditions["sunlit"]                = 0.0
idleConditions["timeSinceActive"]       = 0.0
idleConditions["timeSinceLastActive"]   = 0.0
# Only contains power draws that are marked 100% in power budget.
idlePowerDrain                          = pLBitflip + pLManager + pLGnss + oBCBoard + ePSModule + ePSThSensor + \
                                          aDCSSunSensor + commsReceiving + commsAntenna

modeIdle = mode(
    modeName                            = "idle",
    # TODO: Add payload passive power when available.
    # Idle power active should be the sum of all passive power draws.
    # Comms + Payload
    powerActive                         = 0.1 + 0.0,           # W
    activeConditions                    = idleConditions,
    iD                                  = 0
)

# ===== Comms Mode =====
commsConditions = conditionsDict()

# Comms mode inactivity period per day.
commsDailyInactive                      = 1 - 2.00/100
# Comms mode num. of activations per day. Currently just a sensible amount.
commsDailyActivations                   = 6

commsConditions["batteryCharge"]        = 20.0
commsConditions["sunlit"]               = 1.0
# TODO: Needs better estimate for comms window duration.
# Currently 30 min window.
commsConditions["timeSinceActive"]      = ((24*60**2) * (1-commsDailyInactive)) / commsDailyActivations
# Based on "1-2 comms windows per day" estimate. Uses 2.
commsConditions["timeSinceLastActive"]  = ((24*60**2) * commsDailyInactive) / commsDailyActivations

commsPowerDrain                         = idlePowerDrain + commsTransmitt

modeComms = mode(
    modeName                            = "comms",
    # TODO: Add payload passive power when available.
    # Based on values taken from Bartek.
    powerActive                         = commsPowerDrain,
    activeConditions                    = commsConditions,
    iD                                  = 1
)


# ===== Payload Mode =====
payloadConditions = conditionsDict()

# Payload inactive fraction per day.
payloadDailyInactive                    = 1 - 1.04/100
# Payload daily activations. Currently just a sensible amount.
payloadDailyActivations                 = 8

payloadConditions["batteryCharge"]      = 20.0
payloadConditions["sunlit"]             = 1.0
# Based on 15-30s activation of dice payload. Uses 30.
payloadConditions["timeSinceActive"]    = ((24*60**2) * (1-payloadDailyInactive)) / payloadDailyActivations
# Interval between payload activations.
payloadConditions["timeSinceLastActive"]= ((24*60**2) * payloadDailyInactive) / payloadDailyActivations

payloadPowerDrain                       = idlePowerDrain + aDCSHyperion + pLDice


modePayload = mode(
    modeName                            = "payload",
    # Based on power budget.
    # TODO: This needs more complete values.
    powerActive                         = payloadPowerDrain,
    activeConditions                    = payloadConditions,
    iD                                  = 2
)