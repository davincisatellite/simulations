# 2025 Power Simulations
This folder contains the battery and power average simulations produced during the 2025-26 
academic year. This readme serves as a user guide for modifying the parameters in order to produce desired conditions. 

## Power Average Simulations

WIP

## Battery Simulations
Battery simulation code can be accessed in the power_sim, modes and modes_conditions python files. 
- **power_sim**: Contains the battery_sim function,  the loop which runs through the desired timeframe of the simulation. 
This loop defaults to the idle mode and checks whether the conditions given allow to switch to the comms or payload mode.
If so, it remains in one of these modes until the run conditions are no longer met. 
- **modes**: Contains the definition of the modes class. This class is used to create any desired operational mode, along
with the conditions which are required to run it. Additionally, it contains a conditionsDict class made to ensure the initialization
of the required conditions for operation is always done the same way. 
- **modes_conditions**: Contains the definition of the specific mode objects. Here is where any new modes, or alterations to
the currently existing one's operating parameters should be implemented. An example is given in the file of how the mode objects
are initialized. 

### Modes Class
The modes class contains the values (conditions) for the specific mode to stay active, as well as check functions that compare
those values with simulation "environmental" values. The _check_active_ function combines the other sub-functions into one. It then
sets the current state of the mode as active if all conditions are true, as well as returning True for use within
the power_sim loop. Additionally, the function returns a message indicating which mode, and why, the simulation was forced
to switch off. 

### Modes Conditions
As said, an example is present in the code that shows how to initialize a specific modes object. Additionally, the present modes
can be used as extra examples. This bit only serves to clarify some language, used in the condition keys, that may be unclear.
- **condition["timeSinceActive"]**: Denotes the maximum time the mode should stay active. 
- **condition["timeSinceLastActive"]**: Denotes the minimum time between node activations, or activation frequency. 
- **condition["sunlit"]**: Denotes how illuminated the spacecraft has to be for the mode to be active. Between 0 (fully eclipsed) and 1 (Full sunlight).
- **condition["batteryCharge"]**: Denotes the minimum battery charge required for the mode to be active. Currently given in W*h and has no relation to maximum battery capacity. 