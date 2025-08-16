import LinearRegression as LR
import MultiLinearRegression as MLR

modeSelection = -1

print("Please Select a Mode:")
print("Available Modes: 1. Linear Regression 2. MultiLinear Regression")
modeSelection = int(input('Enter your Selection: '))

if modeSelection == 1:
    LR.linearRegression()
elif modeSelection == 2:
    MLR.multiLinearRegression()
else:
    print("Invalid Mode Selected. Terminating...")