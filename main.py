import LinearRegression as LR

modeSelection = 0

print("Please Select a Mode:")
print("Available Modes: 1. Linear Regression 2. MultiLinear Regression")
modeSelection = int(input('Enter your Selection: '))

if modeSelection == 1:
    LR.linearRegression()