a=1
b=int(input())

if a>b:
    print("a greater then b")
elif a<b:
    print("a lesser then b")
else:
    print("a uis equal to b")

c = int(input())

if a > b and a > c:
    print("a is greatest")
elif b > a and b > a:
    print("b is greatest")
else:
    print("c is greatest")



x = 5
if x > 10:
    print("x is greater than 10")
else:
    print("x is less than or equal to 10")

# If-Elif-Else statement
y = 15
if y < 10:
    print("y is less than 10")
elif y == 10:
    print("y is equal to 10")
else:
    print("y is greater than 10")

# Nested If statement
z = 20
if z > 10:
    if z % 2 == 0:
        print("z is even and greater than 10")
    else:
        print("z is odd and greater than 10")

age = 25
status = "adult" if age >= 18 else "minor"
print(status)


message = "Hello, world!" if True else "Goodbye, world!"
print(message)


x = 5
y = 3
if x > 2 and y < 5:
    print("Both conditions are true")

x = 5
y = 7
if x > 2 or y < 5:
    print("At least one condition is true")

x = 5
if not x == 3:
    print("x is not equal to 3")


fruits = ["apple", "banana", "cherry"]
if "banana" in fruits:
    print("Banana is in the list")

# Not In operator
fruits = ["apple", "banana", "cherry"]
if "grape" not in fruits:
    print("Grape is not in the list")

# Is operator
x = [1, 2, 3]
y = [1, 2, 3]
if x is y:
    print("x and y are the same object")
else:
    print("x and y are different objects")

# Is Not operator
x = [1, 2, 3]
y = [1, 2, 3]
if x is not y:
    print("x and y are different objects")

x = 5
y = 3
z = 7
if 2 < x < 10:
    print("x is between 2 and 10")
if x < y < z:
    print("x is less than y, which is less than z")

