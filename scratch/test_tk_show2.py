import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.bar(["Maths", "OS", "CN"], [95, 88, 92], color="#2563EB")
ax.set_title("Test Window")
plt.show(block=False)
plt.pause(0.2)
print("Figure is open on screen!")
# Notice: fig is NOT closed immediately, so the window stays open on the user's desktop!
