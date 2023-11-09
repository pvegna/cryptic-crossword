import matplotlib.pyplot as plt

loss_data = []
with open ('should_work-10000.txt', 'r') as loss_file:
    for line in loss_file:
        loss_data.append(float(line.strip()))

print(loss_data)
plt.plot(range(len(loss_data)), loss_data)
plt.xlabel("Training steps")
plt.ylabel("Loss")
plt.show()