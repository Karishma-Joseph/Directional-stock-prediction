import matplotlib.pyplot as plt

true_stock = [1,2,3,4,5]
pred_stock = [1,3,3,4,6]
time_interval = ["2019/01/31 01:00 PM", "2019/01/31 01:15 PM","2019/01/31 01:30 PM","2019/01/31 01:45 PM","2019/01/31 02:00 PM"]
x = [1,2,3,4,5]

fig = plt.figure(figsize=(15, 5))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111)
ax.plot(x, true_stock, label="True", color='blue')
ax.plot(x, pred_stock, label="Pred", color='red')

plt.xticks(x, time_interval)

ax.set_title('Stock Market')
ax.legend()

plt.xlabel("Time")
plt.ylabel("EMA")

plt.savefig('amazon.jpg')