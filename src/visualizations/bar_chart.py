import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 5))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111)
models = ["LR", "SWM", "DT", "Random Forest", "Neural Net"]
x = [1.5, 2.5, 3.5, 4.5, 5.5]
accuracy = [54,55,64,77,81]
ax.bar(x, accuracy, align='center', color="#957DAD", edgecolor="gray") 
plt.xticks(x, models)
ax.set_title('Model Accuracy Comparison') 
ax.set_ylabel('Accuracy (%)')
plt.show()