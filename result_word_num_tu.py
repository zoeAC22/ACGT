import matplotlib.pyplot as plt
#条形图的绘制
labels = ['1', '2', '3', '4', '5','6', '7', '8', '9', '10']

Rouge_1 = [35.87,36.14,36.48,36.83,37.81,38.00,39.20,39.96,40.10,39.34]
Rouge_2 = [17.66,18.11,18.98,17.58,19.25,17.26,16.62,16.68,19.88,18.55]
Rouge_l = [33.26,33.77,34.60,34.46,35.43,34.80,35.50,36.54,37.37,35.64]
# men_std = [2, 3, 4, 1, 2]
# women_std = [3, 5, 2, 3, 3]
width = 0.35  # 条形图的宽度

fig,ax = plt.subplots()

ax.bar(labels, Rouge_1,width,label='Rouge_1')
ax.bar(labels, Rouge_2,width,label='Rouge_2')
ax.bar(labels,Rouge_l,width,bottom=Rouge_l,label='Rouge_l')
y_range = range(100)
ax.set_yticks(y_range[::10])
ax.set_ylabel('Scores')
ax.set_xlabel('topic_word num')
ax.set_title('Scores on diff topic_word num')
ax.legend()

plt.show()