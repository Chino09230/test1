import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# 生成示例数据
np.random.seed(0)
num_professions = 5
professions = ['人工智能', '工商管理', '财务管理', '电子商务', '大数据管理']
genders = ['男', '女']

# 各专业男女性别比例数据
gender_ratio = pd.DataFrame(
    np.random.randint(30, 70, size=(num_professions, len(genders))),
    index=professions,
    columns=genders
)

# 各专业学习指标数据
learning_metrics = pd.DataFrame({
    '专业': professions,
    '每周学习时长（小时）': np.random.uniform(10, 25, num_professions),
    '作业完成率': np.random.uniform(0.7, 1.0, num_professions),
    '上课出勤率': np.random.uniform(0.7, 1.0, num_professions),
    '期中考试分数': np.random.randint(60, 100, num_professions),
    '期末考试分数': np.random.randint(60, 100, num_professions)
})

# 大数据管理专业专项分析数据
big_data = learning_metrics[learning_metrics['专业'] == '大数据管理'].squeeze()
big_data_scores = np.random.randint(40, 100, 30)

# 绘制各专业男女性别比例柱状图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
gender_ratio.plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('各专业男女性别比例')
axes[0, 0].set_xlabel('专业')
axes[0, 0].set_ylabel('人数')
axes[0, 0].tick_params(axis='x', rotation=45)

# 绘制各专业学习指标对比折线图
learning_metrics.set_index('专业')[['每周学习时长（小时）', '作业完成率', '上课出勤率']].plot(kind='line', ax=axes[0, 1])
axes[0, 1].set_title('各专业学习指标对比')
axes[0, 1].set_xlabel('专业')
axes[0, 1].set_ylabel('指标值')
axes[0, 1].tick_params(axis='x', rotation=45)

# 绘制各专业出勤率分析色块图（热力图）
sns.heatmap(learning_metrics.set_index('专业')[['上课出勤率']].T, annot=True, cmap='YlGnBu', ax=axes[1, 0])
axes[1, 0].set_title('各专业出勤率分析')
axes[1, 0].set_ylabel('')
axes[1, 0].tick_params(axis='x', rotation=45)

# 大数据管理专业专项分析
axes[1, 1].bar(['平均成绩', '作业完成率', '出勤率', '平均学习时长'],
               [big_data['期末考试分数'], big_data['作业完成率'], big_data['上课出勤率'], big_data['每周学习时长（小时）']])
axes[1, 1].set_title('大数据管理专业专项分析 - 关键指标')
axes[1, 1].tick_params(axis='x', rotation=45)

# 绘制大数据管理专业成绩分布柱状图
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.histplot(big_data_scores, bins=10, kde=False, ax=ax2)
ax2.set_title('大数据管理专业成绩分布')
ax2.set_xlabel('成绩')
ax2.set_ylabel('人数')

plt.tight_layout()
plt.show()
