import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.formula.api import ols
from scipy.stats import f_oneway

# 方差分析与事后多重比较

<<<<<<< HEAD
# 创建数据 合并版本 手动合并
>>>>>>> second_branch

np.random.seed(0)  # 设置随机种子以保证结果可复现  

group1 = np.random.normal(loc=5, scale=1, size=10)  # 只接受营养液  

group2 = np.random.normal(loc=4, scale=1, size=10)  # 接受营养液并服用成分A  

group3 = np.random.normal(loc=3, scale=1, size=10)  # 接受营养液并服用成分B  

group4 = np.random.normal(loc=2, scale=1, size=10)  # 接受营养液并服用成分A和B  

groups = [group1, group2, group3, group4]  

group_names = ['只接受营养液', '接受营养液并服用成分A', '接受营养液并服用成分B', '接受营养液并服用成分A和B']  

# 执行ANOVA  
F_stat, p_value = f_oneway(*groups)   

# 记录是否服用A
a=[0]*10+[1]*10+[0]*10+[1]*10
# 记录是否服用B
b=[0]*20+[1]*20
groups=np.array(groups).flatten()
data={'A':a,'B':b,'groups':groups}
data=pd.DataFrame(data)

# 创建方差分析模型
model = ols('groups ~ A + B + A*B', data=data).fit()
# 分析方差分析模型
anova_results = sm.stats.anova_lm(model, typ=2)
print(anova_results)


# 事后多重检验
import numpy as np  
from scipy.stats import f_oneway
from scipy.stats import tukey_hsd
# 创建数据  
group1 = np.random.normal(loc=5, scale=1, size=10)  # 只接受营养液  
group2 = np.random.normal(loc=4, scale=1, size=10)  # 接受营养液并服用成分A  
group3 = np.random.normal(loc=3, scale=1, size=10)  # 接受营养液并服用成分B  
group4 = np.random.normal(loc=2, scale=1, size=10)  # 接受营养液并服用成分A和B  
groups = [group1, group2, group3, group4]  
group_names = ['只接受营养液', '接受营养液并服用成分A', '接受营养液并服用成分B', '接受营养液并服用成分A和B']  
# 执行ANOVA  
F_stat, p_value = f_oneway(*groups)   
# TurkeyHSD法进行事后多重比较
# 进行事后多重比较  
mc_result = tukey_hsd(group1,group2,group3,group4)  
# 输出结果  
print(mc_result)

import numpy as np  

import statsmodels.api as sm  

import pandas as pd  

# 创建一些模拟数据  

np.random.seed(0)  

X = np.random.rand(100, 3)  # 100个样本，每个样本有3个特征  

y = X[:, 0] + 2 * X[:, 1] + np.random.rand(100)  # 因变量由前两个特征线性生成，并加入一些噪声  

# 将数据转换为Pandas DataFrame格式  

df = pd.DataFrame(X)  

df['y'] = y   

# 使用statsmodels进行线性回归  

X = sm.add_constant(df[df.columns[:-1]])  # 添加常数项作为截距项  

model = sm.OLS(df['y'], X)  

results = model.fit()  

# 输出回归结果  

print(results.summary())
# 我在这里添加了一句注释
