# -- coding:UTF-8 --
import math as m

def calcConditionalEnt(data):
# =============================================================================
#     # 先计算条件熵的每一项
#     def calcSingleEnt(p_xy, p_x):
#         return p_xy * math.log(p_x / p_xy) if p_xy != 0 else 0
# =============================================================================

    ConditionEnt = 0
    for i in range(len(data)):
#        colsum = map(sum, zip(*data))  # 各列求和
        for j in range(len(data[1])):
            if data[i][j] > 0:
                ConditionEnt += data[i][j] * m.log(data[i][j])

    return ConditionEnt