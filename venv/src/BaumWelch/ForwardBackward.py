import numpy as np

def initQuintet():
    Ob_state = np.array([0, 1, 0, 0, 1, 0, 1, 1])

    hidden_state = np.array([1, 2, 3])

    init_matrix = np.array([0.2, 0.4, 0.4])

    trans_state = np.array([[0.5, 0.2, 0.3],
                            [0.3, 0.5, 0.2],
                            [0.2, 0.3, 0.5]])

    confu_matrix = np.array([[0.5, 0.5],
                             [0.4, 0.6],
                             [0.7, 0.3]])

    return Ob_state, hidden_state, init_matrix, trans_state, confu_matrix

def Forward(result_F, init_matrix, trans_state, confu_matrix):
    '''Forward'''
    for i in range(len(Ob_state)):
        if i == 0:
            result_F[:, i] = init_matrix * confu_matrix[:, Ob_state[i]].flatten()
        else:
            result_F[:, i] = np.dot(result_F[:, i - 1].flatten(), trans_state) * confu_matrix[:, Ob_state[i]].flatten()

    return result_F

def Backwward(result_B, trans_state, confu_matrix):
    '''Backward'''
    for i in range(len(Ob_state) - 1, -1, -1):
        if i == len(Ob_state) - 1:
            result_B[:, i] = 1
        else:
            x = result_B[:, i + 1].flatten() * trans_state * confu_matrix[:, Ob_state[i + 1]].flatten()
            result_B[:, i] = np.sum(x, axis=1)

    return result_B

if __name__ == '__main__':
    Ob_state, hidden_state, init_matrix, trans_state, confu_matrix = initQuintet()

    result_F = np.zeros((len(hidden_state), len(Ob_state)))
    result_B = np.zeros((len(hidden_state), len(Ob_state)))

    result_F = Forward(result_F, init_matrix, trans_state, confu_matrix)
    result_B = Backwward(result_B, trans_state, confu_matrix)

    print(result_F, '\n\n', result_B)

    print('给定观察序列O及隐马尔科夫模型lamda，定义t时刻位于隐藏状态Si的概率变量为\n')
    gam_result = np.sum(result_F[:, 3] * result_B[:, 3])
    gamma = result_B[2, 3] * result_F[2, 3] / gam_result
    print('位于4时刻，隐藏状态s3的概率：', gamma)

    print('给定观察序列O及隐马尔科夫模型lamda，定义t时刻位于隐藏状态Si及t+1时刻位于隐藏状态Sj的概率变量为\n')
    xi_result = 0
    for i in range(0, 3):
        for j in range(0, 3):
            xi_result += result_F[i, 4] * trans_state[i, j] * confu_matrix[j, Ob_state[5]] * result_B[j, 5]
    xi = result_F[1, 4] * trans_state[1, 0] * confu_matrix[0, 0] * result_B[0, 5] / xi_result
    print('定义5时刻位于隐藏状态S2及6时刻位于隐藏状态S1的概率变量:', xi)