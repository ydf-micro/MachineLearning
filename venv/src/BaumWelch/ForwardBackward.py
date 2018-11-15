import numpy as np

if __name__ == '__main__':
    Ob_state = np.array([0, 1, 0, 0, 1, 0, 1, 1])
    hidden_state = np.array([1, 2, 3])
    init_matxix = np.array([0.2, 0.4, 0.4])
    trans_state = np.array([[0.5, 0.2, 0.3],
                            [0.3, 0.5, 0.2],
                            [0.2, 0.3, 0.5]])
    confu_matrix = np.array([[0.5, 0.5],
                            [0.4, 0.6],
                            [0.7, 0.3]])

    result_F = np.zeros((3, 8))
    result_B = np.zeros((3, 8))

    '''Forward'''
    for i in range(len(Ob_state)):
        if i == 0:
            result_F[:, i] = init_matxix * confu_matrix[:, Ob_state[i]].flatten()
        else:
            result_F[:, i] = np.dot(result_F[:, i-1].flatten(), trans_state) * confu_matrix[:, Ob_state[i]].flatten()

    print(result_F)
    print('\n')

    '''Backward'''
    for i in range(len(Ob_state)-1, -1, -1):
        if i == len(Ob_state)-1:
            result_B[:, i] = 1
        else:
            x = result_B[:, i + 1].flatten() * trans_state * confu_matrix[:, Ob_state[i + 1]].flatten()
            result_B[:, i] = np.sum(x, axis=1)

    print(result_B)

    print('位于t时刻，隐藏状态si的概率\n')
    result = np.sum(result_F[:, 3] * result_B[:, 3])
    print('位于4时刻，隐藏状态s3的概率：', result_B[2, 3] * result_F[2, 3] / result)