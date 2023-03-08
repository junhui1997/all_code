def calculate_edit_score(pred_segmentation, true_segmentation):
    """
    Calculates the edit distance and edit score between predicted segmentation and true segmentation.

    Args:
        pred_segmentation (list): predicted segmentation
        true_segmentation (list): true segmentation

    Returns:
        tuple: a tuple containing the edit distance and edit score
    """

    m, n = len(pred_segmentation), len(true_segmentation)

    # create a table to store the edit distances
    table = [[0 for j in range(n+1)] for i in range(m+1)]

    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                table[i][j] = j
            elif j == 0:
                table[i][j] = i
            elif pred_segmentation[i-1] == true_segmentation[j-1]:
                table[i][j] = table[i-1][j-1]
            else:
                table[i][j] = 1 + min(table[i][j-1], table[i-1][j], table[i-1][j-1])

    edit_distance = table[m][n]
    edit_score = 1.0 - float(edit_distance) / max(len(pred_segmentation), len(true_segmentation))

    return edit_distance, edit_score

def flatten_list(arrs):
    """
    将包含多个 NumPy 数组的列表展平为一个一维列表
    """
    flattened = []
    for arr in arrs:
        flattened.extend(arr.flatten().tolist())
    return flattened