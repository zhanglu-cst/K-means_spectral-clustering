import numpy
import random
import math

with open('iris.data', 'r') as f:
    lines = f.read().split('\n')
    lines.pop()
    x = []
    for line in lines:
        n_line = []
        line = line.split()
        for item in line:
            n_line.append(float(item))
        x.append(n_line)
    x = numpy.array(x)
    data = x[:, :-1]
    label = x[:, -1]
    print(data.shape)
    print(label.shape)


def distance_cal(p1, p2):
    dim = len(p1)
    distance = 0
    for i in range(dim):
        distance += (p1[i] - p2[i]) ** 2
    distance = math.sqrt(distance)
    return distance


def K_means(data_input: numpy.ndarray, cluster_number = 3, eps = 1e-8):
    dim = len(data_input[0])
    num_data = len(data_input)
    init_centers_row_ids = random.sample(range(num_data), cluster_number)
    centers = data_input[init_centers_row_ids, :]  # init centers
    print('init centers:\n{}'.format(centers))
    total_iter_number = 0
    last_centers = numpy.zeros((num_data, dim))
    while True:

        points_each_cluster = [[] for i in range(cluster_number)]
        for item_point in data_input:  # assign points to cluster
            min_dis_index = 0
            min_distance = 10000000
            for index_cluster, item_center in enumerate(centers):
                distance = distance_cal(item_point, item_center)
                if (distance < min_distance):
                    min_distance = distance
                    min_dis_index = index_cluster
            points_each_cluster[min_dis_index].append(item_point)

        last_centers = centers.copy()
        # cal new centers
        for index_cluster, points_cur_cluster in enumerate(points_each_cluster):
            new_center = numpy.mean(points_cur_cluster, axis = 0)
            centers[index_cluster] = new_center

        dis_sum = 0
        for point_last, point_cur in zip(last_centers, centers):
            dis_cur = distance_cal(point_last, point_cur)
            dis_sum += dis_cur

        print('current iter number:{}, dis_sum:{}'.format(total_iter_number, dis_sum))
        if (dis_sum < eps):
            return centers, points_each_cluster, total_iter_number

        total_iter_number += 1


def get_error_rate(points_each_cluster, data, label, eps = 1e-8):
    total_num = 0
    error_num = 0
    finded_labels = set()
    for points_cur_cluster in points_each_cluster:
        all_labels_cur_cluster = []
        for item_point in points_cur_cluster:
            flag = False
            for index, point_origin in enumerate(data):
                dis = distance_cal(item_point, point_origin)
                if (dis < eps):
                    all_labels_cur_cluster.append(label[index])
                    flag = True
                    break
            assert flag == True
        # find the most frequent label
        D_cnt = {}
        for assign_label in all_labels_cur_cluster:
            if (assign_label not in D_cnt):
                D_cnt[assign_label] = 1
            else:
                D_cnt[assign_label] += 1
        L_items = list(D_cnt.items())
        L_items.sort(key = lambda x: x[1], reverse = True)
        cur_label = L_items[0][0]

        assert cur_label not in finded_labels
        finded_labels.add(cur_label)
        for item in all_labels_cur_cluster:
            if (item != cur_label):
                error_num += 1
            total_num += 1

    return error_num / total_num

if(__name__=='__main__'):
    centers, points_each_cluster, total_iter_number = K_means(data, cluster_number = 3)
    error_rate = get_error_rate(points_each_cluster, data, label)

    print(centers)

    for point_one_c in points_each_cluster:
        stacked_points_one_c = numpy.stack(point_one_c)
        print(stacked_points_one_c)
        print()

    print(error_rate)
