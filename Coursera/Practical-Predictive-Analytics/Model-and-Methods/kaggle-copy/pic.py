import csv, png
import numpy as np
from sklearn.cluster import AffinityPropagation as AP
from sklearn.cluster import AgglomerativeClustering  as AG
from sklearn.cluster import SpectralClustering  as SC
from sklearn.cluster import DBSCAN
from itertools import cycle
import math

five = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,149,156,179,254,254,201,119,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,147,241,253,253,254,253,253,253,253,245,160,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,31,224,253,253,180,174,175,174,174,174,174,223,247,145,6,0,0,0,0,0,0,0,0,0,0,0,0,7,197,254,253,165,2,0,0,0,0,0,0,12,102,184,16,0,0,0,0,0,0,0,0,0,0,0,0,152,253,254,162,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,235,254,158,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,74,250,253,15,0,0,0,16,20,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,199,253,253,0,0,25,130,235,254,247,145,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,253,253,177,100,219,240,253,253,254,253,253,125,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,193,253,253,254,253,253,200,155,155,238,253,229,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,249,254,241,150,30,0,0,0,215,254,254,58,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,36,39,30,0,0,0,0,0,214,253,234,31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,241,253,183,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,201,253,253,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,114,254,253,154,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,62,254,255,241,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,118,235,253,249,103,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,81,0,102,211,253,253,253,135,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,79,243,234,254,253,253,216,117,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,245,253,254,207,126,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

four = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,220,179,6,0,0,0,0,0,0,0,0,9,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,28,247,17,0,0,0,0,0,0,0,0,27,202,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,242,155,0,0,0,0,0,0,0,0,27,254,63,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,160,207,6,0,0,0,0,0,0,0,27,254,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,127,254,21,0,0,0,0,0,0,0,20,239,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,77,254,21,0,0,0,0,0,0,0,0,195,65,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,70,254,21,0,0,0,0,0,0,0,0,195,142,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,56,251,21,0,0,0,0,0,0,0,0,195,227,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,222,153,5,0,0,0,0,0,0,0,120,240,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,67,251,40,0,0,0,0,0,0,0,94,255,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,234,184,0,0,0,0,0,0,0,19,245,69,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,234,169,0,0,0,0,0,0,0,3,199,182,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,154,205,4,0,0,26,72,128,203,208,254,254,131,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,254,129,113,186,245,251,189,75,56,136,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,216,233,233,159,104,52,0,0,0,38,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,254,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,206,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,186,159,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,209,101,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
zero_orig = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,141,202,254,193,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,165,254,179,163,249,244,72,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,135,254,150,0,0,189,254,243,31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,82,248,209,5,0,0,164,236,254,115,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,211,254,58,0,0,0,0,33,230,212,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,119,254,156,3,0,0,0,0,18,230,254,33,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,212,254,35,0,0,0,0,0,33,254,254,33,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,116,254,154,3,0,0,0,0,0,33,254,254,33,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,124,254,115,0,0,0,0,0,0,160,254,239,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,203,254,35,0,0,0,0,0,0,197,254,178,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,239,221,11,0,0,0,0,0,0,198,255,123,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,238,178,0,0,0,0,0,0,10,219,254,96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,30,249,204,0,0,0,0,0,0,25,235,254,62,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,26,243,204,0,0,0,0,0,0,91,254,248,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,254,204,0,0,0,0,0,67,241,254,133,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,33,254,214,7,0,0,0,50,242,254,194,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5,193,254,78,0,0,19,128,254,195,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,103,254,222,74,143,235,254,228,83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,30,242,254,254,254,254,252,84,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,64,158,200,174,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

def plot(filename, matrix):
	with open(filename, 'wb') as pic:
		w = png.Writer(28, 28, greyscale=True)
		w.write(pic, np.asarray(matrix))

def plot_a_row(filename, array):
    matrix = np.matrix(np.array(array)).reshape(28,28)
    with open(filename, 'wb') as pic:
        w = png.Writer(28, 28, greyscale=True)
        w.write(pic, np.asarray(matrix))

def get_color(label, colors, mapping):
    if None == mapping:
        return label
    return colors[mapping[label]]

def plot_color(filename, X, labels, major_points = None, mapping = None):
    black_array = np.array([[255, 255, 255] * 28] * 28)
    colors = ['b','g','r','c','m','y','k','g','p','a']
    if len(labels) <> len(X):
        tags = [None] * len(X)
    else:
        tags = labels
    for x,y in zip(X, tags):
        color = get_color(y, colors, mapping)
        if y == None:
            black_array[x[1]][x[0]*3+0] = 255
            black_array[x[1]][x[0]*3+1] = 255
            black_array[x[1]][x[0]*3+2] = 255
        elif y >= len(colors):
            black_array[x[1]][x[0]*3+0] = 128
            black_array[x[1]][x[0]*3+1] = 128
            black_array[x[1]][x[0]*3+2] = 128
        elif color == 'r':
            black_array[x[1]][x[0]*3+0] = 255
            black_array[x[1]][x[0]*3+1] = 0
            black_array[x[1]][x[0]*3+2] = 0
        elif color == 'g':
            black_array[x[1]][x[0]*3+0] = 0
            black_array[x[1]][x[0]*3+1] = 255
            black_array[x[1]][x[0]*3+2] = 0
        elif color == 'b':
            black_array[x[1]][x[0]*3+0] = 0
            black_array[x[1]][x[0]*3+1] = 0
            black_array[x[1]][x[0]*3+2] = 255
        elif color == 'c':
            black_array[x[1]][x[0]*3+0] = 0
            black_array[x[1]][x[0]*3+1] = 255
            black_array[x[1]][x[0]*3+2] = 255
        elif color == 'm':
            black_array[x[1]][x[0]*3+0] = 255
            black_array[x[1]][x[0]*3+1] = 0
            black_array[x[1]][x[0]*3+2] = 255
        elif color == 'y':
            black_array[x[1]][x[0]*3+0] = 255
            black_array[x[1]][x[0]*3+1] = 255
            black_array[x[1]][x[0]*3+2] = 0
        elif color == 'k':
            black_array[x[1]][x[0]*3+0] = 255
            black_array[x[1]][x[0]*3+1] = 128
            black_array[x[1]][x[0]*3+2] = 80
        elif color == 'g':
            black_array[x[1]][x[0]*3+0] = 255
            black_array[x[1]][x[0]*3+1] = 215
            black_array[x[1]][x[0]*3+2] = 0
        elif color == 'p':
            black_array[x[1]][x[0]*3+0] = 147
            black_array[x[1]][x[0]*3+1] = 112
            black_array[x[1]][x[0]*3+2] = 219
        elif color == 'a':
            black_array[x[1]][x[0]*3+0] = 128
            black_array[x[1]][x[0]*3+1] = 255
            black_array[x[1]][x[0]*3+2] = 212
        else:
            black_array[x[1]][x[0]*3+0] = 128
            black_array[x[1]][x[0]*3+1] = 128
            black_array[x[1]][x[0]*3+2] = 128
    if None != major_points:
        for point in major_points:
            black_array[point[1]][point[0]*3+0] = 0
            black_array[point[1]][point[0]*3+1] = 0
            black_array[point[1]][point[0]*3+2] = 0
    with open(filename, 'wb') as pic:
        w = png.Writer(28, 28, greyscale=False)
        w.write(pic, black_array)

def row_to_data(array, inverse = False):
    ret = []
    for x in range(0, 28):
        for y in range(0, 28):
            if not(inverse):
                if int(array[x + y*28])>100:
                    ret += [[x,y]]
            else:
                if not(int(array[x + y*28])>100):
                    ret += [[x,y]]
    return ret

def row_to_data2(array, inverse = False):
    ret = []
    for x in range(0, 28):
        for y in range(0, 28):
            #point = array[x + y*28]
            #if (not(inverse) and int(point)>100) or ( inverse and (int(point)<100) ):
            #    ret += [[x,y]]
            # black point
            # reduce noise
            one = False
            two = False
            three = False
            four = False
            five = False
            six = False
            seven = False
            eight = False
            nine = False
            if x>0:
                if int(array[x-1 + y*28])>100: four = True
                if y>0:
                    if int(array[x-1 + (y-1)*28])>100: one = True
                if y<27:
                    if int(array[x-1 + (y+1)*28])>100: seven = True
            if x<27:
                if int(array[x+1 + y*28])>100: six = True
                if y>0:
                    if int(array[x+1 + (y-1)*28])>100: three = True
                if y<27:
                    if int(array[x+1 + (y+1)*28])>100: nine = True
            if y>0:
                if int(array[x + (y-1)*28])>100: two = True
            if y<27:
                if int(array[x + (y+1)*28])>100: eight = True
            # The case
            if int(array[x + y*28])>100: five = True
            f = five or (one and nine) or (two and eight) or (three and seven) or (four and six)
            if not(inverse):
                if f: # white if at least 2 white neibours
                    ret += [[x,y]]
            else:
                if not(f):
                    ret += [[x,y]]
    return ret

def iterate_train(process_pattern, csv_out):
    limit=0
    with open('train.csv') as csvfile:
    #with open('8.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader) # skip header
        for row in reader:
            process_pattern(limit, row, csv_out)
            limit+=1
            if limit > 2000:
                break

def ap_cluster_and_plot(row_number, row):
    filename = str(row_number) + "-" + str(row[0]) + "-color" + ".png"
    #matrix = np.matrix(np.array(row[1:])).reshape(28,28)
    #plot(filename, matrix)
    data_entry = row_to_data(row[1:])
    af = AP(verbose=True, damping=0.5, max_iter=1000).fit(data_entry)
    al = af.labels_
    ac = af.cluster_centers_indices_
    print "Row:",row_number," Digit:",row[0],"Clusters: ", len(ac), ":", len(al)
    plot_color(filename, data_entry, al)

def ag_cluster_and_plot(row_number, row):
    filename = str(row_number) + "-" + str(row[0]) + "-color" + ".png"
    #matrix = np.matrix(np.array(row[1:])).reshape(28,28)
    #plot(filename, matrix)
    data_entry = row_to_data(row[1:])
    ap = AG(linkage='average', n_components=6).fit(data_entry)
    al = ap.labels_
    ac = ap.n_components_
    print "Row:",row_number," Digit:",row[0],"Clusters: ", ac, ":", len(al)
    plot_color(filename, data_entry, al)

def get_major_points(data, labels):
    clusters = split_to_clusters(data, labels)
    major_points = []
    for cluster_name in clusters:
        (p1, p2) = find_corner_points(clusters[cluster_name]);
        pM = find_middle_point(clusters[cluster_name], p1, p2)
        if pM == p1 or pM == p1:
            print "Achtung!"
        major_points += [p1, p2, pM]
    return major_points

def get_clusters_metrcs(data, labels):
    clusters = split_to_clusters(data, labels)
    metrics = {}
    for cluster_name in clusters:
        #(sum_distance, sum_distance_sqr) = cluster_metrics(clusters[cluster_name])
        #metrics[cluster_name] = [sum_distance, sum_distance_sqr]
        #(k1, k2, k3) = cluster_metrics(clusters[cluster_name])
        #metrics[cluster_name] = [k1, k2, k3]
        (p, k, d) = cluster_metrics(clusters[cluster_name])
        metrics[cluster_name] = [p[0], p[1], k, d]
    return metrics

def order_clusters(clusters):
    ret = {}
    d2 = {}
    for cluster_name in clusters:
        d2[cluster_name] = 0
        (p1,p2) = find_corner_points(clusters[cluster_name])
        d2[cluster_name] = min(p1[0]*p1[0]+p1[1]*p1[1],p2[0]*p2[0]+p2[1]*p2[1])
        #p = find_middle_point(clusters[cluster_name],p1,p2)
        #d2[cluster_name] = p[0]*p[0]+p[1]*p[1]
    i = 0
    for cluster_name in sorted(d2, key=lambda c: d2[c]):
        ret[cluster_name] = i
        i+=1
    return ret

def split_to_clusters(data, labels):
    ret = {}
    for data_point, label in zip(data, labels):
        if not label in ret.keys():
            ret[label] = []
        ret[label] += [data_point]
    return ret

def find_corner_points(cluster):
    max_distance = 0
    p1 = cluster[0]
    p2 = cluster[-1]
    for pX in cluster:
        for pY in cluster:
            distance = (pX[0] - pY[0]) * (pX[0] - pY[0]) + (pX[1] - pY[1]) * (pX[1] - pY[1])
            if max_distance < distance:
                max_distance = distance
                p1 = pX
                p2 = pY
    return (p1, p2) if (p1[0]*p1[0] + p1[1]*p1[1]) < (p2[0]*p2[0] + p2[1]*p2[1])else (p2, p1)

def find_middle_point(cluster, p1, p2):
    max_distance = 0 #float('infinity')
    max_p = cluster[0]
    for p in cluster:
        #distance = (p1[0] - p[0]) * (p1[0] - p[0]) + (p1[1] - p[1]) * (p1[1] - p[1]) + (p2[0] - p[0]) * (p2[0] - p[0]) + (p2[1] - p[1]) * (p2[1] - p[1])
        #distance = distance_to_line(p1, p2, p)
        distance = ((p1[0] - p[0]) * (p1[0] - p[0]) + (p1[1] - p[1]) * (p1[1] - p[1])) * ((p2[0] - p[0]) * (p2[0] - p[0]) + (p2[1] - p[1]) * (p2[1] - p[1]))
        if max_distance < distance:
            max_distance = distance
            max_p = p
    return max_p


def distance_to_line(p1, p2, p):
    (pF, pS) = (([float(p1[0]), float(p1[1])],[float(p2[0]),float(p2[1])]) if p1[0] < p2[0] else ([float(p2[0]), float(p2[1])],[float(p1[0]),float(p1[1])]))
    # line: y = k*x + b, p1 and p2 are on the line
    #k = ( float('infinity') if pF[0] == pS[0] else (pS[1] - pF[1]) / (pS[0] - pF[0]))
    # small hack to avoid DivZ and Inf
    if pS[0] == pF[0]:
        pS[0] += 1
    if pS[1] == pF[1]:
        pS[1] += 1
    # end of small hack
    k = (pS[1] - pF[1]) / (pS[0] - pF[0])
    b = pF[1] - k*pF[0]
    # line: yp = -k*px + bp, p on the line
    bp = p[1] + p[0]*k
    # intersection point: x = (bp - b)/(k - (-k)), y = kx + b
    x = (bp -b)/(2*k)
    y = k*x + b
    # distance
    distance = math.sqrt((x - p[0])*(x - p[0]) + (y - p[1])*(y - p[1]))
    # above if k*pX + b < pY
    sign = (1 if k*p[0] + b < p[1] else -1 )
    #print "Line: ",pF," to ",pS," => y = ",k,"*x","+",b
    #print "X-point: ",x,",",y
    #print "Distance: ",sign*distance
    return sign*distance

def get_k(p1,p2):
    pF = list(p1) # true copy
    pS = list(p2) # true copy
    # small hack to avoid DivZ and Inf
    if pS[0] == pF[0]:
        pS[0] = pS[0] + 1
    if pS[1] == pF[1]:
        pS[1] = pS[1] + 1
    # end of small hack
    k = float(pS[1] - pF[1]) / float(pS[0] - pF[0])
    return k

def get_angle(pC, pA, pB):
    dAC2 = (pA[0]-pC[0])*(pA[0]-pC[0]) + (pA[1]-pC[1])*(pA[1]-pC[1])
    dBC2 = (pB[0]-pC[0])*(pB[0]-pC[0]) + (pB[1]-pC[1])*(pB[1]-pC[1])
    dAB2 = (pA[0]-pB[0])*(pA[0]-pB[0]) + (pA[1]-pB[1])*(pA[1]-pB[1])
    if dAC2 == 0 or dBC2 == 0:
        return 0
    return math.acos( (dAC2 + dBC2 - dAB2) / (2*math.sqrt(dAC2) * math.sqrt(dBC2)) )



# +--------------------------
# | M 1   M 2   1 M  2 M
# | 2k-   1k-   k+2  k+1
# |
# | 1k+   2k+   k-1  k-2
# | M 2   M 1   2 M  1 M
# |

def test_get_k():
    print get_k([1,0],[0,1])," ",distance_to_line([1,0],[0,1],[0,0])
    print get_k([0,1],[1,0])," ",distance_to_line([0,1],[1,0],[0,0])
    print get_k([0,0],[1,1])," ",distance_to_line([0,0],[1,1],[1,0])
    print get_k([1,1],[0,0])," ",distance_to_line([1,1],[0,0],[1,0])
    print get_k([0,0],[1,1])," ",distance_to_line([0,0],[1,1],[0,1])
    print get_k([1,1],[0,0])," ",distance_to_line([1,1],[0,0],[0,1])
    print get_k([0,1],[1,0])," ",distance_to_line([0,1],[1,0],[1,1])
    print get_k([1,0],[0,1])," ",distance_to_line([1,0],[0,1],[1,1])
    print "----"
    print get_k([3,0],[0,2])," ",distance_to_line([3,0],[0,2],[1,1])," ",get_angle([3,0],[0,2],[1,1])
    print get_k([0,2],[3,0])," ",distance_to_line([0,2],[3,0],[1,1])," ",get_angle([0,2],[3,0],[1,1])

#test_get_k()

def test_angle():
    print get_angle([2,2], [3,0], [1,3])
    print get_angle([3,3], [1,0], [1,2])
    print get_angle([4,6], [3,3], [2,5])

#test_angle()

def cluster_metrics(cluster):
    p1, p2 = find_corner_points(cluster)
    pM = find_middle_point(cluster, p1, p2)
    a = get_angle(pM, p1, p2)
    s = np.sign(get_k(p1,p2))
    d = distance_to_line(p1, p2, pM)
    return [pM, s*a, d]

def cluster_kd_metrics(cluster):
    p1, p2 = find_corner_points(cluster)
    pM = find_middle_point(cluster, p1, p2)
    k = (-1.0)*get_k(p1,p2)
    d = distance_to_line(p1, p2, pM)
    return [pM, k, d]

def cluster_3k_metrics(cluster):
    p1, p2 = find_corner_points(cluster)
    pM = find_middle_point(cluster, p1, p2)
    return get_k(p1, pM), get_k(p2, pM), get_k(p1, p2)

def cluster_spread_metrics(cluster):
    p1, p2 = find_corner_points(cluster)
    sum_distance = 0
    sum_distance_sqr = 0
    for p in cluster:
        d = distance_to_line(p1, p2, p)
        sum_distance += d
        sum_distance_sqr += (d*d)
    spread = len(cluster)*math.sqrt(float((p2[0] - p1[0])*(p2[0] - p1[0]) + (p2[1] - p1[1])*(p2[1] - p1[1])))
    return sum_distance/spread, math.sqrt(sum_distance_sqr)/spread

def sc_cluster_and_plot(row_number, row, csv_out = None):
    filename = str(row_number) + "-" + str(row[0]) + "-color" + ".png"
    #matrix = np.matrix(np.array(row[1:])).reshape(28,28)
    #plot(filename, matrix)
    data_entry = row_to_data(row[1:])
    #sc = SC(assign_labels='discretize', n_clusters=3).fit(data_entry)
    sc = SC(assign_labels='discretize', affinity='rbf', n_clusters=3).fit(data_entry)
    #sc = SC(n_clusters=1).fit(data_entry)
    al = sc.labels_
    #
    metrics = get_clusters_metrcs(data_entry, al)
    print filename," ",metrics
    dump(csv_out, row_number, row[0], metrics)
    #plot_color(filename, data_entry, al)
    major_points = get_major_points(data_entry, al)
    mapping = order_clusters(split_to_clusters(data_entry, al))
    plot_color(filename, data_entry, al, major_points, mapping)
    print "Row:",row_number," Digit:",row[0],"Mapping: ",mapping, "Maj:", major_points

def sc_cluster(data):
    sc = SC(assign_labels='discretize', affinity='rbf', n_clusters=3).fit(data)
    al = sc.labels_
    metrics = get_clusters_metrcs(data, al)
    return metrics

def dump(csv_out, row_number, digit, metrics):
    #data = metrics.copy().update({'n':row_number, 'label':digit})
    if csv_out != None:
        row = [digit]
        for i in metrics:
            if hasattr(i, '__iter__'):
                for j in i:
                    row.extend([j])
            else:
                row.extend([i])
        csv_out.writerow(row)
    #csv_out.writerow([digit, metrics[0][0], metrics[0][1], metrics[0][2], metrics[0][3], metrics[1][0], metrics[1][1], metrics[1][2], metrics[1][3], metrics[2][0], metrics[2][1], metrics[2][2], metrics[2][3]])
    #csv_out.writerow([row_number, digit, metrics[0][0], metrics[0][1], metrics[1][0], metrics[1][1], metrics[2][0],metrics[2][1]])
    #csv_out.writerow([digit, metrics[0][0], metrics[0][1], metrics[0][2], metrics[1][0], metrics[1][1], metrics[1][2], metrics[2][0],metrics[2][1], metrics[2][2]])
    #csv_out.writerow(
    #                [row_number,
    #                 digit,
    #                 math.log10(abs(metrics[0][0])) if metrics[0][0] != 0 else 0 ,
    #                 math.log10(abs(metrics[0][1])) if metrics[0][1] != 0 else 0 ,
    #                 math.log10(abs(metrics[1][0])) if metrics[1][0] != 0 else 0 ,
    #                 math.log10(abs(metrics[1][1])) if metrics[1][1] != 0 else 0 ,
    #                 math.log10(abs(metrics[2][0])) if metrics[2][0] != 0 else 0 ,
    #                 math.log10(abs(metrics[2][1])) if metrics[2][1] != 0 else 0
    #                                     ])

def enclosed_space_clusters(data_inv):
    #filename = str(row_number) + "-" + str(row[0]) + "-inv" + ".png"
    #data_inv = row_to_data(row[1:], inverse = True)
    dbs = DBSCAN(eps=1, min_samples=1).fit(data_inv)
    clusters = {}
    i = 0
    for j in dbs.labels_:
        if j not in clusters:
            clusters[j] = []
        clusters[j] += [data_inv[i]]
        i+=1
    del_me = []
    for k in clusters: # remove cluster with surrounding space
        if float(len(clusters[k]))>float(len(data_inv)/3):
            del_me += k
    for k in del_me:
        del clusters[k]
    #print len(clusters)
    #plot_color(filename, data_entry, al) #, None, {1:1,2:2,3:3,4:4,5:5})
    return clusters

def get_fill_ratio(data):
    min_x = 28
    max_x = 0
    min_y = 28
    max_y = 0
    for p in data:
        if p[0] > max_x:
            max_x = p[0]
        if p[0] < min_x:
            min_x = p[0]
        if p[1] > max_y:
            max_y = p[1]
        if p[1] < min_y:
            min_y = p[1]
    square = float((max_x-min_x)*(max_y-min_y))
    filled = float(len(data))
    return (filled, square)

def enclosed_points_count(clusters):
    points_cnt = 0
    for cluster in clusters:
        points_cnt += len(clusters[cluster])
    return points_cnt

def calc_features(row_number, row, csv_out = None):
    data = row_to_data(row[1:])
    metrics = sc_cluster(data)
    m = [metrics[0][0], metrics[0][1], metrics[0][2], metrics[0][3], metrics[1][0], metrics[1][1], metrics[1][2], metrics[1][3], metrics[2][0], metrics[2][1], metrics[2][2], metrics[2][3]]
    (filled, square) = get_fill_ratio(data)
    fr = filled/square
    data_inv = row_to_data(row[1:], inverse = True)
    inv_clusters = enclosed_space_clusters(data_inv)
    enclosed_points_cnt = enclosed_points_count(inv_clusters)
    enclosed_ratio = enclosed_points_cnt/square
    dump(csv_out, row_number, row[0], [fr, len(inv_clusters), enclosed_ratio, m])

def create_learning_db():
    with open("dump.csv", 'wb') as f_out:
        csv_out = csv.writer(f_out, delimiter=',')
        csv_out.writerow(['n', 'digit', 'c0sd', 'c0sd2', 'c1sd', 'c1sd2', 'c2sd', 'c2sd2'])
        iterate_train(sc_cluster_and_plot, csv_out)

def k_learning_db():
    with open("dump.csv", 'wb') as f_out:
        csv_out = csv.writer(f_out, delimiter=',')
        csv_out.writerow(['digit', 'fill_rate', 'inv_clusters', 'enclosed_ratio','p0x', 'p0y', 'a0', 'd0', 'p1x', 'p1y', 'a1', 'd1', 'p2x', 'p2y', 'a2', 'd2'])
        iterate_train(calc_features, csv_out)


#print len(five)
#print row_to_data(five)
#plot_a_row('five.png', five)
#plot_color(row_to_data(five),[])
#sc_cluster_and_plot(0, [5] + five)
#cluster_and_plot(0, [4] + four)
#cluster_and_plot(0, [0] + zero)

def read_png(filename):
    a = png.Reader(filename).read()
    ret = []
    for row in list(a[2]):
        i = 0
        for x in row:
            if i%2 == 0: # to skip alpha-channel
                ret += [x]
            i += 1
    return ret

def test_line():
    distance_to_line([1,2],[4,8],[2,2])

def test():
    one = read_png('images/one.png')
    zero = read_png('images/zero.png')
    la = read_png('images/la.png')
    ra = read_png('images/ra.png')
    sc_cluster_and_plot(0, [1] + one)
    sc_cluster_and_plot(0, [0] + zero)
    sc_cluster_and_plot(0, ['L'] + la)
    sc_cluster_and_plot(0, ['R'] + ra)

#test()
#create_learning_db()
k_learning_db()

def find_distant_point(cluster):
    distances = {}
    for p in cluster:
        s = 0
        for p2 in cluster:
            s += (p2[0]-p[0])*(p2[0]-p[0]) + (p2[1]-p[1])*(p2[1]-p[1])
        distances[str(p[0])+","+str(p[1])] = s
    sorted_by_distance = sorted(distances, key=lambda x: distances[x])
    # last by
    print sorted_by_distance[-1],sorted_by_distance[-2]

def find_two_other(cluster, p0):
    pass

class Tree:
    def __init__(self, point):
        pass
    
class Node(Tree):
    pass

#find_distant_point(row_to_data(zero_orig))
#find_distant_point(row_to_data(five))
#print len(row_to_data(zero_orig))

#test()
def print_metrics_3_points(p1, p2, pM):
    print [p1, p2, pM]
    a = get_angle(pM, p1, p2)
    s = np.sign(get_k(p1,p2))
    d = distance_to_line(p1, p2, pM)
    print [pM, s*a, d]

def assert_3_points():
    print_metrics_3_points([17, 8], [20, 19], [12, 15])
    print_metrics_3_points([7, 10], [16, 4], [16, 8])
    print_metrics_3_points([7, 22], [19, 19], [15, 23])
    print "---"
    print_metrics_3_points([15, 9], [18, 17], [12, 15])
    print_metrics_3_points([9, 5], [17, 8], [14, 4])
    print_metrics_3_points([9, 22], [18, 18], [14, 23])
    print "---"
    print_metrics_3_points([9, 18], [21, 21], [13, 24])
    print_metrics_3_points([17, 6], [21, 20], [11, 14])
    print_metrics_3_points([5, 9], [16, 6], [11, 5])

#assert_3_points()
