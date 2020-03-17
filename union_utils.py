def iou_box(b1, b2):
    xmin = max(b1[0], b2[0])
    xmax = min(b1[2], b2[2])
    ymin = max(b1[1], b2[1])
    ymax = min(b1[3], b2[3])
    if xmin >= xmax or ymin >= ymax:
        return 0
    union_area = (xmax-xmin)*(ymax-ymin)
    return union_area/((b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - union_area)


def merge_yl_ct_dets(yl_dets, ct_dets):
    final_dets = []
    for i, yl_det in enumerate(yl_dets):
        best_score = 0
        for j, ct_det in enumerate(ct_dets):
            if iou_box(yl_det, ct_det) > best_score:
                best_score = iou_box(yl_det, ct_det)
        if best_score > 0.3:
            final_dets.append(yl_det)
    return final_dets


def parse_data_label(label_file, height, width):
    f = open(label_file)
    items = f.readlines()
    f.close()
    final_labels = []
    for item in items:
        elements = item.split(' ')
        id = int(elements[0])
        x = float(elements[1])
        y = float(elements[2])
        w = float(elements[3])
        h = float(elements[4])
        final_labels.append([(x-w/2)*width, (y-h/2)*height,
                             (x+w/2) * width, (y+h/2)*height, 1.0, id])
    return final_labels


def update_stastics(statics, final_labels, final_dets):
    for final_label in final_labels:
        statics[final_label[-1]]['ground_nums'] += 1
        best_score = 0
        for final_det in final_dets:
            if iou_box(final_label, final_det) > best_score and final_label[-1] == final_det[-1]:
                best_score = iou_box(final_label, final_det)
        if best_score > 0.4:
            statics[final_label[-1]]['recall_ground_nums'] += 1

    for final_det in final_dets:
        statics[final_det[-1]]['detection_nums'] += 1
        best_score = 0
        for final_label in final_labels:
            if iou_box(final_label, final_det) > best_score and final_label[-1] == final_det[-1]:
                best_score = iou_box(final_label, final_det)
        if best_score > 0.4:
            statics[final_det[-1]]['correct_detection_nums'] += 1
    return statics
