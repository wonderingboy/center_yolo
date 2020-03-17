def iou_box(b1, b2):
    xmin = max(b1[0], b2[0])
    xmax = min(b1[2], b2[2])
    ymin = max(b1[1], b2[1])
    ymax = min(b1[3], b2[3])
    if xmin >= xmax or ymin >= ymax:
        return 0
    union_area = (xmax-xmin)*(ymax-ymin)
    return union_area/((xmax-xmin)*(ymax-ymin) - union_area)


def merge_yl_ct_dets(yl_dets, ct_dets):
    for yl_det in yl_dets:
        for ct_det in ct_dets:
            if
