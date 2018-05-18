#ifndef RPN_LAYER_H
#define RPN_LAYER_H
#include "network.h"
#include "layer.h"

typedef layer rpn_layer;

rpn_layer make_rpn_layer(int feat_stride, int basesize, int min_size, int pre_nms_topN, int post_nms_topN, float nms_thresh, int *scale, float *ratio);
void resize_rpn_layer(rpn_layer *l, network *net);
void forward_rpn_layer(const rpn_layer l, network net);