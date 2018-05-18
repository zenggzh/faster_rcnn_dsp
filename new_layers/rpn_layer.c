#include "rpn_layer.h"
#include <stdio.h>


rpn_layer make_rpn_layer(int feat_stride, int basesize, int min_size, int pre_nms_topN, int post_nms_topN, float nms_thresh, int *scale, float *ratio)
{
	int i;
	rpn_layer l = {0};
	l.type = RPN;

	l.feat_stride = feat_stride;
	l.basesize = basesize;
	l.min_size = min_size;
	l.pre_nms_topN = pre_nms_topN;
	l.post_nms_topN = post_nms_topN;
	l.nms_thresh = nms_thresh;
	l.scale_num = sizeof(scale) / sizeof(int);
	l.ratio_num = sizeof(ratio) / sizeof(float);
	l.scale = calloc(scale_num, sizeof(int));
	l.ratio = calloc(ratio_num, sizeof(float));
	l.anchor = calloc(9*4, sizeof(float));

	l.forward = forward_rpn_layer;

	fprintf(stderr, "rpn  %5d\n", feat_stride);

	return l;
}

//generate base anchor
void generate_anchors(rpn_layer l)
{
	float *base_anchor;
	base_anchor[0] = 0;
	base_anchor[1] = 0;
	base_anchor[2] = l.basesize-1;
	base_anchor[3] = l.basesize-1;
	//enum ratio anchors
	l.anchor = ratio_enum(base_anchor);
	for(int i = 0; i < l.ratio_num; ++i)
	{
		float *tmp = scale_enum(l.anchor[i])
	}
}

void scale_enum(float *anchor, )
{
	float *reform_anchor = whctrs(anchors);
	float x_ctr = reform_anchor[2];
	float y_ctr = reform_anchor[3];
	float w = reform_anchor[0];
	float h = reform_anchor[1];
	for (int i = 0; i < scale_num; ++i)
	{
		float ws = w * scale[i];//the scale[i] is l.scale[i]
		float hs = h * scale[i];
		float *tmp = mkanchor(ws, hs, x_ctr, y_ctr);
		//add tmp to result
	}
	return result;
}

void ratio_enum(float *anchor)
{
	float *reform_anchor = whctrs(anchors);
	float x_ctr = reform_anchor[2];
	float y_ctr = reform_anchor[3];
	float size = reform_anchor[0] * reform_anchor[1];
    for (int i = 0; i < ratio_num; ++i)
    {
        float size_ratios = size / ratios[i];//the ratio[i] is l.ratio[i]
        float ws = round(sqrt(size_ratios));//the function of "round()" and "sqrt()" not include
        float hs = round(ws*ratios[i]);
        vector<float> tmp = mkanchor(ws, hs, x_ctr, y_ctr);//<vector> for C++
        result.push_back(tmp);//push_back() for C++
    }
    return result;
}

float *mkanchor(float w, float h, float x_ctr, float y_ctr)
{
	float *tmp;
    tmp[0] = x_ctr - 0.5*(w - 1);
    tmp[1] = y_ctr - 0.5*(h - 1);
    tmp[2] = x_ctr + 0.5*(w - 1);
    tmp[3] = y_ctr + 0.5*(h - 1);
    return *tmp;
}

float *whctrs(float *anchor)
{
	float *reform_anchor;
	reform_anchor[0] = anchor[2] - anchor[0] + 1;//w
	reform_anchor[1] = anchor[3] - anchor[1] + 1;//h
	reform_anchor[2] = anchor[0] + 0.5 * (w - 1);//x_ctr
	reform_anchor[3] = anchor[1] + 0.5 * (h - 1);//y_ctr
	return *reform_anchor;
}

void forward_rpn_layer(rpn_layer l, network net)
{
	int i = 0;
	generate_anchors();
}