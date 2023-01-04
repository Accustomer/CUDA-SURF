#pragma once


namespace surf
{
    /* Structure of SURF feature point */
    struct SurfPoint
    {
        // x, y value of the interest point
        float x = -1;
        float y = -1;
        // detected scale
        float scale = 1;
        int o = 0;
        // strength of the interest point
        float strength = 0;
        // sign of Laplacian
        int laplace = 1;
        // orientation
        float ori = 0;
        // Match score
        float score = 0;
        // Matched index
        int match = -1;
        // X coordinate of matched point
        float match_x = 0;
        // Y coordinate of matched point
        float match_y = 0;
        // Second max score vs. Max score
        float ambiguity = 0;
    };


    /* Structure of SURF matching data */
    struct SurfData
    {
        int num_pts;         // Number of available Sift points
        int max_pts;         // Number of allocated Sift points
        SurfPoint* h_data;   // Host (CPU) data
        SurfPoint* d_data;   // Device (GPU) data
    };


    struct SurfParam
    {
        // threshold for interest point detection
        float thresh;
        // initial lobe size for the second derivative in one direction
        int init_lobe;
        // Set this flag "true" to double the image size
        bool doubled;
        // number of scales
        int max_scale;
        // Number of analysed octaves (default 4)
        int noctaves;
        // Initial sampling step (default 2)
        int sampling;
        // Coordinates scale factor affected by "double"
        float divisor;
        // Upright SURF or rotation invaraiant
        bool upright;
        // If the extended flag is turned on, SURF 128 is used
        bool extend;
        // Spatial size of the descriptor window (default 4)
        int desc_wsz;
        // The factor for sampling step in different scale
        int mag_factor;
        // The size of orientations used for feature extraction
        int orient_size;
        // The number of features
        int nfeatures;
    };
}


