#ifndef GAIT_API_H
#define GAIT_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GAIT_API_VERSION_MAJOR 1
#define GAIT_API_VERSION_MINOR 0

typedef enum gait_status_v1 {
    GAIT_OK = 0,
    GAIT_ERR_INVALID_ARGUMENT = 1,
    GAIT_ERR_INVALID_SHAPE = 2,
    GAIT_ERR_MISSING_REQUIRED_SIGNAL = 3,
    GAIT_ERR_INTERNAL = 4,
    GAIT_ERR_UNSUPPORTED_METHOD = 5
} gait_status_v1;

typedef enum gait_method_id_v1 {
    GAIT_METHOD_ZENI = 0,
    GAIT_METHOD_OCONNOR = 1,
    GAIT_METHOD_HRELJAC = 2,
    GAIT_METHOD_MICKELBOROUGH = 3,
    GAIT_METHOD_GHOUSSAYNI = 4,
    GAIT_METHOD_DGEI = 5,
    GAIT_METHOD_VANCANNEYT = 6,
    GAIT_METHOD_BAYESIAN_BIS = 7,
    GAIT_METHOD_DEEPEVENT = 8
} gait_method_id_v1;

typedef struct gait_trial_v1 {
    uint32_t n_frames;
    double fps_hz; /* sampling frequency in Hz, must be > 0 */

    /* Optional marker trajectories (flattened XYZ triples by frame). */
    const double* left_knee_xyz;
    const double* right_knee_xyz;
    const double* left_ankle_xyz;
    const double* right_ankle_xyz;

    const double* left_heel_xyz;
    const double* right_heel_xyz;
    const double* left_toe_xyz;
    const double* right_toe_xyz;

    /* Optional angle trajectories (degrees, one value per frame). */
    const double* left_hip_deg;
    const double* right_hip_deg;
    const double* left_knee_deg;
    const double* right_knee_deg;
    const double* left_ankle_deg;
    const double* right_ankle_deg;

    /* Optional axis metadata; set to -1 when unknown. */
    int32_t ap_axis;
    int32_t vertical_axis;
    int32_t walking_direction_sign;
    int32_t valid_start_frame;
    int32_t valid_end_frame;
} gait_trial_v1;

typedef struct gait_event_v1 {
    int32_t frame0;
    int32_t frame1;
    double time_s;
    int32_t event_type; /* 0=heel_strike, 1=toe_off */
    int32_t side;       /* 0=left, 1=right */
    double confidence;
} gait_event_v1;

typedef struct gait_cycle_v1 {
    int32_t cycle_id;
    int32_t side;            /* 0=left, 1=right */
    int32_t start_frame0;
    int32_t toe_off_frame0;  /* -1 if missing */
    int32_t end_frame0;
    double start_time_s;
    double toe_off_time_s;   /* NaN if missing */
    double end_time_s;
    double duration_s;
    double stance_duration_s; /* NaN if missing */
    double swing_duration_s;  /* NaN if missing */
    double stance_percentage; /* NaN if missing */
    double swing_percentage;  /* NaN if missing */
} gait_cycle_v1;

typedef struct gait_result_v1 {
    /* Outputs are heap-owned by the API; release with gait_free_result_v1(). */
    gait_event_v1* events;
    uint32_t n_events;
    gait_cycle_v1* cycles;
    uint32_t n_cycles;
    char** warnings;
    uint32_t n_warnings;
} gait_result_v1;

typedef struct gait_options_bayesian_bis_v1 {
    uint32_t smoothing_window;
    double min_crossing_distance_s;
    double rhythm_sigma_ratio;
} gait_options_bayesian_bis_v1;

/* Generic options envelope for method-specific structs. */
typedef struct gait_method_options_v1 {
    const void* ptr;
    uint32_t size_bytes;
} gait_method_options_v1;

uint32_t gait_api_version_major(void);
uint32_t gait_api_version_minor(void);

gait_status_v1 gait_get_available_methods_v1(
    gait_method_id_v1* out_methods,
    uint32_t max_methods,
    uint32_t* out_count
);

gait_status_v1 gait_detect_v1(
    gait_method_id_v1 method_id,
    const gait_trial_v1* trial,
    const gait_method_options_v1* method_options,
    gait_result_v1* out_result
);

void gait_free_result_v1(gait_result_v1* result);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* GAIT_API_H */
